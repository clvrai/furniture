from collections import OrderedDict

import numpy as np

import env.transform_utils as T
from env.furniture_sawyer import FurnitureSawyerEnv
from env.models import background_names, furniture_name2id, furniture_xmls
from env.transform_utils import euler_to_quat
from util.logger import logger

# from math import pi, sin


class FurnitureSawyerPushEnv(FurnitureSawyerEnv):
    """
    Sawyer environment for placing a block onto table.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.furniture_id = furniture_name2id["placeblock"]

        super().__init__(config)

        self._success_rew = config.success_rew
        self._ctrl_penalty_coeff = config.ctrl_penalty_coeff
        self._obj_to_point_coeff = config.obj_to_point_coeff
        self._reset_obj_to_point_coeff = config.reset_obj_to_point_coeff
        self._rand_robot_start_range = config.rand_robot_start_range
        self._rand_block_range = config.rand_block_range
        self._rand_block_rotation_range = config.rand_block_rotation_range
        self._robot_start_pos_threshold = config.robot_start_pos_threshold
        self._start_pos_threshold = config.start_pos_threshold
        self._goal_pos_threshold = config.goal_pos_threshold
        self._sparse_forward_rew = config.sparse_forward_rew
        self._sparse_remove_rew = config.use_aot or config.sparse_remove_rew
        self._reversible_state_type = config.reversible_state_type

        self._gravity_compensation = 1
        self._goal_type = config.goal_type
        self._subtask_part1 = 0  # get block ob
        self._task = "forward"  # ["forward", "reverse"]

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        # always close gripper and don't connect
        a = a.copy()
        if self._control_type == "ik":
            a = np.concatenate([a, np.zeros(4)])  # add empty dz, rotation
        a = np.concatenate([a, [1, 0]])
        ob, _, _, _ = super(FurnitureSawyerEnv, self)._step(a)
        rew_fn = self._push_reward if self._task == "forward" else self._reset_reward
        reward, info = rew_fn(ob, a)
        done = False
        if self._success:
            done = True
        info["episode_success"] = int(self._success)
        info["reward"] = reward
        return ob, reward, done, info

    def reset(
        self, is_train=True, record=False, furniture_id=None, background=None,
    ):
        # clear previous demos
        if self._record_demo:
            self._demo.reset()

        # reset robot and objects
        self._reset(furniture_id, background)

        self._task = "forward"
        self._success = False
        obj_pos = self._get_pos("1_block_l")[:2]
        dist_to_start = np.linalg.norm(self._start_pose[:2] - obj_pos)
        self._prev_reset_dist = dist_to_start
        dist_to_goal = np.linalg.norm(self._goal_pose[:2] - obj_pos)
        self._prev_push_dist = dist_to_goal

        # reset mujoco viewer
        if self._render_mode == "human" and not self._unity:
            self._viewer = self._get_viewer()
        self._after_reset()

        ob = self._get_obs()
        if self._record_demo:
            self._demo.add(ob=ob)
        return ob

    def _reset(self, furniture_id=None, background=None):
        """
        Initialize robot at starting point of demonstration.
        Take a random step to increase diversity of starting states.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        if self._config.furniture_name == "Random":
            furniture_id = self._rng.randint(len(furniture_xmls))
        if self._furniture_id is None or (
            self._furniture_id != furniture_id and furniture_id is not None
        ):
            # construct mujoco xml for furniture_id
            if furniture_id is None:
                self._furniture_id = self._config.furniture_id
            else:
                self._furniture_id = furniture_id
            self._reset_internal()

        # reset simulation data and clear buffers
        self.sim.reset()

        # store robot's contype, conaffinity (search MuJoCo XML API for details)
        # disable robot collision
        robot_col = {}
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name not in self._object_names and self.mujoco_robot.is_robot_part(
                geom_name
            ):
                robot_col[geom_name] = (
                    self.sim.model.geom_contype[geom_id],
                    self.sim.model.geom_conaffinity[geom_id],
                )
                self.sim.model.geom_contype[geom_id] = 0
                self.sim.model.geom_conaffinity[geom_id] = 0

        # initialize collision for non-mesh geoms
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name in self._object_names and "collision" in geom_name:
                self.sim.model.geom_contype[geom_id] = 1
                self.sim.model.geom_conaffinity[geom_id] = 1

        # initialize group
        self._object_group = list(range(len(self._object_names)))

        # initialize member variables
        self._connect_step = 0
        self._connected_sites = set()
        self._connected_body1 = None
        self._connected_body1_pos = None
        self._connected_body1_quat = None
        self._num_connected = 0
        if self._agent_type == "Cursor":
            self._cursor_selected = [None, None]

        self._do_simulation(None)
        # stablize furniture pieces
        for _ in range(100):
            for obj_name in self._object_names:
                self._stop_object(obj_name, gravity=0)
            self.sim.forward()
            self.sim.step()

        logger.debug("*** furniture initialization ***")

        # add random noise to block position and rotation
        self._start_pose = np.array([0.01044, -0.028, 0.025, 0.5, 0.5, 0.5, 0.5])
        # set goal position to 1 cm away from block
        self._goal_pose = self._start_pose.copy()
        self._goal_pose[:3] += [0, -0.1, 0]

        block_pose = self._start_pose.copy()
        r = self._rand_block_range
        pos_offset = np.zeros(3)
        pos_offset[0] = self._rng.uniform(-r, r)  # left and right
        # pos_offset[1] = self._rng.uniform(-r, 0)  # up and down
        block_pose[:3] += pos_offset
        r = self._rand_block_rotation_range
        block_rot = [90, 90, 0]
        block_rot[2] += self._rng.uniform(-r, r)
        # print(f"reset angle is {block_rot}")
        block_quat = euler_to_quat(block_rot)
        # print(f"reset block quat: {block_quat}")
        block_pose[3:] = block_quat
        # initialize the robot and block to initial demonstraiton state
        self._init_qpos = {
            "qpos": [
                -0.41392622,
                -0.31336937,
                0.2903396,
                1.45706689,
                -0.60448083,
                0.52339887,
                2.07347478,
            ],
            "l_gripper": [-0.01965996, 0.01963994],
            "1_block_l": block_pose,
        }

        pos_init = []
        quat_init = []

        for body in self._object_names:
            qpos = self._init_qpos[body]
            pos_init.append(qpos[:3])
            quat_init.append(qpos[3:])
        if (
            "l_gripper" in self._init_qpos
            and "r_gripper" not in self._init_qpos
            and "qpos" in self._init_qpos
        ):
            self.sim.data.qpos[self._ref_joint_pos_indexes] = self._init_qpos["qpos"]
            self.sim.data.qpos[self._ref_gripper_joint_pos_indexes] = self._init_qpos[
                "l_gripper"
            ]

        # enable robot collision
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name not in self._object_names and self.mujoco_robot.is_robot_part(
                geom_name
            ):
                contype, conaffinity = robot_col[geom_name]
                self.sim.model.geom_contype[geom_id] = contype
                self.sim.model.geom_conaffinity[geom_id] = conaffinity

        # set furniture positions
        for i, body in enumerate(self._object_names):
            logger.debug(f"{body} {pos_init[i]} {quat_init[i]}")
            self._set_qpos(body, pos_init[i], quat_init[i])

        self.sim.forward()

        # store qpos of furniture and robot
        if self._record_demo:
            self._store_qpos()

        if self._agent_type in ["Sawyer", "Panda", "Jaco", "Baxter"]:
            self._initial_right_hand_quat = self._right_hand_quat
            if self._control_type == "ik":
                # set up ik controller
                self._controller.sync_state()

        # set next subtask
        self._get_next_subtask()

        # set object positions in unity
        if self._unity:
            if background is None and self._background is None:
                background = self._config.background
            if self._config.background == "Random":
                background = self._rng.choice(background_names)
            if background and background != self._background:
                self._background = background
                self._unity.set_background(background)

        # take some random step away from starting state
        self._robot_start_pose = self._get_cursor_pos().copy()
        action = np.zeros((8,))
        r = self._rand_robot_start_range
        # add left and right noise
        action[0] = self._rng.uniform(-r, r)
        action[6] = 1  # keep gripper closed
        self._step_continuous(action)
        # eef_pos = self._get_cursor_pos()
        # dist = np.linalg.norm(self._robot_start_pose - eef_pos)
        # print(dist)

    def begin_forward(self):
        """
        Switch to forward mode. Init forward reward
        """
        self._task = "forward"
        self._success = False
        obj_pos = self._get_pos("1_block_l")[:2]
        dist_to_start = np.linalg.norm(self._start_pose[:2] - obj_pos)
        self._prev_reset_dist = dist_to_start
        dist_to_goal = np.linalg.norm(self._goal_pose[:2] - obj_pos)
        self._prev_push_dist = dist_to_goal

    def begin_reset(self):
        """
        Switch to reset mode. Init forward reward
        """
        self._task = "reset"
        self._success = False
        obj_pos = self._get_pos("1_block_l")[:2]
        dist_to_start = np.linalg.norm(self._start_pose[:2] - obj_pos)
        self._prev_reset_dist = dist_to_start
        dist_to_goal = np.linalg.norm(self._goal_pose[:2] - obj_pos)
        self._prev_push_dist = dist_to_goal

    def reset_success(self):
        ob = self._get_obs()
        r = self._robot_success(ob, self._robot_start_pose, self._robot_start_pos_threshold)
        o = self._object_success(ob, self._start_pose, self._start_pos_threshold)
        return r and o

    def _robot_success(self, ob, goal, threshold) -> bool:
        eef_pos = ob["robot_ob"][:2]
        start_pos = goal[:2]
        return np.linalg.norm(eef_pos - start_pos) < threshold

    def _object_success(self, ob, goal, threshold) -> bool:
        """
        Checks if block pose is close
        to the goal poses
        Ob is format from get_goal
        Goal is format from demo['goal']
        """
        obj = ob["object_ob"]
        object_pos = obj[:2]
        goal_object_pos = goal[:2]
        obj_pos_success = np.linalg.norm(object_pos - goal_object_pos) < threshold
        return obj_pos_success

    def _ctrl_reward(self, action):
        if self._config.control_type == "ik":
            a = np.linalg.norm(action[:6])
        elif self._config.control_type == "impedance":
            a = np.linalg.norm(action[:7])
        ctrl_penalty = -self._ctrl_penalty_coeff * a
        return ctrl_penalty

    def _push_reward(self, ob, action):
        """
        Ob is observation dictionary
        """
        info = {}
        obj_pos = ob["object_ob"][:2]  # xy pos
        # obj_quat = ob["1_block_l"][2:]
        dist_to_goal = np.linalg.norm(self._goal_pose[:2] - obj_pos)
        dist_diff = self._prev_push_dist - dist_to_goal
        obj_to_goal_reward = dist_diff * self._obj_to_point_coeff

        self._success = self._object_success(
            ob, self._goal_pose, self._goal_pos_threshold
        )
        control_reward = self._ctrl_reward(action)
        success_reward = 0
        if self._success:
            success_reward = self._success_rew

        if self._sparse_forward_rew:
            push_reward = control_reward + success_reward
        else:
            push_reward = obj_to_goal_reward + control_reward + success_reward

        info["dist_to_goal"] = dist_to_goal
        info["success_rew"] = success_reward
        info["control_rew"] = control_reward
        info["block_to_goal_rew"] = obj_to_goal_reward

        return push_reward, info

    def _reset_reward(self, ob, action):
        """
        Ob is observation dictionary
        """
        info = {}
        obj_pos = ob["object_ob"][:2]  # xy pos
        # obj_quat = ob["1_block_l"][2:]
        dist_to_start = np.linalg.norm(self._start_pose[:2] - obj_pos)
        dist_diff = self._prev_reset_dist - dist_to_start
        self._prev_reset_dist = dist_to_start
        obj_to_start_reward = dist_diff * self._reset_obj_to_point_coeff

        obj_succ = self._object_success(ob, self._start_pose, self._start_pos_threshold)
        robot_succ = self._robot_success(
            ob, self._robot_start_pose, self._robot_start_pos_threshold
        )
        self._success = obj_succ and robot_succ
        control_reward = self._ctrl_reward(action)
        success_reward = 0
        if self._success:
            success_reward = self._success_rew
            if self._config.use_aot:
                success_reward = self._config.aot_succ_rew

        if self._sparse_remove_rew:
            reset_reward = control_reward + success_reward
        else:
            reset_reward = obj_to_start_reward + control_reward + success_reward

        info["dist_to_start"] = dist_to_start
        info["success_rew"] = success_reward
        info["control_rew"] = control_reward
        info["block_to_start_rew"] = obj_to_start_reward
        info["object_success"] = int(obj_succ)
        info["robot_success"] = int(robot_succ)

        return reset_reward, info

    def _get_obs(self) -> dict:
        state = OrderedDict()
        # visual obs
        if self._visual_ob:
            camera_obs, depth_obs = self.render("rgbd_array")
            state["camera_ob"] = camera_obs
            if depth_obs is not None:
                state["depth_ob"] = depth_obs

        if self._segmentation_ob:
            segmentation_obs = self.render("segmentation")
            state["segmentation_ob"] = segmentation_obs

        # object states
        if self._object_ob:
            block_pos = self._get_pos("1_block_l")
            block_quat = self._get_quat("1_block_l")
            # # print(f"get obs block quat: {block_quat}")
            # give only x,y coordinates
            pos = block_pos[:2]
            # TODO: figure out rotation axis
            # # give only sin(4 * rz) rotation
            # # block_rot = np.array(T.euler_from_quaternion(block_quat, "rxyz"))
            # # convert block rotation
            # # print(f"get obs angle is {block_rot * 180/pi}")
            # rz = T.euler_from_quaternion(block_quat)[2]
            # # print(f"Z rotation is {rz * 180/pi} degrees")
            # z = sin(rz)
            # print(f"sin(z) value: {z}")
            state["object_ob"] = np.concatenate([pos, block_quat])
        # proprioceptive features
        if self._robot_ob:
            robot_states = OrderedDict()
            if self._control_type == "impedance":
                robot_states["joint_pos"] = np.array(
                    [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
                )
                robot_states["joint_vel"] = np.array(
                    [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
                )
                robot_states["gripper_qpos"] = np.array(
                    [self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes]
                )
                robot_states["gripper_qvel"] = np.array(
                    [self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes]
                )
            # IK observation only sees eef
            elif self._control_type == "ik":
                # gripper_qpos = [
                #     self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes
                # ]
                # robot_states["gripper_dis"] = np.array(
                #     [abs(gripper_qpos[0] - gripper_qpos[1])]
                # )
                robot_states["eef_pos"] = np.array(
                    self.sim.data.site_xpos[self.eef_site_id]
                )
                robot_states["eef_velp"] = np.array(
                    self.sim.data.site_xvelp[self.eef_site_id]
                )  # 3-dim
                robot_states["eef_velr"] = self.sim.data.site_xvelr[
                    self.eef_site_id
                ]  # 3-dim
                # robot_states["eef_quat"] = T.convert_quat(
                #     self.sim.data.get_body_xquat("right_hand"), to="xyzw"
                # )
            state["robot_ob"] = np.concatenate(
                [x.ravel() for _, x in robot_states.items()]
            )
        return state

    @property
    def observation_space(self):
        """
        Returns the observation space.
        """
        ob_space = OrderedDict()
        if self._robot_ob:
            if self._control_type == "impedance":
                # ob_space["robot_ob"] = [32]
                ob_space["robot_ob"] = [18]
            elif self._control_type == "ik":
                # ob_space["robot_ob"] = [
                #     3 + 4 + 3 + 3 + 1
                # ]  # pos, quat, vel, rot_vel, gripper
                ob_space["robot_ob"] = [9]
        ob_space["object_ob"] = [6]
        return ob_space

    @property
    def goal_space(self):
        if self._goal_type == "state_obj":
            return [6]  # block pose
        elif self._goal_type == "state_obj_robot":
            return [13]  # block pose, eef pose

    @property
    def reversible_space(self):
        if self._reversible_state_type == "obj_position":
            return [2]  # block pose
        elif self._reversible_state_type == "obj_pose":
            return [6]  # block pose, eef pose

    def get_reverse(self, ob):
        """
        Gets reversible portion of observation
        """
        if self._reversible_state_type == "obj_position":
            # get block qpose
            return ob["object_ob"][:2]
        elif self._reversible_state_type == "obj_pose":
            return ob["object_ob"]

    @property
    def dof(self):
        """
        2 less dimension since gripper is fixed and no connection
        """
        dof = 0  # 'No' Agent
        if self._control_type == "impedance":
            dof = 7  # 7 joints
        elif self._control_type == "ik":
            dof = 2  # move <x,y>
        return dof

    def _load_model(self):
        """
        Loads the Task, which is composed of arena, robot, objects, equality
        """
        # task includes arena, robot, and objects of interest
        from env.models.tasks.push_task import PushTask

        self.mujoco_model = PushTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.mujoco_equality,
            self._rng,
        )

    def get_goal(self, ob):
        """
        Converts an observation object into a goal array
        """
        if self._goal_type == "state_obj":
            # get block qpose
            return ob["object_ob"]
        elif self._goal_type == "state_obj_robot":
            # get block qpose, eef qpose
            rob = ob["robot_ob"]
            # gripper_dis = rob[0]
            eef_pos = rob[1:4]
            # eef_velp = rob[4:7]
            # eef_velr = rob[7:10]
            eef_quat = rob[10:]
            return np.concatenate([ob["object_ob"], eef_pos, eef_quat])

    def _get_next_subtask(self):
        self._subtask_part1 = 0
        self._subtask_part2 = -1


def main():
    from config import create_parser
    import time

    parser = create_parser(env="FurnitureSawyerPushEnv")
    config, unparsed = parser.parse_known_args()

    # generate placing demonstrations
    env = FurnitureSawyerPushEnv(config)
    env.reset()
    while True:
        action = np.zeros(3)
        env.step(action)
        env.render()
        time.sleep(0.2)


def get_demo():
    import pickle

    with open("demos/Sawyer_placeblock_123_0000.pkl", "rb") as f:
        demo = pickle.load(f)
        print(demo["qpos"][-1])


if __name__ == "__main__":
    main()
