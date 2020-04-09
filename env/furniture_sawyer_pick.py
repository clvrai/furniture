import os
import pickle

import numpy as np

from env.furniture_sawyer import FurnitureSawyerEnv
from env.models import background_names, furniture_name2id, furniture_xmls
from util.logger import logger


class FurnitureSawyerPickEnv(FurnitureSawyerEnv):
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
        # default values for rew function
        self._env_config.update(
            {
                "success_rew": config.success_rew,
                "pick_rew": config.pick_rew,
                "ctrl_penalty": config.ctrl_penalty,
                "hold_duration": config.hold_duration,
                "rand_start_range": config.rand_start_range,
                "rand_block_range": config.rand_block_range,
                "goal_pos_threshold": config.goal_pos_threshold,
                'goal_quat_threshold': config.goal_quat_threshold,
            }
        )
        self._gravity_compensation = 1
        # requires multiple connection actions to make connection between two
        # parts.
        self._num_connect_steps = 0
        self._discretize_grip = config.discretize_grip

        self._goal_type = config.goal_type
        # load demonstrations
        self._data_dir = config.data_dir
        train_dir = os.path.join(self._data_dir, "train")
        test_dir = os.path.join(self._data_dir, "test")

        train_fps = [
            (d.name, d.path)
            for d in os.scandir(train_dir)
            if d.is_file() and d.path.endswith("pkl")
        ]
        test_fps = [
            (d.name, d.path)
            for d in os.scandir(test_dir)
            if d.is_file() and d.path.endswith("pkl")
        ]
        # combine filepath arrays and record start index of test
        self.all_fps = train_fps + test_fps
        self.seed_train = np.arange(0, len(train_fps))
        self.seed_test = np.arange(len(train_fps), len(train_fps) + len(test_fps))

        self._subtask_part1 = 0  # get block ob

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        # discretize gripper action
        if self._discretize_grip:
            a = a.copy()
            a[-2] = -1 if a[-2] < 0 else 1

        ob, _, done, _ = super(FurnitureSawyerEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)

        # for i, body in enumerate(self._object_names):
        #     pose = self._get_qpos(body)
        #     logger.debug(f"{body} {pose[:3]} {pose[3:]}")

        # info["ac"] = a

        return ob, reward, done, info

    def new_game(self, seed=None, is_train=True, record=False):
        if seed is None:
            if is_train:
                seed = self._rng.choice(self.seed_train)
            else:
                seed = self._rng.choice(self.seed_test)
        ob = self.reset(seed, is_train, record)
        return ob, seed

    def reset(
        self,
        seed=None,
        is_train=True,
        record=False,
        furniture_id=None,
        background=None,
    ):
        if seed is None:
            if is_train:
                seed = self._rng.choice(self.seed_train)
            else:
                seed = self._rng.choice(self.seed_test)
        self._reset(seed, furniture_id, background)
        # reset mujoco viewer
        if self._render_mode == "human" and not self._unity:
            self._viewer = self._get_viewer()
        self._after_reset()

        ob = self._get_obs()
        if self._record_demo:
            self._demo.add(ob=ob)

        return ob

    def _reset(self, seed=None, furniture_id=None, background=None):
        """
        Initialize robot at starting point of demonstration.
        Take a random step to increase diversity of starting states.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        self._phase = 1
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
        # load demonstration from filepath, initialize furniture and robot
        name, path = self.all_fps[seed]
        print("loaded", name)
        demo = self.load_demo(seed)
        # initialize the robot and block to initial demonstraiton state
        # TODO: add randomness to this staritng position?
        self._init_qpos = {
            "qpos": demo["qpos"][0]["qpos"],
            "l_gripper": demo["qpos"][0]["l_gripper"],
            "1_block_l": demo["qpos"][0]["1_block_l"],
        }
        pos_init = []
        quat_init = []

        for body in self._object_names:
            qpos = self._init_qpos[body]
            pos_init.append(qpos[:3])
            quat_init.append(qpos[3:])
        if self._agent_type in ["Sawyer", "Panda", "Jaco"]:
            if (
                "l_gripper" in self._init_qpos
                and "r_gripper" not in self._init_qpos
                and "qpos" in self._init_qpos
            ):
                self.sim.data.qpos[self._ref_joint_pos_indexes] = self._init_qpos[
                    "qpos"
                ]
                self.sim.data.qpos[
                    self._ref_gripper_joint_pos_indexes
                ] = self._init_qpos["l_gripper"]

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
        action = np.zeros((8,))
        r = self._env_config["rand_start_range"]
        action[:3] = self._rng.uniform(-r, r, size=3)
        action[6] = -1  # keep gripper open
        self._step_continuous(action)

    def load_demo(self, seed):
        name, path = self.all_fps[seed]
        demo = {}
        with open(path, "rb") as f:
            data = pickle.load(f)
            """
            TIME REVERSAL
            see furniture.py _store_qpos method
            skip to the last frame where block didn't move
            so robot arm is right above block for easy pick
            """
            qpos = data["qpos"][::-1]
            for i in range(len(qpos) - 1):
                q = qpos[i]["1_block_l"]
                q_next = qpos[i + 1]["1_block_l"]
                if np.linalg.norm(q - q_next) > 1e-5:
                    break
            # go back 2 frames so gripper isn't directly over block
            i = max(0, i - 2)
            obs = data["obs"][::-1][i:]
            qpos = qpos[i:]
            # load frames
            demo["qpos"] = qpos
            goal = [self.get_goal(o) for o in obs]
            demo["goal"] = goal
            demo["goal_gt"] = goal[-1]
        return demo

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

    def is_success(self, ob, goal):
        """
        Checks if block pose and robot eef pose are close
        to the goal poses
        Ob is format from get_goal
        Goal is format from demo['goal']
        """
        if isinstance(ob, dict):
            ob = self.get_goal(ob)

        if self._goal_type == "state_obj_robot":
            object_pos = ob[:3]
            goal_object_pos = goal[:3]
            eef_pos = ob[7:10]
            goal_eef_pos = goal[7:10]

            object_success = (
                np.linalg.norm(object_pos - goal_object_pos)
                < self._env_config["goal_pos_threshold"]
            )
            eef_success = (
                np.linalg.norm(eef_pos - goal_eef_pos)
                < self._env_config["goal_pos_threshold"]
            )

            return object_success and eef_success
        elif self._goal_type == "state_obj":
            object_pos, object_quat = ob[:3], ob[3:]
            goal_object_pos, goal_object_quat = goal[:3], goal[3:]
            assert len(object_pos.shape) == 1

            pos_success = (
                np.linalg.norm(object_pos - goal_object_pos)
                < self._env_config["goal_pos_threshold"]
            )

            quat_success = (
                np.linalg.norm(object_quat - goal_object_quat)
                < self._env_config["goal_quat_threshold"]
            )
            return pos_success and quat_success

    def is_possible_goal(self, goal):
        """
        Checks if the goal is a physically
        feasible goal
        """
        return True

    def get_env_success(self, ob, goal):
        return self.is_success(ob, goal)

    def compute_reward(self, achieved_goal, goal, info=None):
        success = self.is_success(achieved_goal, goal).astype(np.float32)
        return success - 1.0

    def _ctrl_reward(self, action):
        if self._config.control_type == "ik":
            a = np.linalg.norm(action[:6])
        elif self._config.control_type == "impedance":
            a = np.linalg.norm(action[:7])

        # grasp_offset, grasp_leg, grip_leg
        ctrl_penalty = -self._env_config["ctrl_penalty"] * a
        if self._phase in [
            "move_leg_up",
            "move_leg",
            "connect",
        ]:  # move slower when moving leg
            ctrl_penalty *= 1

        return ctrl_penalty

    def _compute_reward(self, action):
        rew = 0
        done = self._phase == 4
        self._success = self._phase == 4
        info = {}
        return rew, done, info

    def _get_next_subtask(self):
        self._subtask_part1 = 0
        self._subtask_part2 = -1

    @property
    def observation_space(self):
        """
        Returns the observation space.
        """
        ob_space = super().observation_space
        ob_space["object_ob"] = [7]
        return ob_space

    @property
    def goal_space(self):
        if self._goal_type == "state_obj":
            return [7]  # block pose
        elif self._goal_type == "state_obj_robot":
            return [14]  # block pose, eef pose


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureSawyerPickEnv")
    config, unparsed = parser.parse_known_args()

    # generate placing demonstrations
    env = FurnitureSawyerPickEnv(config)
    env.run_manual(config)


if __name__ == "__main__":
    main()
