import os
import pickle
from typing import Tuple

import numpy as np

from env.furniture_baxter import FurnitureBaxterEnv
from env.models import background_names, furniture_name2id, furniture_xmls
from util.logger import logger


class FurnitureBaxterToyTableAssembleEnv(FurnitureBaxterEnv):
    """
    Baxter environment for following reversed disassembly of toytable.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.furniture_id = furniture_name2id["toy_table"]

        super().__init__(config)
        # default values for rew function
        self._env_config.update(
            {
                "pos_dist": 0.04,
                "rot_dist_up": 0.95,
                "rot_dist_forward": 0.9,
                "project_dist": -1,
            }
        )
        self._gravity_compensation = 1
        # requires multiple connection actions to make connection between two
        # parts.
        self._num_connect_steps = 0
        self._discretize_grip = config.discretize_grip

        # SILO code
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

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        # discretize gripper action
        if self._discretize_grip:
            a = a.copy()
            a[-2] = -1 if a[-2] < 0 else 1
            a[-3] = -1 if a[-3] < 0 else 1

        ob, _, done, _ = super(FurnitureBaxterEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)

        if self._debug:
            for i, body in enumerate(self._object_names):
                pose = self._get_qpos(body)
                logger.debug(f"{body} {pose[:3]} {pose[3:]}")

        info["ac"] = a

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
        # clear previous demos
        if self._record_demo:
            self._demo.reset()

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

        # initialize weld constraints
        eq_obj1id = self.sim.model.eq_obj1id
        eq_obj2id = self.sim.model.eq_obj2id
        p = self._preassembled  # list of weld equality ids to activate
        if len(p) > 0:
            for eq_id in p:
                self.sim.model.eq_active[eq_id] = 1
                object_body_id1 = eq_obj1id[eq_id]
                object_body_id2 = eq_obj2id[eq_id]
                object_name1 = self._object_body_id2name[object_body_id1]
                object_name2 = self._object_body_id2name[object_body_id2]
                self._merge_groups(object_name1, object_name2)
        elif eq_obj1id is not None:
            for i, (id1, id2) in enumerate(zip(eq_obj1id, eq_obj2id)):
                self.sim.model.eq_active[i] = 1 if self._config.assembled else 0

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
        demo = self.load_demo(seed)
        # TODO: figure out best qpos for robot
        # initialize the robot and block to initial demonstraiton state
        self._init_qpos = {
            "qpos": [
                0.74958287,
                -0.1565779,
                -0.01960647,
                0.78434619,
                -0.15412162,
                0.93463559,
                -2.69661249,
                -0.64094791,
                -0.61681124,
                0.20662154,
                1.58147726,
                -0.24183052,
                0.66581204,
                -2.83085012,
            ],
            "4_part4": demo["qpos"][0]["4_part4"],
            "2_part2": demo["qpos"][0]["2_part2"],
            "r_gripper": [-0.01962848, 0.01962187],
        }
        # set toy table pose
        pos_init = []
        quat_init = []
        for body in self._object_names:
            qpos = self._init_qpos[body]
            pos_init.append(qpos[:3])
            quat_init.append(qpos[3:])
        # set baxter pose
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self._init_qpos["qpos"]
        self.sim.data.qpos[self._ref_gripper_right_joint_pos_indexes] = self._init_qpos[
            "r_gripper"
        ]
        self.sim.data.qpos[
            self._ref_gripper_left_joint_pos_indexes
        ] = self.gripper_left.init_qpos

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
            if self._config.assembled:
                self._object_group[i] = 0
            else:
                self._set_qpos(body, pos_init[i], quat_init[i])

        self.sim.forward()

        # store qpos of furniture and robot
        if self._record_demo:
            self._store_qpos()

        if self._agent_type in ["Sawyer", "Panda", "Jaco", "Baxter"]:
            self._initial_right_hand_quat = self._right_hand_quat
            if self._agent_type == "Baxter":
                self._initial_left_hand_quat = self._left_hand_quat

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

        # set two bodies for picking or assemblying
        id1 = self.sim.model.eq_obj1id[0]
        id2 = self.sim.model.eq_obj2id[0]
        self._target_body1 = self.sim.model.body_id2name(id1)
        self._target_body2 = self.sim.model.body_id2name(id2)

    def _place_objects(self):
        """
        Returns fixed initial position and rotations of the toy table.
        The first case has the table top on the left and legs on the right.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        pos_init = [
            [-0.1106984, -0.16384359, 0.05900397],
            [0.09076961, 0.04358102, 0.17718201],
        ]
        quat_init = [
            [-0.00003682, 0.70149268, 0.71195775, -0.03200301],
            [-0.00003672, 0.70149266, 0.71195776, -0.03200302],
        ]

        return pos_init, quat_init

    def _compute_reward(self, action) -> Tuple[float, bool, dict]:
        rew = 0
        done = False
        info = {}
        return rew, done, info

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
            # go back 1 frames so gripper isn't directly over block
            obs = data["obs"][::-1]
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

    def _get_next_subtask(self):
        self._subtask_part1 = 0
        self._subtask_part2 = -1

    @property
    def observation_space(self):
        """
        Returns the observation space.
        """
        ob_space = super().observation_space
        ob_space["object_ob"] = [14]
        return ob_space

    @property
    def goal_space(self):
        if self._goal_type == "state_obj":
            return [14]  # block pose


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureBaxterToyTableAssembleEnv")
    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterToyTableAssembleEnv(config)
    env.run_manual(config)

    # import pickle

    # with open("demos/Baxter_toy_table_0001.pkl", "rb") as f:
    #     demo = pickle.load(f)
    #     qpos = demo["qpos"][::-1]
    #     import ipdb

    #     ipdb.set_trace()

    # env.reset()
    # print(len(demo['actions']))

    # from util.video_recorder import VideoRecorder
    # vr = VideoRecorder()
    # vr.add(env.render('rgb_array')[0])
    # for ac in demo['actions']:
    #     env.step(ac)
    #     vr.add(env.render('rgb_array')[0])
    # vr.save_video('test.mp4')


if __name__ == "__main__":
    main()
