import os
import pickle
from typing import Tuple

import numpy as np

from env.furniture_baxter import FurnitureBaxterEnv
from env.models import background_names, furniture_name2id, furniture_xmls
from env.transform_utils import forward_vector_cos_dist
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
                "goal_pos_threshold": config.goal_pos_threshold,
                "goal_quat_threshold": config.goal_quat_threshold,
                "goal_cos_threshold": config.goal_cos_threshold,
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
        self.sim.model.opt.gravity[-1] = -1
        # discretize gripper action
        if self._discretize_grip:
            a = a.copy()
            a[-2] = -1 if a[-2] < 0 else 1
            a[-3] = -1 if a[-3] < 0 else 1

        ob, _, done, _ = super(FurnitureBaxterEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)
        # check if table moved too much
        # table_pos = ob["object_ob"][:7]
        # table_drift = np.abs(np.linalg.norm(table_pos - self._init_qpos["4_part4"]))
        # if table_drift > 0.03:
        #     done = True
        #     info["table_oob"] = True
        #     logger.warning(f"Table moved too much: {table_drift}")

        if self._debug:
            part2_ob_pose = ob["object_ob"][7:]
            cos_sim = forward_vector_cos_dist(
                part2_ob_pose[3:], self._init_qpos["2_part2"][3:]
            )
            logger.debug(f"cos sim: {cos_sim}")
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
        for _ in range(1):
            for obj_name in self._object_names:
                self._stop_object(obj_name, gravity=0)
            self.sim.forward()
            self.sim.step()

        logger.debug("*** furniture initialization ***")
        # load demonstration from filepath, initialize furniture and robot
        name, path = self.all_fps[seed]
        demo = self.load_demo(seed)
        # initialize the robot and block to initial demonstraiton state
        qpos_idx = 0
        self._init_qpos = {
            "qpos": demo["qpos"][qpos_idx]["qpos"],
            "4_part4": demo["qpos"][qpos_idx]["4_part4"],
            "2_part2": demo["qpos"][qpos_idx]["2_part2"],
            "r_gripper": demo["qpos"][qpos_idx]["r_gripper"],
            "l_gripper": demo["qpos"][qpos_idx]["l_gripper"],
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
        self._success = self._is_aligned("leg-top,,conn_site4", "top-leg,,conn_site4")
        done = self._success
        info = {}
        return rew, done, info

    def load_demo(self, seed):
        path = self.all_fps[seed][1]
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

        if self._goal_type == "state_obj":
            part4_ob_pose, part2_ob_pose = ob[0:7], ob[7:]
            object_pos = np.concatenate([ob[0:3], ob[7:10]])
            goal_object_pos = np.concatenate([goal[0:3], goal[7:10]])
            part4_goal_pose, part2_goal_pose = goal[0:7], goal[7:]
            assert len(object_pos.shape) == 1

            pos_success = (
                np.linalg.norm(object_pos - goal_object_pos)
                < self._env_config["goal_pos_threshold"]
            )
            # Use cosine similarity between up vectors for table leg (part2)
            cos_sim = forward_vector_cos_dist(part2_ob_pose[3:], part2_goal_pose[3:])
            cos_success = cos_sim > self._env_config["goal_cos_threshold"]
            # use eucl distance between quaternions for table top (part4)
            quat_success = (
                np.linalg.norm(part4_ob_pose[3:] - part4_goal_pose[3:])
                < self._env_config["goal_quat_threshold"]
            )
            return pos_success and quat_success and cos_success

    def is_possible_goal(self, goal):
        """
        Checks if the goal is a physically
        feasible goal
        """
        return True

    def get_env_success(self, ob, goal):
        return self._success

    def compute_reward(self, achieved_goal, goal, info=None):
        success = self.is_success(achieved_goal, goal).astype(np.float32)
        return success - 1.0

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
            return [14]  # 2 block poses


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureBaxterToyTableAssembleEnv")
    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterToyTableAssembleEnv(config)
    env.run_manual(config)

    # import pickle

    # with open("demos/Baxter_toy_table_0002.pkl", "rb") as f:
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
