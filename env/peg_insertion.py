import os
import pickle
from typing import Tuple

import mujoco_py
import numpy as np
from gym.envs.mujoco import mujoco_env

from env.action_spec import ActionSpec
from env.base import EnvMeta
from util.demo_recorder import DemoRecorder


class PegInsertionEnv(mujoco_env.MujocoEnv, metaclass=EnvMeta):
    """PegInsertionEnv
    Extends https://github.com/brain-research/LeaveNoTrace
    We define the forward task to be pulling the peg out of the hole, and the
    reset task to be putting the peg into the hole.
    """

    def __init__(self, config):
        self._config = config
        self._algo = config.algo
        self._seed = config.seed
        self._sparse = config.sparse_rew
        self._task = config.task
        self._lfd = config.lfd
        self.name = "Peg" + self._task.capitalize()
        self._max_episode_steps = float("inf")
        self._robot_ob = config.robot_ob
        self._goal_pos_threshold = config.goal_pos_threshold
        self._start_pos_threshold = config.start_pos_threshold
        self._goal_quat_threshold = config.goal_quat_threshold
        self._record_demo = config.record_demo
        self._goal_type = config.goal_type
        self._action_noise = config.action_noise
        self._wrist_noise = config.wrist_noise
        self._body_noise = config.body_noise
        self._sparse_remove_rew = config.use_aot or config.sparse_remove_rew
        # self._dist_count = self._dist_sum = 0

        # load demonstrations if learning from demonstrations
        if self._lfd:
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
            self.all_fps = train_fps + test_fps
            self.seed_train = np.arange(0, len(train_fps))
            self.seed_test = np.arange(len(train_fps), len(train_fps) + len(test_fps))

        # reward config
        self._peg_to_point_rew_coeff = config.peg_to_point_rew_coeff
        self._success_rew = config.success_rew
        self._control_penalty_coeff = config.control_penalty_coeff

        # demo loader
        if self._record_demo:
            self._demo = DemoRecorder(config.demo_dir)

        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder, "models/assets/peg_insertion.xml")
        self._initialize_mujoco(xml_filename, 5)
        self._reset_episodic_vars()

    def _initialize_mujoco(self, model_path, frame_skip):
        """Taken from mujoco_env.py __init__ from mujoco_py package"""
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.seed()

    def reset_reward(self, ob, a, next_ob):
        if isinstance(a, dict):
            a = np.concatenate([a[key] for key in self.action_space.shape.keys()])
        return self._remove_reward(next_ob, a)

    def reset_done(self):
        peg_pos = np.hstack(
            [self.get_body_com("leg_bottom"), self.get_body_com("leg_top")]
        )
        dist_to_start = np.linalg.norm(self._start_pos - peg_pos)
        peg_at_start = dist_to_start < self._start_pos_threshold
        return peg_at_start

    def step(self, a) -> Tuple[dict, float, bool, dict]:
        if isinstance(a, dict):
            a = np.concatenate([a[key] for key in self.action_space.shape.keys()])
        # no action noise during bc evaluation!
        # if self._algo != "bc" and self._action_noise is not None:
        #     r = self._action_noise
        #     a = a + self.np_random.uniform(-r, r, size=len(a))
        self.do_simulation(a, self.frame_skip)

        done = False
        obs = self._get_obs()
        self._episode_length += 1

        # ignore reward if lfd
        if self._lfd:
            reward, info = 0, {}
        elif self._task == "insert":
            reward, info = self._insert_reward(obs, a)
        elif self._task == "remove":
            reward, info = self._remove_reward(obs, a)
        self._episode_reward += reward

        info["episode_success"] = int(self._success)
        if self._success or self._episode_length == self._max_episode_steps:
            done = True
            info["episode_reward"] = self._episode_reward

        info["reward"] = reward
        if self._record_demo:
            self._demo.add(ob=obs, action=a, reward=reward)
        return obs, reward, done, info

    def new_game(self, seed=None, is_train=True, record=False) -> Tuple[dict, int]:
        """
        Wrapper for SILO to reset environment
        Returns initial ob and seed used for reset
        """
        if seed is None:
            if is_train:
                seed = self.np_random.choice(self.seed_train)
            else:
                seed = self.np_random.choice(self.seed_test)
        ob = self.reset(seed, is_train, record)
        return ob, seed

    def reset(self, seed=None, is_train=True, record=False):
        """
        Resets the environment. If lfd, then utilize the seed parameter.
        If seed is none, then we choose a random seed else use the given seed.
        Used by run_episode in evaluation code for BC in rl/rollouts.py
        """
        # if seed is not None:
        #    assert self._lfd
        # determine seed if lfd and seed not given
        if self._lfd and seed is None:
            if is_train:
                seed = self.np_random.choice(self.seed_train)
            else:
                seed = self.np_random.choice(self.seed_test)
        self.sim.reset()
        ob = self.reset_model(seed)
        # add initialization noise for evaluation of bc
        # if self._action_noise is not None:
        #     r = self._action_noise
        #     a = np.zeros(7) + self.np_random.uniform(-r, r, size=7)
        #     self.do_simulation(a, self.frame_skip)
        #     ob = self._get_obs()

        self._reset_episodic_vars()
        if self._record_demo:
            self._demo.add(ob=ob)
        return ob

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def begin_reset(self):
        """
        Switch to reset mode. Init reset reward
        """
        self._success = False
        peg_pos = np.hstack(
            [self.get_body_com("leg_bottom"), self.get_body_com("leg_top")]
        )
        dist_to_start = np.linalg.norm(self._start_pos - peg_pos)
        self._prev_remove_dist = dist_to_start
        dist_to_goal = np.linalg.norm(self._goal_pos - peg_pos)
        self._prev_insert_dist = dist_to_goal

    def begin_forward(self):
        """
        Switch to forward mode. Init forward reward
        """
        self._success = False
        peg_pos = np.hstack(
            [self.get_body_com("leg_bottom"), self.get_body_com("leg_top")]
        )
        dist_to_start = np.linalg.norm(self._start_pos - peg_pos)
        self._prev_remove_dist = dist_to_start
        dist_to_goal = np.linalg.norm(self._goal_pos - peg_pos)
        self._prev_insert_dist = dist_to_goal

    def _reset_episodic_vars(self):
        """
        Resets episodic variables
        """
        self._episode_length = 0
        self._episode_reward = 0
        self._success = False
        if self._record_demo:
            self._demo.reset()

        peg_pos = np.hstack(
            [self.get_body_com("leg_bottom"), self.get_body_com("leg_top")]
        )
        self._start_pos = np.array(
            [0.10600084, 0.15715909, 0.1496843, 0.24442536, -0.09417238, 0.23726938]
        )
        self._goal_pos = np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2])
        dist_to_start = np.linalg.norm(self._start_pos - peg_pos)
        self._prev_remove_dist = dist_to_start
        dist_to_goal = np.linalg.norm(self._goal_pos - peg_pos)
        self._prev_insert_dist = dist_to_goal

    def reset_model(self, seed=None) -> dict:
        """
        Resets the mujoco model. If lfd, use seed parameter to
        load the demonstration and reset mujoco model to demo.
        Returns an observation
        """
        if self._lfd:
            if self._task == "insert":
                demo = self.load_demo(seed)
                rob = demo["obs"][0]["robot_ob"]
                qpos = rob[:7]
                qvel = np.zeros(7)

            else:
                raise NotImplementedError
        else:
            if self._task == "insert":
                # Reset peg above hole:
                wn = self._wrist_noise
                bn = self._body_noise
                wristnoise = self.np_random.uniform(-wn, wn, size=3)
                bodynoise = self.np_random.uniform(-bn, bn, size=4)
                qpos = np.array(
                    [
                        0.44542705,
                        0.64189252,
                        -0.39544481,
                        -2.32144865,
                        -0.17935136,
                        -0.60320289,
                        1.57110214,
                    ]
                )
                qpos[:4] = qpos[:4] + bodynoise
                qpos[4:] = qpos[4:] + wristnoise

            else:
                # Reset peg in hole
                qpos = np.array(
                    [
                        0.52601062,
                        0.57254126,
                        -2.0747581,
                        -1.55342248,
                        0.15375072,
                        -0.5747922,
                        0.70163815,
                    ]
                )
            qvel = np.zeros(7)
        self.set_state(qpos, qvel)
        # peg_pos = np.hstack(
        #     [self.get_body_com("leg_bottom"), self.get_body_com("leg_top")]
        # )
        # dist_to_start = np.linalg.norm(self._start_pos - peg_pos)
        # self._dist_sum += dist_to_start
        # self._dist_count += 1
        # print(
        #     f"avg dist: {self._dist_sum/self._dist_count}, dist to start: {dist_to_start}"
        # )
        return self._get_obs()

    def _remove_reward(self, s, a) -> Tuple[float, dict]:
        """Compute the peg removal reward.
        Note: We assume that the reward is computed on-policy, so the given
        state is equal to the current observation.
        Returns reward and info dict
        """
        info = {}
        peg_pos = s["object_ob"]
        # peg_pos = np.hstack(
        #     [self.get_body_com("leg_bottom"), self.get_body_com("leg_top")]
        # )
        dist_to_start = np.linalg.norm(self._start_pos - peg_pos)
        # we want the current distance to be smaller than the previous step's distnace
        dist_diff = self._prev_remove_dist - dist_to_start
        self._prev_remove_dist = dist_to_start
        peg_to_start_reward = dist_diff * self._peg_to_point_rew_coeff

        control_reward = np.dot(a, a) * self._control_penalty_coeff * -1
        peg_at_start = dist_to_start < self._start_pos_threshold

        self._success = peg_at_start
        success_reward = 0
        if self._success:
            success_reward = self._config.success_rew
            if self._config.use_aot:
                success_reward = self._config.aot_succ_rew

        if self._sparse_remove_rew:
            remove_reward = control_reward + success_reward
        else:
            remove_reward = peg_to_start_reward + control_reward + success_reward

        info["dist_to_start"] = dist_to_start
        info["control_rew"] = control_reward
        info["peg_to_start_rew"] = peg_to_start_reward
        info["success_rew"] = success_reward
        return remove_reward, info

    def _insert_reward(self, s, a) -> Tuple[float, dict]:
        """Compute the insertion reward.
        Note: We assume that the reward is computed on-policy, so the given
        state is equal to the current observation.
        """
        info = {}
        peg_pos = s["object_ob"]
        # peg_pos = np.hstack(
        #     [self.get_body_com("leg_bottom"), self.get_body_com("leg_top")]
        # )
        dist_to_goal = np.linalg.norm(self._goal_pos - peg_pos)
        dist_diff = self._prev_insert_dist - dist_to_goal
        self._prev_insert_dist = dist_to_goal
        peg_to_goal_reward = dist_diff * self._peg_to_point_rew_coeff

        control_reward = np.dot(a, a) * self._control_penalty_coeff * -1
        peg_at_goal = (
            dist_to_goal < self._goal_pos_threshold
            and self.get_body_com("leg_bottom")[2] < -0.4
        )

        self._success = peg_at_goal
        success_reward = 0
        if self._success:
            success_reward = self._success_rew

        if self._sparse:
            insert_reward = control_reward + success_reward
        else:
            insert_reward = peg_to_goal_reward + control_reward + success_reward

        info["dist_to_goal"] = dist_to_goal
        info["control_rew"] = control_reward
        info["peg_to_goal_rew"] = peg_to_goal_reward
        info["success_rew"] = success_reward
        # info["leg_bottom_z"] = self.get_body_com("leg_bottom")[2]

        return insert_reward, info

    def _get_obs(self) -> dict:
        """
        Returns the robot actuator states, and the object pose.
        By default, returns the object pose.
        """
        # obs = {
        #     "object_ob": np.concatenate(
        #         [self.data.get_body_xpos("ball"), self.data.get_body_xquat("ball")]
        #     )
        # }
        obs = {
            "object_ob": np.hstack(
                [self.get_body_com("leg_bottom"), self.get_body_com("leg_top")]
            )
        }
        if self._robot_ob:
            obs["robot_ob"] = np.concatenate(
                [self.sim.data.qpos.flat, self.sim.data.qvel.flat]
            )
        return obs

    def render(self, mode="human"):
        img = super().render(mode, camera_id=0)
        if mode != "rgb_array":
            return img
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img

    def save_demo(self):
        self._demo.save(self.name)

    def load_demo(self, seed) -> dict:
        name, path = self.all_fps[seed]
        demo = {}
        with open(path, "rb") as f:
            data = pickle.load(f)
            """
            TIME REVERSAL
            """
            obs = data["obs"][::-1]
            # load frames
            demo["obs"] = obs
            goal = [self.get_goal(o) for o in obs]
            demo["goal"] = goal
            demo["goal_gt"] = goal[-1]
        return demo

    def get_goal(self, ob) -> list:
        """
        Converts an observation object into a goal array
        """
        if self._goal_type == "state_obj":
            # get peg qpose
            return ob["object_ob"]
        elif self._goal_type == "state_obj_robot":
            # get peg qpose, robot qpos, robot qvel
            return np.concatenate([ob["object_ob"], ob["robot_ob"]])

    def is_success(self, ob, goal) -> bool:
        """
        Checks if block pose and robot eef pose are close
        to the goal poses
        Ob is format from get_goal
        Goal is format from demo['goal']
        """
        if isinstance(ob, dict):
            ob = self.get_goal(ob)
        if self._goal_type == "state_obj":
            pos_success = np.linalg.norm(ob - goal) < self._goal_pos_threshold
            return pos_success
        else:
            raise NotImplementedError()

    def is_possible_goal(self, goal):
        return True

    def get_env_success(self, ob, goal):
        return self.is_success(ob, goal)

    def compute_reward(self, achieved_goal, goal, info=None):
        success = self.is_success(achieved_goal, goal).astype(np.float32)
        return success - 1.0

    @property
    def dof(self) -> int:
        """
        Returns the DoF of the robot.
        """
        return 7

    @property
    def observation_space(self) -> dict:
        """
        Object ob: top and bottom pos of peg
        Robot ob: 14D qpos and qvel of robot
        """
        ob_space = {"robot_ob": [14], "object_ob": [6]}

        return ob_space

    @property
    def action_space(self):
        """
        Returns ActionSpec of action space, see
        action_spec.py for more documentation.
        """
        return ActionSpec(self.dof)

    @property
    def goal_space(self):
        if self._goal_type == "state_obj":
            return [6]  # peg pos
        elif self._goal_type == "state_obj_robot":
            return [20]  # peg pose and robot qpos and robot qvel


if __name__ == "__main__":
    import time
    from config import create_parser

    parser = create_parser("PegInsertionEnv")
    parser.set_defaults(env="PegInsertionEnv")
    config, unparsed = parser.parse_known_args()
    env = PegInsertionEnv(config)

    for _ in range(10000):
        env.reset(seed=None)
        env.render()
        time.sleep(0.01)
