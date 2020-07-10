import os
from typing import Tuple

import mujoco_py
import numpy as np
from gym.envs.mujoco import mujoco_env

from env.action_spec import ActionSpec
from env.base import EnvMeta


class PusherEnv(mujoco_env.MujocoEnv, metaclass=EnvMeta):
    """PusherEnv.
    Extends leave no trace
    """

    GOAL_ZERO_POS = [0.45, -0.05, -0.3230]  # from xml
    OBJ_ZERO_POS = [0.45, -0.05, -0.275]  # from xml

    def __init__(self, config):
        self._config = config
        self._task = config.task
        self.name = "Push" + self._task.capitalize()
        self._robot_ob = config.robot_ob
        self._record_demo = config.record_demo
        self._reversible_state_type = config.reversible_state_type
        self._goal_pos_threshold = config.goal_pos_threshold
        self._start_pos_threshold = config.start_pos_threshold
        self._sparse_remove_rew = config.use_aot or config.sparse_remove_rew
        self._obj_to_point_coeff = config.obj_to_point_coeff
        self._use_diff_rew = config.use_diff_rew
        self._max_episode_steps = config.max_episode_steps
        self._frame_skip = config.frame_skip
        self._lnt_obj_to_goal_coeff = config.lnt_obj_to_goal_coeff
        self._lnt_obj_to_arm_rew_coeff = config.lnt_obj_to_arm_rew_coeff
        self._lnt_control_rew_coeff = config.lnt_control_rew_coeff

        # Note: self._goal is the same for the forward and reset tasks. Only
        # the reward function changes.
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder, "models/assets/pusher.xml")
        self._initialize_mujoco(xml_filename, self._frame_skip)
        (self._goal, self._start) = self._get_goal_start()

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

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def _get_goal_start(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        qpos[:] = 0
        qvel[:] = 0
        self.set_state(qpos, qvel)
        goal = self.get_body_com("goal").copy()[:2]
        start = self.get_body_com("object").copy()[:2]
        return (goal, start)

    def _preprocess_action(self, a):
        """Make into array and scale to -2 and 2"""
        if isinstance(a, dict):
            a = np.concatenate([a[key] for key in self.action_space.shape.keys()])
        a = a * 2
        return a

    def step(self, a) -> Tuple[dict, float, bool, dict]:
        a = self._preprocess_action(a)
        self.do_simulation(a, self.frame_skip)
        done = False
        obs = self._get_obs()
        self._episode_length += 1
        if self._task == "forward":
            if self._use_diff_rew:
                r, info = self._forward_diff_reward(obs, a)
            else:
                r, info = self._forward_reward(obs, a)
        elif self._task == "reset":
            if self._use_diff_rew:
                r, info = self._reset_diff_reward(obs, a)
            else:
                r, info = self._reset_reward(obs, a)
        if self._record_demo:
            self._demo.add(ob=obs, action=a, reward=r)
        info["reward"] = r
        if self._task == "forward":
            dist = np.linalg.norm(obs["object_ob"] - self._goal)
            success = dist < self._goal_pos_threshold
        elif self._task == "reset":
            dist = np.linalg.norm(obs["object_ob"] - self._start)
            success = dist < self._start_pos_threshold
        info["episode_success"] = int(success)
        self._episode_reward += r
        if self._config.algo in ["ppo", "ddpg"]:
            done = self._episode_length == self._max_episode_steps
        if success:
            done = True
        return obs, r, done, info

    def _get_obs(self):
        obs = {
            "robot_ob": np.concatenate(
                [
                    self.data.qpos.flat[:3],
                    self.data.qvel.flat[:3],
                    self.get_body_com("tips_arm")[:2],
                ]
            ),
            "object_ob": self.get_body_com("object").copy()[:2],
        }
        return obs

    def render(self, mode="human"):
        img = super().render(mode, camera_id=0)
        if mode != "rgb_array":
            return img
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img

    def reset_success(self):
        ob = self._get_obs()
        dist = np.linalg.norm(ob["object_ob"] - self._start)
        success = dist < self._start_pos_threshold
        return success

    def begin_reset(self):
        """
        Switch to reset mode. Init reset reward
        """
        self._task = "reset"
        self._reset_episodic_vars()

    def begin_forward(self):
        """
        Switch to forward mode. Init forward reward
        """
        self._task = "forward"
        self._reset_episodic_vars()

    def _reset_episodic_vars(self):
        """
        Resets episodic variables
        """
        self._prev_push_dist = np.linalg.norm(
            self.get_body_com("object")[:2] - self._goal
        )
        self._prev_pull_dist = np.linalg.norm(
            self.get_body_com("object")[:2] - self._start
        )
        self._episode_length = 0
        self._episode_reward = 0
        if self._record_demo:
            self._demo.reset()

    def reset(self, is_train=True, record=False, **kwargs):
        """
        Resets the environment. If lfd, then utilize the seed parameter.
        If seed is none, then we choose a random seed else use the given seed.
        Used by run_episode in evaluation code for BC in rl/rollouts.py
        """
        self.sim.reset()
        ob = self.reset_model()

        self._reset_episodic_vars()
        if self._record_demo:
            self._demo.add(ob=ob)
        return ob

    def reset_model(self):
        # qpos is 3 robot joints, 2 object joints
        qpos = self.init_qpos
        qpos[:] = 0
        qpos[:3] = [0.3, 1.2, 0]  # shoulder, elbox, wrist
        qpos[:3] += np.random.uniform(-0.05, 0.05)

        # object noise
        # qpos[-2:] += [0, -0.4]
        r = 0.021
        qpos[-2:] += self.np_random.uniform(-r, r, 2)
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0

        # For the reset task, flip the initial positions of the goal and puck
        # if self._task == "reset":
        #     qpos[-3] -= 0.7
        #     qpos[-1] += 0.7

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _huber(self, x, bound, delta=0.2):
        assert delta < bound
        if x < delta:
            loss = 0.5 * x * x
        else:
            loss = delta * (x - 0.5 * delta)
        return loss

    def _reward_fn(self, x, bound=0.42):
        # Using bound = 0.4 because that's the initial puck-goal distance.
        x = np.clip(x, 0, bound)
        loss = self._huber(x, bound)
        loss /= self._huber(bound, bound)
        reward = 1 - loss
        assert 0 <= loss <= 1
        return reward

    def _forward_diff_reward(self, s, a):
        """
        Computes the difference in position as reward
        """
        rew = 0
        info = {}
        obj_to_arm = self.get_body_com("object")[:2] - self.get_body_com("tips_arm")[:2]
        obj_to_arm_dist = np.linalg.norm(obj_to_arm)
        obj_to_goal = self.get_body_com("object")[:2] - self._goal
        obj_to_goal_dist = np.linalg.norm(obj_to_goal)
        dist_diff = self._prev_push_dist - obj_to_goal_dist
        obj_to_goal_rew = dist_diff * self._obj_to_point_coeff
        success = obj_to_goal_dist < self._goal_pos_threshold

        control_dist = np.linalg.norm(a)
        control_reward = self._reward_fn(control_dist, bound=3.464) * 0.01
        rew = obj_to_goal_rew + control_reward

        info["forward_reward"] = rew
        info["control_penalty"] = control_reward
        info["obj_to_arm_dist"] = obj_to_arm_dist
        info["obj_to_goal_dist"] = obj_to_goal_dist
        return rew, info

    def _reset_diff_reward(self, s, a):
        """
        Computes the difference in position as reward
        """
        rew = 0
        info = {}
        obj_to_arm = self.get_body_com("object")[:2] - self.get_body_com("tips_arm")[:2]
        obj_to_arm_dist = np.linalg.norm(obj_to_arm)
        obj_to_start = self.get_body_com("object")[:2] - self._start
        obj_to_start_dist = np.linalg.norm(obj_to_start)
        dist_diff = self._prev_pull_dist - obj_to_start_dist
        obj_to_start_rew = dist_diff * self._obj_to_point_coeff
        success = obj_to_start_dist < self._start_pos_threshold

        control_dist = np.linalg.norm(a)
        control_reward = self._reward_fn(control_dist, bound=3.464) * 0.01
        rew = obj_to_start_rew + control_reward

        info["forward_reward"] = rew
        info["control_penalty"] = control_reward
        info["obj_to_arm_dist"] = obj_to_arm_dist
        info["obj_to_start_dist"] = obj_to_start_dist
        return rew, info

    def _forward_reward(self, s, a):
        info = {}
        obj_to_arm = s["object_ob"] - s["robot_ob"][-2:]
        obj_to_goal = s["object_ob"] - self._goal
        obj_to_arm_dist = np.linalg.norm(obj_to_arm)
        obj_to_goal_dist = np.linalg.norm(obj_to_goal)
        control_dist = np.linalg.norm(a)
        success = obj_to_goal_dist < self._goal_pos_threshold

        forward_reward = self._reward_fn(obj_to_goal_dist)
        obj_to_arm_reward = self._reward_fn(obj_to_arm_dist)
        # The control_dist is between 0 and sqrt(2^2 + 2^2 + 2^2) = 3.464
        control_reward = self._reward_fn(control_dist, bound=3.464)

        forward_reward *= self._lnt_obj_to_goal_coeff
        obj_to_arm_reward *= self._lnt_obj_to_arm_rew_coeff
        control_reward *= self._lnt_control_rew_coeff
        forward_shaped_reward = forward_reward + obj_to_arm_reward + control_reward
        assert 0 <= forward_shaped_reward <= 1
        if success:
            forward_shaped_reward += self._config.success_rew
        info["forward_reward"] = forward_reward
        info["obj_to_arm_reward"] = obj_to_arm_reward
        info["control_penalty"] = control_reward
        info["obj_to_arm_dist"] = obj_to_arm_dist
        info["obj_to_goal_dist"] = obj_to_goal_dist
        return forward_shaped_reward, info

    def _reset_reward(self, s, a):
        info = {}
        obj_to_arm = s["object_ob"] - s["robot_ob"][-2:]
        obj_to_start = s["object_ob"] - self._start
        obj_to_arm_dist = np.linalg.norm(obj_to_arm)
        obj_to_start_dist = np.linalg.norm(obj_to_start)
        control_dist = np.linalg.norm(a)
        success = obj_to_start_dist < self._start_pos_threshold

        reset_reward = self._reward_fn(obj_to_start_dist)
        obj_to_arm_reward = self._reward_fn(obj_to_arm_dist)
        # The control_dist is between 0 and sqrt(2^2 + 2^2 + 2^2) = 3.464
        control_reward = self._reward_fn(control_dist, bound=3.464)

        if self._sparse_remove_rew:
            success_reward = 0
            if success:
                success_reward = self._config.success_rew
                if self._config.use_aot:
                    success_reward = self._config.aot_succ_rew

            reset_shaped_reward = success_reward + 0.125 * control_reward
            info["success_reward"] = success_reward
        else:
            reset_reward *= self._lnt_obj_to_goal_coeff
            obj_to_arm_reward *= self._lnt_obj_to_arm_rew_coeff
            control_reward *= self._lnt_control_rew_coeff
            reset_shaped_reward = reset_reward + obj_to_arm_reward + control_reward
            assert 0 <= reset_shaped_reward <= 1
            if success:
                reset_shaped_reward += self._config.reset_success_rew

        info["reset_reward"] = reset_reward
        info["obj_to_arm_reward"] = obj_to_arm_reward
        info["control_penalty"] = control_reward
        info["obj_to_arm_dist"] = obj_to_arm_dist
        info["obj_to_start_dist"] = obj_to_start_dist
        return reset_shaped_reward, info

    def reset_reward(self, s, a, preprocess_action=True):
        if preprocess_action:
            a = self._preprocess_action(a)
        return self._reset_reward(s, a)[0]

    def forward_reward(self, s, a, preprocess_action=True):
        if preprocess_action:
            a = self._preprocess_action(a)
        return self._forward_reward(s, a)[0]

    @property
    def dof(self) -> int:
        """
        Returns the DoF of the robot.
        """
        return 3

    @property
    def observation_space(self) -> dict:
        """
        Object ob: top and bottom pos of peg
        Robot ob: 10d of qpos, qvel, and robot
        """
        ob_space = {"robot_ob": [8], "object_ob": [2]}

        return ob_space

    @property
    def action_space(self):
        """
        Returns ActionSpec of action space, see
        action_spec.py for more documentation.
        """
        return ActionSpec(self.dof)

    @property
    def reversible_space(self):
        if self._reversible_state_type == "obj_position":
            return [2]  # peg positions

    def get_reverse(self, ob):
        """
        Gets reversible portion of observation
        """
        if self._reversible_state_type == "obj_position":
            # get block qpose
            return ob["object_ob"]


if __name__ == "__main__":
    import time
    from config import create_parser

    parser = create_parser("PusherEnv")
    parser.set_defaults(env="PusherEnv")
    config, unparsed = parser.parse_known_args()
    env = PusherEnv(config)
    ob = env.reset()
    # import ipdb; ipdb.set_trace()
    for _ in range(10000):
        action = np.zeros(env.dof)
        # action = env.action_space.sample()
        ob, rew, done, info = env.step(action)
        # print(env.get_body_com("object"))
        env.render()
        env.reset()
        # time.sleep(1)
