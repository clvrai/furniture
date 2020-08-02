from typing import Dict
import os

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.tune.registry import register_env

from config import create_parser
from config.furniture import get_default_config
from env import make_env


class FurnitureGym(gym.Env):
    """
    Gym wrapper class for Furniture assmebly environment.
    """

    def __init__(self, env_config):
        """
        Args:
            env_config: dict containing config values
        """
        config = get_default_config()

        name = "FurnitureSawyerTableLackEnv"
        for key, value in env_config.items():
            setattr(config, key, value)

        # create an environment
        self.env = make_env(name, config)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space["default"]
        self._max_episode_steps = config.max_episode_steps
        self._reward_scale = config.reward_scale

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        ac = {"default": action}
        obs, rew, done, info = self.env.step(ac)
        return obs, rew * self._reward_scale, done, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()


def env_creator(env_config: dict):
    return FurnitureGym(env_config)


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        print("episode {} started".format(episode.episode_id))
        assert len(episode.user_data) == 0

    def on_episode_step(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        info = episode.last_info_for()
        if info:
            for k, v in info.items():
                if k not in episode.user_data:
                    episode.user_data[k] = []
                episode.user_data[k].append(v)

    def on_episode_end(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        for k, v in episode.user_data.items():
            episode.custom_metrics[k] = np.sum(v)
            episode.hist_data[k] = v


parser = create_parser(env="furniture-sawyer-tablelack-v0")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--reward_scale", type=float, default=1)
parser.add_argument("--run_prefix", type=str)
parser.add_argument("--entropy_coeff", type=float, default=1e-4)
parser.add_argument("--clip_param", type=float, default=0.3)


parsed, unparsed = parser.parse_known_args()
env_config = parsed.__dict__
env_config["name"] = "FurnitureSawyerTableLackEnv"

if parsed.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{parsed.gpu}"

register_env("furniture-sawyer-tablelack-v0", env_creator)
ray.init(num_cpus=parsed.num_workers, num_gpus=int(parsed.gpu is not None))


def stopper(trial_id, result):
    success = result["custom_metrics"]["phase_mean"] >= 5
    phase_max = result["custom_metrics"]["phase_max"]
    timesteps = result["timesteps_total"]
    earlystop = phase_max < 1 and timesteps > 5e5
    earlystop |= phase_max < 2 and timesteps > 1e6
    earlystop |= phase_max < 3 and timesteps > 3e6
    earlystop |= phase_max < 4 and timesteps > 5e6

    return success or earlystop


def create_trial_fn(parsed):
    name = parsed.run_prefix

    def trial_str_creator(trial):
        return name

    return trial_str_creator


trial_str_creator = create_trial_fn(parsed)

tune.run(
    "PPO",
    stop=stopper,
    config={
        "env": "furniture-sawyer-tablelack-v0",
        "framework": "torch",
        "callbacks": MyCallbacks,
        "env_config": env_config,
        "observation_filter": "MeanStdFilter",
        "num_workers": max(parsed.num_workers - 1, 0),
        "vf_clip_param": 2000,
        "entropy_coeff": parsed.entropy_coeff
        "clip_param": parsed.clip_param,
    },
    trial_name_creator=tune.function(trial_str_creator),
)
