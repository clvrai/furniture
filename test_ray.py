import argparse
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
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
        self.action_space = self.env.action_space
        self._max_episode_steps = config.max_episode_steps

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()


def env_creator(env_config: dict):
    return FurnitureGym(env_config)


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        print("episode {} started".format(episode.episode_id))
        assert len(episode.user_data) == 0

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        info = episode.last_info_for()
        if info:
            for k, v in info.items():
                if k not in episode.user_data:
                    episode.user_data[k] = []
                episode.user_data[k].append(v)

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        for k, v in episode.user_data.items():
            episode.custom_metrics[k] = np.sum(v)
            episode.hist_data[k] = v


parser = create_parser(env="furniture-sawyer-tablelack-v0")
parser.add_argument("--num_workers", type=int, default=0)
parsed, unparsed = parser.parse_known_args()
env_config = parsed.__dict__
env_config["name"] = "FurnitureSawyerTableLackEnv"

register_env("furniture-sawyer-tablelack-v0", env_creator)
ray.init(num_cpus=1)
trainer = ppo.PPOTrainer(
    env="furniture-sawyer-tablelack-v0",
    config={
        "framework": "torch",
        "num_workers": parsed.num_workers,
        "callbacks": MyCallbacks,
        "env_config": env_config,
        # "train_batch_size": 200,
        # "sgd_minibatch_size": 32,
        # "num_sgd_iter": 1,
    },
)

while True:
    info = trainer.train()
