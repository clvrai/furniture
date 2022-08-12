""" Define all environments and provide helper functions to load environments. """

import gym
import hydra
from omegaconf import OmegaConf

from ..util.subproc_vec_env import SubprocVecEnv


REGISTERED_ENVS = {}


def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class


def get_env(name):
    """
    Gets the environment class given @name.
    """
    if name not in REGISTERED_ENVS:
        raise Exception(
            f"Unknown environment name: {name}\nAvailable environments: {REGISTERED_ENVS}"
        )
    return REGISTERED_ENVS[name]


def get_cfg(**kwarg):
    if "ikea_cfg" in kwarg:
        ikea_cfg = kwarg["ikea_cfg"]
    else:
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            raise Exception(
                """Include config for IKEA environment in your hydra config, e.g., - env: ikea_dense
                Then, pass env when creating the environment, e.g., gym.make("IKEASawyerDense-v0", ikea_cfg=ikea_cfg)
                """
            )
        if "config_name" not in kwarg:
            raise Exception(
                "config_name should be specified to load environment config (e.g. ikea_dense)"
            )
        if "class_name" not in kwarg:
            raise Exception(
                "class_name should be specified to instantiate environment (e.g. FurnitureSawyerEnv)"
            )

        with hydra.initialize(config_path="../config/env"):
            ikea_cfg = hydra.compose(config_name=kwarg["config_name"])["ikea_cfg"]
            OmegaConf.set_struct(ikea_cfg, False)

    for key, value in kwarg.items():
        if key != "ikea_cfg":
            OmegaConf.update(ikea_cfg, key, value)

    return ikea_cfg


def make_env(**kwarg):
    """
    Creates a new environment instance with @kwarg.cfg or @kwarg.class_name and @kwarg.config_name.
    """
    cfg = get_cfg(**kwarg)
    env = get_env(cfg.class_name)
    return env(cfg)


def get_gym_env(env_id, cfg):
    env = gym.make(env_id, ikea_cfg=cfg)
    return env


def make_vec_env(env_id, num_env, **kwarg):
    """
    Creates a wrapped SubprocVecEnv using OpenAI gym interface.
    Unity app will use the port number from @cfg.unity.port to (@cfg.unity.port + @num_env - 1).

    Code modified based on
    https://github.com/openai/baselines/blob/master/baselines/common/cmd_util.py

    Args:
        env_id: environment id registered in in `env/__init__.py`.
        num_env: number of environments to launch.
        cfg: general configuration for the environment.
    """
    cfg = get_cfg(**kwarg)
    def make_thunk(rank):
        new_cfg = cfg.copy()
        new_cfg.unity.port = cfg.unity.port + rank
        new_cfg.seed = cfg.seed + rank
        return lambda: get_gym_env(env_id, new_cfg)

    return SubprocVecEnv([make_thunk(i) for i in range(num_env)])


class EnvMeta(type):
    """ Meta class for registering environments. """

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["FurnitureEnv"]

        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls
