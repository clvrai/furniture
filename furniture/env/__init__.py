""" Define all environments and provide helper functions to load environments. """

# OpenAI gym interface
from gym.envs.registration import register

from .base import make_env, make_vec_env

# register all environment to use
from .furniture_baxter import FurnitureBaxterEnv
from .furniture_baxter_block import FurnitureBaxterBlockEnv
from .furniture_baxter_toytable import FurnitureBaxterToyTableEnv
from .furniture_cursor import FurnitureCursorEnv
from .furniture_jaco import FurnitureJacoEnv
from .furniture_panda import FurniturePandaEnv
from .furniture_sawyer import FurnitureSawyerEnv
from .furniture_sawyer_dense import FurnitureSawyerDenseRewardEnv


# add cursor environment to Gym
register(
    id="furniture-cursor-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "name": "FurnitureCursorEnv",
        "furniture_id": 0,
        "background": "Lab",
        "port": 1050,
    },
)


# add sawyer environment to Gym
register(
    id="furniture-sawyer-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "name": "FurnitureSawyerEnv",
        "furniture_name": "swivel_chair_0700",
        "background": "Industrial",
        "port": 1050,
    },
)


# add baxter environment to Gym
register(
    id="furniture-baxter-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "name": "FurnitureBaxterEnv",
        "furniture_id": 1,
        "background": "Interior",
        "port": 1050,
    },
)


# add jaco environment to Gym
register(
    id="furniture-jaco-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "name": "FurnitureJacoEnv",
        "furniture_id": 1,
        "background": "Interior",
        "port": 1050,
    },
)


# add panda environment to Gym
register(
    id="furniture-panda-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "name": "FurniturePandaEnv",
        "furniture_id": 1,
        "background": "Interior",
        "port": 1050,
    },
)


# add baxter flipping environment to Gym
register(
    id="furniture-baxter-flip-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "name": "FurnitureBaxterToyTableEnv",
        "furniture_id": 1,
        "background": "Interior",
        "port": 1050,
    },
)


register(
    id="furniture-sawyer-densereward-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={"name": "FurnitureSawyerDenseRewardEnv", "unity": False},
)
