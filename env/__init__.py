""" Define all environments and provide helper functions to load environments. """

from env.base import make_env, make_vec_env

# register all environment to use
from env.furniture_baxter import FurnitureBaxterEnv
from env.furniture_sawyer import FurnitureSawyerEnv
from env.furniture_cursor import FurnitureCursorEnv
from env.furniture_jaco import FurnitureJacoEnv
from env.furniture_panda import FurniturePandaEnv

from env.furniture_baxter_block import FurnitureBaxterBlockEnv
from env.furniture_cursor_toytable import FurnitureCursorToyTableEnv
from env.furniture_sawyer_toytable import FurnitureSawyerToyTableEnv

# OpenAI gym interface
from gym.envs.registration import register


# add cursor environment to Gym
register(
    id="furniture-cursor-v0",
    entry_point="env.furniture_gym:FurnitureGym",
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
    entry_point="env.furniture_gym:FurnitureGym",
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
    entry_point="env.furniture_gym:FurnitureGym",
    kwargs={
        "name": "FurnitureBaxterEnv",
        "furniture_id": 1,
        "background": "Interior",
        "port": 1050,
    },
)

register(
    id="furniture-jaco-v0",
    entry_point="env.furniture_gym:FurnitureGym",
    kwargs={
        "name": "FurnitureJacoEnv",
        "furniture_id": 1,
        "background": "Interior",
        "port": 1050,
    },
)


register(
    id="furniture-panda-v0",
    entry_point="env.furniture_gym:FurnitureGym",
    kwargs={
        "name": "FurniturePandaEnv",
        "furniture_id": 1,
        "background": "Interior",
        "port": 1050,
    },
)
