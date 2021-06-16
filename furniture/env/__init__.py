""" Define all environments and provide helper functions to load environments. """

# OpenAI gym interface
from gym.envs.registration import register

from .base import make_env, make_vec_env

# register all environment to use
from .furniture_baxter import FurnitureBaxterEnv
from .furniture_cursor import FurnitureCursorEnv
from .furniture_jaco import FurnitureJacoEnv
from .furniture_panda import FurniturePandaEnv
from .furniture_sawyer import FurnitureSawyerEnv
from .furniture_fetch import FurnitureFetchEnv
from .furniture_sawyer_dense import FurnitureSawyerDenseRewardEnv


# add cursor environment to Gym
register(
    id="IKEACursor-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "id": "IKEACursor-v0",
        "name": "FurnitureCursorEnv",
        "furniture_id": 0,
        "background": "Lab",
        "port": 1050,
    },
)


# add sawyer environment to Gym
register(
    id="IKEASawyer-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "id": "IKEASawyer-v0",
        "name": "FurnitureSawyerEnv",
        "furniture_name": "swivel_chair_0700",
        "background": "Industrial",
        "port": 1050,
    },
)


# add baxter environment to Gym
register(
    id="IKEABaxter-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "id": "IKEABaxter-v0",
        "name": "FurnitureBaxterEnv",
        "furniture_id": 1,
        "background": "Interior",
        "port": 1050,
    },
)


# add jaco environment to Gym
register(
    id="IKEAJaco-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "id": "IKEAJaco-v0",
        "name": "FurnitureJacoEnv",
        "furniture_id": 1,
        "background": "Interior",
        "port": 1050,
    },
)


# add panda environment to Gym
register(
    id="IKEAPanda-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "id": "IKEAPanda-v0",
        "name": "FurniturePandaEnv",
        "furniture_id": 1,
        "background": "Interior",
        "port": 1050,
    },
)


# add panda environment to Gym
register(
    id="IKEAFetch-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "id": "IKEAFetch-v0",
        "name": "FurnitureFetchEnv",
        "furniture_id": 59,
        "background": "Interior",
        "port": 1050,
    },
)


# add sawyer dense reward environment to Gym
register(
    id="IKEASawyerDense-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={"id": "IKEASawyerDense-v0", "name": "FurnitureSawyerDenseRewardEnv", "unity": False},
)


register(
    id="furniture-sawyer-densereward-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={"id": "IKEASawyerDense-v0", "name": "FurnitureSawyerDenseRewardEnv", "unity": False},
)
