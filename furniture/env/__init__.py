""" Define all environments and provide helper functions to load environments. """

# OpenAI gym interface
from gym import register

# register all environment to use
from .furniture_baxter import FurnitureBaxterEnv
from .furniture_cursor import FurnitureCursorEnv
from .furniture_jaco import FurnitureJacoEnv
from .furniture_panda import FurniturePandaEnv
from .furniture_sawyer import FurnitureSawyerEnv
from .furniture_fetch import FurnitureFetchEnv
from .furniture_sawyer_dense import FurnitureSawyerDenseRewardEnv


entry_point = "furniture.env.furniture_gym:FurnitureGym"


envs = {
    "IKEACursor-v0": {
        "class_name": "FurnitureCursorEnv",
        "config_name": "ikea_cursor",
        "agent_type": "Cursor",
        "furniture_name": "three_blocks",
        "unity": {"background": "Lab"},
    },
    "IKEASawyer-v0": {
        "class_name": "FurnitureSawyerEnv",
        "config_name": "ikea",
        "agent_type": "Sawyer",
        "furniture_name": "swivel_chair_0700",
        "unity": {"background": "Industrial"},
    },
    "IKEABaxter-v0": {
        "class_name": "FurnitureBaxterEnv",
        "config_name": "ikea",
        "agent_type": "Baxter",
        "furniture_name": "three_blocks",
        "unity": {"background": "Interior"},
    },
    "IKEAJaco-v0": {
        "class_name": "FurnitureJacoEnv",
        "config_name": "ikea",
        "agent_type": "Jaco",
        "furniture_name": "three_blocks",
        "unity": {"background": "Interior"},
    },
    "IKEAPanda-v0": {
        "class_name": "FurniturePandaEnv",
        "config_name": "ikea",
        "agent_type": "Panda",
        "furniture_name": "three_blocks",
        "unity": {"background": "Interior"},
    },
    "IKEAFetch-v0": {
        "class_name": "FurnitureFetchEnv",
        "config_name": "ikea",
        "agent_type": "Fetch",
        "furniture_name": "three_blocks",
        "unity": {"background": "Interior"},
    },
    "IKEASawyerDense-v0": {
        "class_name": "FurnitureSawyerDenseRewardEnv",
        "config_name": "ikea_dense",
        "agent_type": "Sawyer",
        "furniture_name": "three_blocks",
        "unity": {"use_unity": False},
    },
}


for id, kwargs in envs.items():
    register(id=id, entry_point=entry_point, order_enforce=False, kwargs=kwargs)
