"""
Human control for the IKEA furniture assembly environment.
The user will learn how to configure the environment and control it.
If you're coming from the demo_demonstration script:
Pass --record_demo True and press Y to save the the current scene
into demos/test.pkl.
"""

import gym
import hydra
from omegaconf import OmegaConf, DictConfig

# from .env import *
from . import agent_names, background_names, furniture_names

# available agents
agent_names

# available furnitures
furniture_names

# available background scenes
background_names


def main_vr_test(cfg):
    agent_name = agent_names[1]
    furniture_name = "three_blocks"
    background_name = background_names[0]

    # set correct environment name based on agent_name
    env_name = "IKEA{}-v0".format(agent_name)

    # make environment following arguments
    env = gym.make(
        env_name, furniture_name=furniture_name, background=background_name, ikea_cfg=cfg
    )

    # manual control of agent using Oculus Quest2
    env.run_vr_oculus()

    # close the environment instance
    env.close()


@hydra.main(config_path="config", config_name="default_config")
def main(cfg: DictConfig) -> None:
    # make config writable
    OmegaConf.set_struct(cfg, False)

    main_vr_test(cfg.env)


if __name__ == "__main__":
    main()
