"""
Human control for the IKEA furniture assembly environment.
"""

import gym
import hydra
from omegaconf import OmegaConf, DictConfig

from furniture import agent_names  # list of available agents
from furniture import background_names  # list of available background scenes
from furniture import furniture_names  # list of available furnitures


def main_demo_test(cfg):
    # specify agent, furniture, and background
    agent_name = agent_names[1]
    furniture_name = "three_blocks"
    background_name = background_names[0]

    # set correct environment name based on agent_name
    env_name = "IKEA{}-v0".format(agent_name)

    # make environment following arguments
    env = gym.make(
        env_name,
        furniture_name=furniture_name,
        background=background_name,
        ikea_cfg=cfg.ikea_cfg,
    )

    # manual control of agent using keyboard
    env.run_demo()

    # close the environment instance
    env.close()


@hydra.main(config_path="config", config_name="default_config")
def main(cfg: DictConfig) -> None:
    # make config writable
    OmegaConf.set_struct(cfg, False)

    # set environment config for keyboard control
    cfg.env.ikea_cfg.unity.use_unity = True
    cfg.env.ikea_cfg.render = True
    cfg.env.ikea_cfg.control_type = "ik"
    cfg.env.ikea_cfg.max_episode_steps = 10000
    cfg.env.ikea_cfg.screen_size = [1024, 1024]

    cfg.env.ikea_cfg.render_mode = "file"
    cfg.env.ikea_cfg.record_vid = True
    cfg.env.ikea_cfg.depth_ob = True
    cfg.env.ikea_cfg.segmentation_ob = True
    cfg.env.ikea_cfg.load_demo = "demos/Sawyer_three_blocks_0002.pkl"
    # cfg.env.ikea_cfg.load_demo = "demo/Sawyer_chair_agne_0007_0000.pkl"
    # cfg.env.ikea_cfg.camera_ids = [0, 1, 2]

    main_demo_test(cfg.env)


if __name__ == "__main__":
    main()
