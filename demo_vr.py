import sys
import time
import openvr
import argparse

from env import make_env
from env.models import furniture_names, background_names, agent_names
from util import str2bool
from util.triad_openvr import triad_openvr


# available agents
agent_names


def main(args):
    """
    Inputs types of agent, furniture model, and background and simulates the environment.
    """
    agent_name = agent_names[0]

    # set parameters for the environment (env, furniture_id, background)
    env_name = "Furniture{}Env".format(agent_name)
    args.env = env_name

    env = make_env(env_name, args)
    env.run_vr(args)
    env.close()


def argsparser():
    """
    Returns argument parser for furniture assembly environment.
    """
    parser = argparse.ArgumentParser("Demo for IKEA Furniture Assembly Environment")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", type=str2bool, default=False)

    import config.furniture as furniture_config

    furniture_config.add_argument(parser)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
