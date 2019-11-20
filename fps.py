import time
import argparse

import imageio
import numpy as np

from env import make_env
from env.models import furniture_names, background_names
from util import str2bool


# available agents
agent_names = ['Baxter', 'Sawyer', 'Cursor']

# available furnitures
furniture_names

# available background scenes
background_names


def argsparser():
    """
    Returns argument parser for furniture assembly environment.
    """
    parser = argparse.ArgumentParser("Demo for IKEA Furniture Assembly Environment")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--all', type=str2bool, default=False)

    import config.furniture as furniture_config
    furniture_config.add_argument(parser)

    args = parser.parse_args()
    return args


def main(args):
    """
    Inputs types of agent, furniture model, and background and measure FPS.
    """
    print("IKEA Furniture Assembly Environment!")

    # choose an agent
    print()
    print("Supported robots:\n")
    for i, agent in enumerate(agent_names):
        print('{}: {}'.format(i, agent))
    print()
    try:
        s = input("Choose an agent (enter a number from 0 to {}): ".format(len(agent_names) - 1))
        k = int(s)
        agent_name = agent_names[k]
    except:
        print("Input is not valid. Use 0 by default.")
        agent_name = agent_names[0]


    # choose a furniture model
    print()
    print("Supported furniture:\n")
    for i, furniture_name in enumerate(furniture_names):
        print('{}: {}'.format(i, furniture_name))
    print()
    try:
        s = input("Choose a furniture model (enter a number from 0 to {}): ".format(len(furniture_names) - 1))
        furniture_id = int(s)
        furniture_name = furniture_names[furniture_id]
    except:
        print("Input is not valid. Use 0 by default.")
        furniture_id = 0
        furniture_name = furniture_names[0]


    # choose a background scene
    print()
    print("Supported backgrounds:\n")
    for i, background in enumerate(background_names):
        print('{}: {}'.format(i, background))
    print()
    try:
        s = input("Choose an agent (enter a number from 0 to {}): ".format(len(background_names) - 1))
        k = int(s)
        background_name = background_names[k]
    except:
        print("Input is not valid. Use 0 by default.")
        background_name = background_names[0]


    # set parameters for the environment (env, furniture_id, background)
    env_name = 'Furniture{}Env'.format(agent_name)
    args.env = env_name
    args.furniture_id = furniture_id
    args.background = background_name

    print()
    print("Creating environment (robot: {}, furniture: {}, background: {})".format(
        env_name, furniture_name, background_name))

    # make environment following arguments
    env = make_env(env_name, args)

    # reset the environment with new furniture and background
    env.reset(furniture_id, background_name)

    # measure FPS of simulation and rendering
    done = False
    st = time.time()
    step = 0
    while not done and step < 500:
        step += 1
        ob, rew, done, info = env.step(env.action_space.sample())

    print('fps = {}'.format(step / (time.time() - st)))

    # close the environment instance
    env.close()


def test_all(args):
    """
    Measure FPS of all configurations.
    """

    agent_types = ['Baxter_ik', 'Baxter_impedance', 'Sawyer_ik', 'Sawyer_impedance', 'Cursor_ik']
    rendering_qualities = ['no_200', 'low_200', 'high_200', 'low_500', 'high_500']
    furniture_ids = [0, 9, 6]

    results = {}
    for agent in agent_types:
        results[agent] = {}
        for rendering in rendering_qualities:
            results[agent][rendering] = {}
            for furniture_id in furniture_ids:
                if rendering.startswith('no'):
                    args.unity = False
                    args.visual_ob = False
                elif rendering.startswith('low'):
                    args.unity = True
                    args.quality = 0
                    args.visual_ob = True
                elif rendering.startswith('high'):
                    args.unity = True
                    args.quality = 4
                    args.visual_ob = True

                if '200' in rendering:
                    args.screen_width = 200
                    args.screen_height = 200
                else:
                    args.screen_width = 500
                    args.screen_height = 500

                background_name = 'Simple'

                # set parameters for the environment (env, furniture_id, background)
                env_name = 'Furniture{}Env'.format(agent.split('_')[0])
                args.env = env_name
                args.control_type = agent.split('_')[1]
                args.furniture_id = furniture_id
                args.background = background_name

                print()
                print("Creating environment (robot: {}, furniture: {}, background: {})".format(
                    env_name, furniture_names[furniture_id], background_name))

                FPS = []
                for i in range(2):
                    # make environment following arguments
                    env = make_env(env_name, args)

                    # reset the environment with new furniture and background
                    env.reset(furniture_id, background_name)

                    # measure FPS of simulation and rendering
                    done = False
                    st = time.time()
                    step = 0
                    while not done and step < 500:
                        step += 1
                        ob, rew, done, info = env.step(env.action_space.sample())

                    FPS.append(step / (time.time() - st))

                    # close the environment instance
                    env.close()

                print('fps = {:.2f}'.format(np.mean(FPS)))
                results[agent][rendering][furniture_id] = np.mean(FPS)

    # output summary
    for agent in agent_types:
        print(agent)
        for rendering in rendering_qualities:
            output = "\t".join([
                '{:.2f}'.format(results[agent][rendering][furniture_id])
                for furniture_id in furniture_ids
            ])
            print(rendering, output)


if __name__ == '__main__':
    args = argsparser()
    if args.all:
        test_all(args)
    else:
        main(args)

