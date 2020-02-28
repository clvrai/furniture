"""
Demonstration demo for the IKEA furniture assembly environment.
This script will take the user through the 1) playback of existing
demonstrations and 2) recording and playback of their own demos.
"""

import argparse

import numpy as np

from env import make_env
from env.models import furniture_names, background_names, furniture_name2id
from util import str2bool, parseDemoName
from util.video_recorder import VideoRecorder


# available agents
agent_names = ['Baxter', 'Sawyer', 'Cursor']

# available furnitures
furniture_names
# print(furniture_names)

# available background scenes
background_names


def argsparser():
    """
    Returns argument parser for furniture assembly environment.
    """
    parser = argparse.ArgumentParser("Demo for IKEA Furniture Assembly Environment")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)

    import config.furniture as furniture_config
    furniture_config.add_argument(parser)

    parser.set_defaults(render=True)
    parser.set_defaults(load_demo='demos/Cursor_swivel_chair_0700.pkl')

    args = parser.parse_args()
    return args


def main(args):
    """
    Shows the user how to record and playback demonstrations
    """

    background_name = background_names[1]
    print("In this demo, we'll show you how to record and playback demonstrations\n")
    print("Choice 1: Playback existing demo")
    print("Choice 2: Record your own demonstration")
    print("Choice 3: Playback your own recording")
    print()

    choice = int(input("Press 1, 2, or 3:  "))
    print()

    if choice == 1:
        print()
        print("Let's begin by playing back an existing demonstration.")
        print("We'll use the run_demo function to run a demo passed through --load_demo")
        agent_name, furniture_name = parseDemoName(parser.load_demo)
        furniture_id = furniture_name2id[furniture_name]

        # set parameters for the environment (env, furniture_id, background)
        env_name = 'Furniture{}Env'.format(agent_name)
        args.env = env_name
        args.furniture_id = furniture_id
        args.background = background_name

        print()
        print("Creating environment (robot: {}, furniture: {}, background: {})".format(
            env_name, furniture_name, background_name))
        env = make_env(env_name, args)
        env.run_demo(args)
        env.close()

        print('Check out the video "test.mp4"! Pretty cool right? You can look at the run_demo function for more details.')
        print()

    elif choice == 2:
        print('Now you can try recording your own demonstrations.')
        print('Run "python -m demo_manual --record_demo True" to record your own demonstration.')
        print('Move stuff around and then press Y to save the demo.')
        print('It will be saved to demos/test.pkl.')

    elif choice == 3:
        demo_path = input('Enter the path to your demo (e.g. demos/test.pkl):  ')

        agent_name = input("What was the agent (Sawyer, Baxter, Cursor)?: ")
        assert agent_name in ['Sawyer', 'Baxter', 'Cursor']

        furniture_name = input("What was the furniture name?: ")
        furniture_id = furniture_name2id[furniture_name]

        # set parameters for the environment (env, furniture_id, background)
        env_name = 'Furniture{}Env'.format(agent_name)
        args.load_demo = demo_path
        args.env = env_name
        args.furniture_id = furniture_id
        args.background = background_name

        print()
        print("Creating environment (robot: {}, furniture: {}, background: {})".format(
            env_name, furniture_name, background_name))
        env = make_env(env_name, args)
        env.run_demo(args)
        env.close()

    else:
        print('You entered wrong input %d' % choice)


if __name__ == '__main__':
    args = argsparser()
    main(args)

