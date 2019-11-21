"""
How to use the IKEA furniture assembly environment with gym interface.
For parallel execution, set `--num_envs` flag to larger than 1.
"""

import time

import gym
import imageio

from env import make_vec_env


def main(config):
    """
    Makes a gym environment with name @env_id and runs one episode.
    If @config.num_env is larger than 1, it will use SubprocVecEnv to run
    multiple environments in parallel.
    """
    num_env = config.num_env
    env_id = config.env_id

    if num_env == 1:
        # create a gym environment instance
        env = gym.make(env_id, **config.__dict__)

        # reset the environment
        observation = env.reset()
        done = False

        st = time.time()
        for i in range(500):
            # take a step with a randomly sampled action
            observation, reward, done, info = env.step(env.action_space.sample())
        print('FPS = {}'.format(500 / (time.time() - st)))

        # close the environment instance
        env.close()

    else:
        envs = make_vec_env(env_id, num_env, config=config)
        envs.reset()
        img = envs.render('rgb_array')
        imageio.imwrite('vec_env_image.png', img)

        st = time.time()
        for i in range(500):
            # take a step with a randomly sampled action
            observation, reward, done, info = envs.step(
                [envs.action_space.sample() for _ in range(num_env)]
            )
        print('FPS = {}'.format((500 * num_env) / (time.time() - st)))

        envs.close()


def argsparser():
    """
    Returns argument parser for furniture assembly environment.
    """
    import argparse
    from util import str2bool

    parser = argparse.ArgumentParser("Demo for IKEA Furniture Assembly Environment")
    parser.add_argument('--env_id', type=str, default='furniture-baxter-v0')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)

    import config.furniture as furniture_config
    furniture_config.add_argument(parser)

    parser.set_defaults(visual_ob=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argsparser()
    main(args)
