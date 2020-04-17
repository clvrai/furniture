"""
Demonstration for RL experiments with new environment design.

This script tells you how to use our IKEA furniture assembly environment for RL
experiments and design your own reward function and task.

First, FurnitureExampleEnv shows you how to define a new task.
* `__init__`: sets environment- and task-specific configurations
* `_reset`: initializes variables when an episode is reset
* `_place_objects`: specifies the initialization of furniture parts
* `_get_obs`: includes more information for your task
* `_step`: simulates the environment and returns observation and reward
* `_compute_reward`: designs your own reward function

We describe how to collect trajectories with a random policy in `main`.

Please refer to `furniture/rl` for more advanced RL implementations.
"""


from collections import OrderedDict

import numpy as np

from env.furniture_baxter import FurnitureBaxterEnv
from env.models import furniture_names, background_names, agent_names, furniture_name2id
import env.transform_utils as T


class FurnitureExampleEnv(FurnitureBaxterEnv):
    """
    Baxter robot environment with a reaching task as an example.
    """

    def __init__(self, config):
        """
        Args:
            config: general configuration for the environment.
        """
        ###################################################
        #  Change @config before creating a MuJoCo scene  #
        ###################################################

        # set the furniture to be always the simple blocks
        config.furniture_id = furniture_name2id["block"]
        # set subtask_ob for getting target object
        config.subtask_ob = True

        # create a MuJoCo environment based on @config
        super().__init__(config)

        # set environment- and task-specific configurations
        self._env_config.update({
            "max_episode_steps": 50,
            "distance_reward": 1,
            "success_reward": 5,
        })

    def _reset(self, furniture_id=None, background=None):
        """
        Resets simulation and variables to compute reward.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        super()._reset(furniture_id, background)

        ##########################################
        # Set variables needed to compute reward #
        ##########################################

        # pick an object to reach
        assert self._subtask_part1 != -1
        self._target_body = self._object_names[self._subtask_part1]

    def _place_objects(self):
        """
        Returns the initial positions and rotations of furniture parts.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        ######################################################
        # Specify initial position and rotation of each part #
        ######################################################
        pos_init = [[-0.3, -0.2, 0.05], [0.1, -0.2, 0.05]]
        quat_init = [[1, 0, 0, 0], [1, 0, 0, 0]]
        return pos_init, quat_init

    def _get_obs(self):
        """
        Returns the current observation.
        """
        obs = super()._get_obs()
        return obs

    def _step(self, a):
        """
        Takes a simulation step with action @a.
        """
        # zero out left arm's action and only use right arm
        a[6:12] = 0

        # simulate action @a
        ob, _, _, _ = super(FurnitureBaxterEnv, self)._step(a)

        # compute your own reward
        reward, done, info = self._compute_reward(a)

        # store some information for log
        info['right_arm_action'] = a[0:6]
        info['right_gripper_action'] = a[12]

        return ob, reward, done, info

    def _compute_reward(self, a):
        """
        Computes reward for the task.
        """
        info = {}

        # control penalty
        ctrl_reward = self._ctrl_reward(a)

        # distance-based reward
        hand_pos = np.array(self.sim.data.site_xpos[self.right_eef_site_id])
        dist = T.l2_dist(hand_pos, self._get_pos(self._target_body))
        distance_reward = -self._env_config["distance_reward"] * dist

        # reward for successful reaching
        success_reward = 0
        if dist < 0.05:
            success_reward = self._env_config["success_reward"]

        # add up all rewards
        reward = ctrl_reward + distance_reward + success_reward
        done = False

        # log each component of reward
        info['reward_ctrl'] = ctrl_reward
        info['reward_distance'] = distance_reward
        info['reward_success'] = success_reward

        return reward, done, info


def main(args):
    """
    Shows basic rollout code for simulating the environment.
    """
    print("IKEA Furniture Assembly Environment!")

    # make environment following arguments
    from env import make_env
    env = make_env('FurnitureExampleEnv', args)

    # define a random policy
    def policy_action(ob):
        return env.action_space.sample()

    # define policy update
    def update_policy(rollout):
        pass

    # run one episode and collect transitions
    rollout = []
    done = False
    observation = env.reset()
    ep_length = 0

    # update unity rendering
    env.render()

    while not done:
        ep_length += 1

        # sample action from policy
        action = policy_action(observation)

        # simulate environment
        observation, reward, done, info = env.step(action)

        print('{:3d} step:  reward ({:5.3f})  action ({})'.format(
            ep_length, reward, action[:3]))

        # update unity rendering
        env.render()

        # collect transition
        rollout.append({'ob': observation,
                        'reward': reward,
                        'done': done})

    # update your network using @rollout
    update_policy(rollout)

    # close the environment instance
    env.close()


def argsparser():
    """
    Returns argument parser for furniture assembly environment.
    """
    import argparse
    import config.furniture as furniture_config
    from util import str2bool

    parser = argparse.ArgumentParser("Demo for IKEA Furniture Assembly Environment")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)

    furniture_config.add_argument(parser)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argsparser()
    main(args)

