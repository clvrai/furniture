""" Define cursor environment class FurnitureCursorEnv. """

from collections import OrderedDict

import numpy as np

from env.furniture import FurnitureEnv
import env.transform_utils as T
from util.logger import logger

class FurnitureCursorEnv(FurnitureEnv):
    """
    Cursor environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.agent_type = 'Cursor'

        super().__init__(config)

        self._env_config.update({
            "success_reward": 100,
            "pos_dist": 0.1,
            "rot_dist_up": 0.9,
            "rot_dist_forward": 0.9,
            "project_dist": -1,
        })

        # turn on the gravity compensation for selected furniture pieces
        self._gravity_compensation = 1

        # requires multiple connection actions to make connection between two
        # parts.
        self._num_connect_steps = 10

        self._cursor_selected = [None, None]

    @property
    def observation_space(self):
        """
        Returns the observation space.
        """
        ob_space = super().observation_space

        if self._robot_ob:
            ob_space['robot_ob'] = [(3 + 1) * 2]

        return ob_space

    @property
    def dof(self):
        """
        Returns the DoF of the curosr agent.
        """
        assert self._control_type == 'ik'
        dof = (3 + 3 + 1) * 2 + 1  # (move, rotate, select) * 2 + connect
        return dof

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        prev_reward, _, old_info = self._compute_reward()

        ob, _, done, _ = super()._step(a)

        reward, done, info = self._compute_reward()

        connect_reward = reward - prev_reward
        info['reward_connect'] = connect_reward

        if self._success:
            logger.info('Success!')

        reward = connect_reward

        return ob, reward, done, info

    def _reset(self, furniture_id=None, background=None):
        """
        Resets simulation.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        super()._reset(furniture_id, background)

        # set two bodies for picking or assemblying
        id1 = self.sim.model.eq_obj1id[0]
        id2 = self.sim.model.eq_obj2id[0]
        self._target_body1 = self.sim.model.body_id2name(id1)
        self._target_body2 = self.sim.model.body_id2name(id2)

    def _get_obs(self):
        """
        Returns the current observation.
        """
        state = super()._get_obs()

        # proprioceptive features
        if self._robot_ob:
            robot_states = OrderedDict()
            robot_states["cursor_pos"] = self._get_cursor_pos()
            robot_states["cursor_state"] = np.array([self._cursor_selected[0] is not None,
                                                     self._cursor_selected[1] is not None])

            state['robot_ob'] = np.concatenate(
                [x.ravel() for _, x in robot_states.items()]
            )

        return state

    def _compute_reward(self):
        """
        Computes reward of the current state.
        """
        return super()._compute_reward()


def main():
    import argparse
    import config.furniture as furniture_config
    from util import str2bool

    parser = argparse.ArgumentParser()
    furniture_config.add_argument(parser)

    # change default config for Cursors
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)

    parser.set_defaults(render=True)

    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Cursor environment
    env = FurnitureCursorEnv(config)
    env.run_manual(config)


if __name__ == "__main__":
    main()
