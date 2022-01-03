""" Define cursor environment class FurnitureCursorEnv. """

from collections import OrderedDict

import numpy as np
import gym.spaces

from . import transform_utils as T
from .furniture import FurnitureEnv
from ..util.logger import logger


class FurnitureCursorEnv(FurnitureEnv):
    """
    Cursor environment.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: configurations for the environment.
        """
        cfg.agent_type = "Cursor"

        super().__init__(cfg)

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
            ob_space.spaces["robot_ob"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=((3 + 1) * 2,),
            )

        return ob_space

    @property
    def dof(self):
        """
        Returns the DoF of the curosr agent.
        """
        assert self._control_type == "ik"
        dof = (3 + 3 + 1) * 2 + 1  # (move, rotate, select) * 2 + connect
        return dof

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        ob, _, done, _ = super()._step(a)

        reward, _done, info = self._compute_reward(a)

        if self._success:
            logger.info("Success!")

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
            robot_states["cursor_state"] = np.array(
                [
                    self._cursor_selected[0] is not None,
                    self._cursor_selected[1] is not None,
                ]
            )

            state["robot_ob"] = np.concatenate(
                [x.ravel() for _, x in robot_states.items()]
            )

        return state

    def _compute_reward(self, ac):
        """
        Computes reward of the current state.
        """
        return super()._compute_reward(ac)
