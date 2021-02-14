import glob
import os
import pickle

import numpy as np

from .logger import logger


class DemoRecorder(object):
    def __init__(self, demo_dir="./", metadata=None):
        self._obs = []
        self._actions = []
        self._states = []
        self._rewards = []
        self._low_level_obs = []
        self._low_level_actions = []
        self._connect_actions = []
        self._metadata = metadata

        self._demo_dir = demo_dir
        os.makedirs(demo_dir, exist_ok=True)

    def reset(self):
        self._obs = []
        self._actions = []
        self._states = []
        self._rewards = []
        self._low_level_obs = []
        self._low_level_actions = []
        self._connect_actions = []

    def add(
        self,
        ob=None,
        state=None,
        action=None,
        reward=None,
        low_level_ob=None,
        low_level_action=None,
        connect_action=None,
    ):
        if ob is not None:
            self._obs.append(ob)
        if action is not None:
            self._actions.append(action)
        if state is not None:
            self._states.append(state)
        if reward is not None:
            self._rewards.append(reward)
        if low_level_ob is not None:
            self._low_level_obs.append(low_level_ob)
        if low_level_action is not None:
            self._low_level_actions.append(low_level_action)
        if connect_action is not None:
            self._connect_actions.append(connect_action)

    def save(self, prefix, count=None):
        if count is None:
            count = min(9999, self._get_demo_count(prefix))
        fname = prefix + "{:04d}.pkl".format(count)
        path = os.path.join(self._demo_dir, fname)
        demo = {
            "states": self._states,
            "obs": self._obs,
            "actions": self._actions,
            "rewards": self._rewards,
            "low_level_obs": self._low_level_obs,
            "low_level_actions": self._low_level_actions,
            "connect_actions": self._connect_actions,
            "metadata": self._metadata,
        }

        if len(self._low_level_actions) > 0:
            for i in range(len(self._low_level_actions)):
                self._low_level_actions[i] = np.concatenate(
                    [self._low_level_actions[i], [self._connect_actions[i]]]
                )

        assert len(self._low_level_obs) == len(self._low_level_actions) + 1
        assert len(self._obs) == len(self._actions) + 1

        with open(path, "wb") as f:
            pickle.dump(demo, f)
        logger.warn("Save demo of length %d to %s", len(self._obs), path)

        self.reset()

    def _get_demo_count(self, prefix):
        return len(glob.glob(os.path.join(self._demo_dir, prefix) + "*"))
