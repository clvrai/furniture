import glob
import os
import pickle


class DemoRecorder(object):
    def __init__(self, demo_dir="./", metadata=None):
        self._obs = []
        self._actions = []
        self._qpos = []
        self._rewards = []
        self._metadata = metadata

        self._demo_dir = demo_dir
        os.makedirs(demo_dir, exist_ok=True)

    def reset(self):
        self._obs = []
        self._actions = []
        self._qpos = []
        self._rewards = []

    def add(self, ob=None, qpos=None, action=None, reward=None):
        if ob is not None:
            self._obs.append(ob)
        if action is not None:
            self._actions.append(action)
        if qpos is not None:
            self._qpos.append(qpos)
        if reward is not None:
            self._rewards.append(reward)

    def save(self, prefix: str):
        count = min(9999, self._get_demo_count(prefix))
        fname = prefix + "_{:04d}.pkl".format(count)
        path = os.path.join(self._demo_dir, fname)
        demo = {
            "qpos": self._qpos,
            "obs": self._obs,
            "actions": self._actions,
            "rewards": self._rewards,
            "metadata": self._metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(demo, f)
        self.reset()

    def _get_demo_count(self, prefix):
        return len(glob.glob(os.path.join(self._demo_dir, prefix) + "_*"))

    def __len__(self) -> int:
        return len(self._obs)
