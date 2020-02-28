import os
import pickle
import glob


class DemoRecorder(object):
    def __init__(self, demo_dir='./'):
        self._actions = []
        self._qpos = []
        self._rewards = []

        self._demo_dir = demo_dir
        os.makedirs(demo_dir, exist_ok=True)

    def reset(self):
        self._actions = []
        self._qpos = []
        self._rewards = []

    def add(self, qpos=None, action=None, reward=None):
        if action is not None:
            self._actions.append(action)
        if qpos is not None:
            self._qpos.append(qpos)
        if reward is not None:
            self._rewards.append(reward)

    def save(self, prefix):
        count = min(9999, self._get_demo_count(prefix))
        fname = prefix + '_{:04d}.pkl'.format(count)
        path = os.path.join(self._demo_dir, fname)
        demo = {'qpos': self._qpos,
                'actions': self._actions,
                'rewards': self._rewards}
        with open(path, 'wb') as f:
            pickle.dump(demo, f)
        self.reset()

    def _get_demo_count(self, prefix):
        return len(glob.glob(os.path.join(self.demo_dir, prefix) + "_*"))

