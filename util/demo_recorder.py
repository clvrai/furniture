import os
import pickle


class DemoRecorder(object):
    def __init__(self, demo_dir='./'):
        self._actions = []
        self._qpos = []

        self._demo_dir = demo_dir
        os.makedirs(demo_dir, exist_ok=True)

    def reset(self):
        self._actions = []
        self._qpos = []

    def add(self, qpos=None, action=None):
        if action is not None:
            self._actions.append(action)
        if qpos is not None:
            self._qpos.append(qpos)

    def save(self, fname):
        path = os.path.join(self._demo_dir, fname)
        demo = {'qpos': self._qpos, 'actions': self._actions}
        with open(path, 'wb') as f:
            pickle.dump(demo, f)
        self.reset()

