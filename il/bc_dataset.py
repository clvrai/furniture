import os
import pickle

import torch
from torch.utils.data import Dataset

from env.models import furniture_names, agent_names


class ILDataset(Dataset):
    """ Dataset class for Imitation Learning. """

    def __init__(self, agent_name, furniture_name, train=True, transform=None, target_transform=None, download=False):
        self.root = 'demos/'
        self.train = train  # training set or test set
        # TODO: split dataset into train/val

        assert agent_name in agent_names
        assert furniture_name in furniture_names

        self._obs = []
        self._acs = []
        self._rews = []
        self._obs_space = None
        self._action_space = None

        demo_files = self._get_demo_files(agent_name + '_' + furniture_name)

        # now load the picked numpy arrays
        for demo_file in demo_files:
            file_path = os.path.join(self.root, demo_file)
            with open(file_path, 'rb') as f:
                demo = pickle.load(f)

                # add observations
                for ob in demo['qpos']:
                    qpos = []
                    for x in ob.values():
                        qpos.extend(x)
                    if self._obs_space is None:
                        self._obs_space = len(qpos)
                    self._data.append(torch.tensor(qpos, dtype=torch.double))

                # add actions
                for ac in demo['actions']:
                    if self._action_space is None:
                        self._action_space = len(ac)
                    self._acs.append(torch.tensor(ac, dtype=torch.double))

                # add rewards
                for rew in demo['rewards']:
                    self._rews.append(torch.tensor(rew, dtype=torch.double))

        assert len(self._obs) == len(self._acs) + 1

    def _get_demo_files(self, tgtfilestr):
        demos = []
        for f in os.listdir(self.root):
            if os.path.isfile(os.path.join(self.root, f)) and tgtfilestr in f:
                demos.append(f)
        return demos

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (ob, ac) where target is index of the target class.
        """
        ob, ac, rew = self._obs[index], self._acs[index], self._rews[index]

        return ob, ac, rew

    def __len__(self):
        return len(self._acs)

