import os
import pickle
import glob

import torch
from torch.utils.data import Dataset


class ILDataset(Dataset):
    """ Dataset class for Imitation Learning. """

    def __init__(self, path, train=True, transform=None, target_transform=None, download=False):
        self.train = train  # training set or test set
        # TODO: split dataset into train/val

        self._obs = []
        self._acs = []
        self._rews = []

        demo_files = self._get_demo_files(path)

        # now load the picked numpy arrays
        for file_path in demo_files:
            with open(file_path, 'rb') as f:
                demo = pickle.load(f)

                # add observations
                for ob in demo['obs']:
                    self._obs.append(ob)

                # add actions
                for ac in demo['actions']:
                    self._acs.append(ac)

                # add rewards
                if 'rewards' in demo:
                    for rew in demo['rewards']:
                        self._rews.append(rew)

        assert len(self._obs) == len(self._acs) + 1

    def _get_demo_files(self, tgtfilestr):
        demos = []
        for f in glob.glob(tgtfilestr + "_*"):
            if os.path.isfile(f):
                demos.append(f)
        return demos

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (ob, ac) where target is index of the target class.
        """
        ob, ac = self._obs[index], self._acs[index]

        if len(self._rews) > index:
            rew = self._obs[index], self._acs[index], self._rews[index]
            return {'ob': ob, 'ac': ac, 'rew': rew}
        else:
            return {'ob': ob, 'ac': ac}


    def __len__(self):
        return len(self._acs)

