from __future__ import print_function
import os
import os.path
import numpy as np
import sys
import os
import torch

import pickle

from torch.utils.data import Dataset, DataLoader

from env.models import furniture_names, furniture_name2id, agent_names

class BCDataset(Dataset):
    
    def __init__(self, agent_name, furniture_name, train=True, transform=None, target_transform=None, download=False):
        self.root = 'demos/'
        assert agent_name in agent_names
        assert furniture_name in furniture_names
        tgtfilestr = agent_name + '_' + furniture_name 
        demo_files = self.getDemos(tgtfilestr)

        self.train = train  # training set or test set

        self.data = []
        self.targets = []
        self.obs_space = None
        self.action_space = None

        # now load the picked numpy arrays
        for demo_file in demo_files:
            file_path = os.path.join(self.root, demo_file)
            with open(file_path, 'rb') as f:
                demo = pickle.load(f)
                for obs in demo['qpos']:
                    #add data
                    qpos = []
                    for x in obs.values():
                        qpos.extend(x)
                    if self.obs_space is None:
                        self.obs_space = len(qpos)
                    self.data.append(torch.tensor(qpos, dtype=torch.double))
                # add targets
                for acts in demo['actions']:
                    if self.action_space is None:
                        self.action_space = len(acts)
                    self.targets.append(torch.tensor(acts, dtype=torch.double))
        # print(len(self.data), len(self.targets))
        self.data = self.data[:-1]
        assert len(self.data) == len(self.targets)

    def getDemos(self, tgtfilestr):
        demos = []
        for f in os.listdir(self.root):
            if os.path.isfile(os.path.join(self.root,f)) and tgtfilestr in f:
                demos.append(f)
        return demos

    def getObsActs(self):
        return self.obs_space, self.action_space

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        obs, target = self.data[index], self.targets[index]

        # if self.transform is not None:
        #     obs = self.transform(obs)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return obs, target


    def __len__(self):
        return len(self.data)
