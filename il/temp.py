from __future__ import print_function
import os
import os.path
import numpy as np
import sys
import os

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join('..', '.')))
from env.models import furniture_names, furniture_name2id


def getDemos(tgtfilestr):
    demos = []
    base_folder = '../demos'
    for f in os.listdir(base_folder):
        print(f)
        if os.path.isfile(os.path.join(base_folder,f)) and tgtfilestr in f:
            demos.append(f)
    return demos

tgtfilestr = 'Cursor_swivel_chair_0700'
demos = getDemos(tgtfilestr)
print('demos', demos)

