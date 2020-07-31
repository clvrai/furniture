""" Code used to collect evaluation results. """

from glob import glob
from collections import defaultdict

import numpy as np
import h5py


files = glob('result/*.hdf5')
files.sort()

keys = ['episode_success', 'episode_num_connected', 'pick_reward', 'success_reward', 'touch_reward']
results = {k: defaultdict(list) for k in keys}

for file_name in files:
    run_name = file_name.split('.')[-3]
    with h5py.File(file_name, 'r') as hf:
        connect = np.array(hf["episode_num_connected"])
        pick = np.array(hf["pick_reward"]) / 100
        touch = np.array(hf["touch_reward"]) / 10
        success = np.array(hf["episode_success"])

        N = len(connect)
        phase = []
        for i in range(N):
            phase.append(connect[i] + min(pick[i], connect[i]) + min(touch[i], pick[i], connect[i]))

        print("{}".format(run_name))
        # print("{:.03f} & {:.03f} & {:.03f}".format(np.mean(phase), np.mean(pick), np.mean(connect)))
        print("[{:.03f}, {:.03f}, {:.03f}, {:.03f}],".format(np.mean(pick), np.std(pick), np.mean(connect), np.std(connect)))

