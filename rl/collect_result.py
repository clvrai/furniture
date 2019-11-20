from glob import glob
from collections import defaultdict

import numpy as np
import h5py


files = glob('result/*.hdf5')
files.sort()

keys = ['episode_success', 'reward_goal_dist']
results = {k: defaultdict(list) for k in keys}

for file_name in files:
    run_name = file_name.rsplit('.', 2)[0].split('/')[-1]
    with h5py.File(file_name, 'r') as hf:
        for key in keys:
            results[key][run_name].extend(hf[key])

for k in keys:
    print(k)
    for run_name, v in results[k].items():
        if k == 'reward_goal_dist':
            values = [float(x) / 200 for x in v]
        else:
            values = [float(x) for x in v]
        print('{}\t{:.03f} $\\pm$ {:.03f}'.format(run_name, np.mean(values), np.std(values)))

