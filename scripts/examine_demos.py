import pickle
import os
import numpy as np

from env.models import furniture_names


base_dir = 'demos'
demo_dirs = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir)]
for demo_dir in demo_dirs:
    if 'XX' in demo_dir and '.zip' not in demo_dir: 
        print('demo_dir', demo_dir)
        for demo_name in os.listdir(demo_dir):
            if '.pkl' in demo_name:
                filepath = os.path.join(demo_dir, demo_name)
                with open(filepath, "rb") as f:
                    demo = pickle.load(f)
                    if len(demo['qpos']) != len(demo['obs']) or len(demo['obs']) != 1 + len(demo['actions']):
                        print(demo_name, 'has different lengths', len(demo['qpos']), len(demo['obs']), 1 + len(demo['actions']))
                print(len(demo['qpos']), len(demo['obs']), len(demo['actions']), len(demo['rewards']))


#duplicate check
# for i in range(0, len(demo_dirs), 2):
#     demo_dir = os.path.join('demos', demo_dirs[i])
#     demo_dir2 = os.path.join('demos', demo_dirs[i+1])
#     demos = sorted(os.listdir(demo_dir))
#     demos2 = sorted(os.listdir(demo_dir2))
#     for j in range(2, 99, 20):
#         for k in range(j, 99, 20):
#             filepath = os.path.join(demo_dir, demos[j])
#             filepath2 = os.path.join(demo_dir2, demos2[k])
#             with open(filepath, "rb") as f:
#                 demo = pickle.load(f)
#             with open(filepath2, "rb") as f2:
#                 demo2 = pickle.load(f2)
#             if (demo['obs'][0]['object_ob'] == demo2['obs'][0]['object_ob']).all():
#                 print(filepath, 'equals', filepath2)


