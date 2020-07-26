import gdown
import os
from zipfile import ZipFile

qpos_demos = {
    'Sawyer_bench_bjursta_0210_00XX.zip': '1uGeamzI5VkNNjCITSpitg-BivcIiQ0iB',
    'Sawyer_bench_bjursta_0210_01XX.zip': '1RwzSFn1dRDkWfd9dWmKRZNav8B43H6tP',
    'Sawyer_table_bjorkudden_0207_00XX.zip': '1tUyLxpUo_IFakgXtRFPNjI3u6ISTMIQz',
    'Sawyer_table_bjorkudden_0207_01XX.zip': '1shslsnTYSyscJyEXwRWFVHWFo4Ruj10Y',
    'Sawyer_table_lack_0825_00XX.zip': '1IN_H79aa9ndcuckmpXXlEJcoKEzKFLN-',
    'Sawyer_table_lack_0825_01XX.zip': '1gCeiJ2XN5O5acudxq37A3JQqg162vO4R',
    'Sawyer_toy_table_00XX.zip': '14F-6wgVpz3P_sGhJlU7gKZ-qFTcDySzH',
    'Sawyer_toy_table_01XX.zip': '16gRRYaLLJwrWLhUuI0v8y_9FW7nbqqCs',
    'Sawyer_table_dockstra_0279_00XX.zip': '1UOkgSBgIa34cRKySCpwstxJ0IpDcYnGQ',
    'Sawyer_table_dockstra_0279_01XX.zip': '1wusFZLDsq9DCRf_U9DEPnjSdmWvrgY3U',
    'Sawyer_chair_agne_0007_00XX.zip': '1DCmpI0_5n65UOzGC6-86D-BAyvyxXX8u',
    'Sawyer_chair_agne_0007_01XX.zip': '1Au3TXYXbJt-_3fhYQpEYg9kk6SpJgTcE',
    'Sawyer_chair_ingolf_0650_00XX.zip': '1tHaucRmqBIwDqi0DnUikDxM2WQEA3l3c',
    'Sawyer_chair_ingolf_0650_01XX.zip': '1Pw65zAF78ZoGLR0WpkcygfuoIO9a_xrL',
    'Sawyer_table_liden_0920_00XX.zip': '1SAgycZb0A6SBaufXv7gHiJ2Cm98eUPbi',
    'Sawyer_table_liden_0920_01XX.zip': '16ZvJyJiU5hmZQ_iy8ZPF4xZQXv3cuV0x',
    'Sawyer_chair_bernhard_0146_01XX.zip': '1K9Op9aHfoAFLaA1pTAHfAtiCTlMz1IIX',
    'Sawyer_chair_bernhard_0146_00XX.zip': '1GG_oZbIKTQotIxutTC8a9J3UDyfzkBfs'
}

qvel_demos = {
    'Sawyer_chair_agne_0007_00XX.zip': '1-m4Isy4EFcpOO0IV42VqdzsZTEqbr6y4',
    'Sawyer_chair_agne_0007_01XX.zip': '1gqBhLlZCvn0Xpb7h4E_U316E9kHz60j8',
}

# url = 'https://drive.google.com/uc?id=' + unique google drive ID
# compression format = '.zip'

demo_type = int(input("Enter (1) for qpos only demos\n" \
                      "      (2) for qpos and qvel demos"))

if demo_type == 1:
    demos = qpos_demos
else:
    demos= qvel_demos


for key, value in demos.items():
    url = 'https://drive.google.com/uc?id=' + value
    outfile = os.path.join('demos', key)
    if os.path.exists(outfile):
        print('already downloaded', outfile)
    else:
        gdown.download(url, outfile, quiet=False)


answer = input("Do you want to unzip demos? [y/n] ")

if answer == "y":
    for key in demos.keys():
        furniture_name = key.rsplit("_", 1)[0]
        demo_path = os.path.join("demos", furniture_name)
        os.makedirs(demo_path, exist_ok=True)
        zip_file = os.path.join("demos", key)
        with ZipFile(zip_file, "r") as zf:
            zf.extractall(demo_path)

