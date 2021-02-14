import gdown
import os
from zipfile import ZipFile

demos = {
    "Sawyer_chair_agne_0007_00XX.zip": "1-lVTCH4oPq22cLC4Mmia9AKqzDIIVDO0",
    'Sawyer_table_dockstra_0279_00XX': '1QAchFmYpQGqa6zaZ2QeZH5ET-iuyerU0',
    "Sawyer_bench_bjursta_0210_00XX.zip": "12b8_j1mC8-pgotjARF1aTcqH2T7FNHNF",
    "Sawyer_table_bjorkudden_0207_00XX.zip": "19DA5M2iPvOYa9KG54uIxOhNF0r2zXClK",
    "Sawyer_table_lack_0825_00XX.zip": "1BrgbaE9Wx-Si7VtXpUJSHRrRnyqdLJA7",
    "Sawyer_toy_table_00XX.zip": "1Wg6oxkiiOX8DsYVdr7sYNmYnSdaxIskc",
    "Sawyer_chair_ingolf_0650_00XX.zip": "1i9A9CVPys7LiUnePRn4OkVgczRjqT4kZ",
    "Sawyer_chair_bernhard_0146_00XX.zip": "1nWnHDSQq33INXdOmIAL_28wrd6BKEUr-",
}

# url = 'https://drive.google.com/uc?id=' + unique google drive ID
# compression format = '.zip'

for key, value in demos.items():
    url = "https://drive.google.com/uc?id=" + value
    outfile = os.path.join("demos", key)
    if os.path.exists(outfile):
        print("already downloaded", outfile)
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
