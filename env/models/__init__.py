import os
import glob
from .world import MujocoWorldBase


# path to the asset directory
assets_root = os.path.join(os.path.dirname(__file__), "assets")

# load all xmls in assets/objects/
_furniture_xmls = glob.glob(os.path.join(assets_root, "objects") + "/*.xml")
_furniture_xmls.sort()
_furniture_names = [x.rsplit('/')[-1] for x in _furniture_xmls]
furniture_xmls = ["objects/" + name for name in _furniture_names]

# list of furniture models
furniture_name2id = {
    furniture_name.split('.')[0]: i for i, furniture_name in enumerate(_furniture_names)
}
furniture_names = [furniture_name.split('.')[0] for furniture_name in _furniture_names]
furniture_ids = [i for i in range(len(furniture_names))]

# list of background names
background_names = ['Simple', 'Industrial', 'Lab', 'Garage', 'Ambient', 'NightTime', 'Interior']
