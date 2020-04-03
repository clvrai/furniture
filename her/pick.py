import argparse

# furniture
import config.furniture as furniture_config
from config.furniture_sawyer_pick import add_argument

from . import main

parser = argparse.ArgumentParser(
    "IKEA Furniture Assembly Environment",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--env_args", type=str, default=None)
furniture_config.add_argument(parser)
# add Pick Env and SILO args
add_argument(parser)

config, unparsed = parser.parse_known_args()

main(config, unparsed)
