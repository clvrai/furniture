import argparse

# furniture
from config.furniture import add_argument as add_furniture_args
from config.furniture_cursor_toytable_assemble import \
    add_argument as add_pick_args
from util import str2bool

from . import main

parser = argparse.ArgumentParser(
    "IKEA Furniture Assembly Environment",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--env_args", type=str, default=None)
parser.add_argument("--record_caption", type=str2bool, default=True)

add_furniture_args(parser)
# add Pick Env and SILO args
add_pick_args(parser)

config, unparsed = parser.parse_known_args()

main(config, unparsed)
