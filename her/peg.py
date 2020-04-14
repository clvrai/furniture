import argparse

# peg insertion and SILO arguments
from config.furniture import add_argument as add_furniture_args
from config.peg_insertion import add_argument as add_peg_silo_args
from util import str2bool

from . import main

parser = argparse.ArgumentParser(
    "Peg Insertion Environment",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    conflict_handler="resolve"
)
parser.add_argument("--env_args", type=str, default=None)
parser.add_argument("--record_caption", type=str2bool, default=True)

# add configs like visual_ob
add_furniture_args(parser)
# add SILO arguments
add_peg_silo_args(parser)
config, unparsed = parser.parse_known_args()

main(config, unparsed)
