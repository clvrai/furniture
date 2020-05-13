from env.models import background_names, furniture_ids, furniture_names
from util import str2bool, str2intlist


def add_argument(parser):
    parser.add_argument(
        "--site_dist_rew", type=float, default=40, help="rew for dist between sites"
    )
    parser.add_argument(
        "--site_up_rew",
        type=float,
        default=10,
        help="rew for angular dist between site up vecs",
    )
    parser.add_argument(
        "--connect_rew", type=float, default=25, help="rew for connecting"
    )
    parser.add_argument(
        "--success_rew", type=float, default=100, help="rew for successful connect"
    )
    parser.add_argument(
        "--pick_rew", type=float, default=10, help="rew for successful pick"
    )
    parser.add_argument(
        "--aligned_rew", type=float, default=0.5, help="rew for successful pick and rot"
    )
    parser.add_argument(
        "--ctrl_penalty", type=float, default=0.00001, help="penalty for moving"
    )

    parser.add_argument(
        "--rand_block_range",
        type=float,
        default=0.0,
        help="add U(-r,r) to x,y of block position",
    )
