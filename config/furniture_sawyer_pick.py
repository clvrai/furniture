from util import str2bool


def add_argument(parser):
    parser.add_argument(
        "--success_rew", type=float, default=5000, help="rew for successful connect"
    )
    parser.add_argument(
        "--pick_rew", type=float, default=10, help="rew for successful pick"
    )
    parser.add_argument(
        "--ctrl_penalty", type=float, default=0.00001, help="penalty for moving"
    )
    parser.add_argument(
        "--hold_duration", type=int, default=1, help="number of frames to hold leg"
    )
    parser.add_argument(
        "--discretize_grip",
        type=str2bool,
        default=False,
        help="make grip dimension discrete action",
    )
    parser.add_argument(
        "--rand_start_range",
        type=float,
        default=0.0,
        help="add U(-r,r) to each dim of starting state",
    )

    parser.add_argument(
        "--rand_block_range",
        type=float,
        default=0.0,
        help="add U(-r,r) to x,y of block position",
    )
