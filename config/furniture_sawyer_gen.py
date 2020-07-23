from util import str2bool


def add_argument(parser):
    parser.set_defaults(max_episode_steps=200)

    parser.add_argument(
        "--start_count",
        type=int,
        default=None,
        help="specific demo count for overwriting automatic demo count",
    )