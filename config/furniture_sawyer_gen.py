from util import str2bool


def add_argument(parser):
    parser.set_defaults(max_episode_steps=200)
    parser.set_defaults(render=False)
    parser.set_defaults(record_vid=False)
    parser.set_defaults(unity=False)

    parser.add_argument(
        "--start_count",
        type=int,
        default=None,
        help="specific demo prefix to use instead of default demo prefix",
    )

    parser.add_argument(
        "--n_demos",
        type=int,
        default=20,
        help="number of demos to generate",
    )
