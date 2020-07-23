import argparse

import config.furniture as furniture_config
from util import str2bool


def create_parser(env=None):
    """
    Creates the argparser.  Use this to add additional arguments
    to the parser later.
    """
    parser = argparse.ArgumentParser(
        "IKEA Furniture Assembly Environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # environment
    parser.add_argument(
        "--env",
        type=str,
        default=env if env is not None else "furniture-baxter-flip-v0",
        help="Environment name",
    )

    args, unparsed = parser.parse_known_args()

    # furniture config
    furniture_config.add_argument(parser)

    env_config = get_env_specific_argument(args.env)
    if env_config:
        env_config.add_argument(parser)

    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--debug", type=str2bool, default=False)

    return parser


def get_env_specific_argument(env):
    f = None
    if env == "FurnitureCursorToyTableEnv":
        import config.furniture_cursor_toytable as f

    elif env == "FurnitureSawyerGenEnv":
        import config.furniture_sawyer_gen as f
        
    elif env == "FurnitureSawyerToyTableEnv":
        import config.furniture_sawyer_toytable as f

    elif env == "FurnitureSawyerPlaceEnv":
        import config.furniture_sawyer_place as f

    elif env == "FurnitureSawyerPickEnv":
        import config.furniture_sawyer_pick as f

    elif env == "furniture-sawyer-tablelack-v0":
        import config.furniture_sawyer_tablelack as f

    return f


def argparser():
    """
    Directly parses the arguments
    """
    parser = create_parser()
    args, unparsed = parser.parse_known_args()

    return args, unparsed
