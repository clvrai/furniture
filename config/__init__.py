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
        default=env if env else "FurnitureBaxterEnv",
        choices=[
            "FurnitureBaxterEnv",
            "FurnitureSawyerEnv",
            "FurnitureCursorEnv",
            "FurnitureBaxterBlockEnv",
            "FurnitureCursorToyTableEnv",
            "FurnitureSawyerToyTableEnv",
            "FurnitureSaywerPlaceEnv",
            "FurnitureSawyerPickEnv",
        ],
        help="Environment name",
    )
    parser.add_argument("--env_args", type=str, default=None)

    parser.add_argument(
        "--algo", type=str, default="sac", choices=["sac", "ppo", "ddpg", "bc", "gail"]
    )

    args, unparsed = parser.parse_known_args()

    # furniture config
    furniture_config.add_argument(parser)

    env_config = get_env_specific_argument(args.env)
    if env_config:
        env_config.add_argument(parser)

    # training algorithm
    parser.add_argument(
        "--discount_factor", type=float, default=0.99, help="the discount factor"
    )

    # rl
    parser.add_argument("--rl_hid_size", type=int, default=1024)
    parser.add_argument(
        "--rl_activation", type=str, default="relu", choices=["relu", "elu", "tanh"]
    )
    parser.add_argument("--tanh_policy", type=str2bool, default=True)
    parser.add_argument("--rl_deterministic", type=str2bool, default=False)
    if args.algo == "bc":
        parser.set_defaults(rl_deterministic=True)
    parser.add_argument(
        "--lr_actor", type=float, default=3e-4, help="the learning rate of the actor"
    )
    parser.add_argument(
        "--lr_critic", type=float, default=3e-4, help="the learning rate of the critic"
    )
    parser.add_argument(
        "--polyak", type=float, default=0.995, help="the average coefficient"
    )

    # off-policy rl
    parser.add_argument(
        "--buffer_size", type=int, default=int(1e6), help="the size of the buffer"
    )

    # ddpg
    parser.add_argument("--noise_eps", type=float, default=0.2, help="noise eps")
    parser.add_argument("--random_eps", type=float, default=0.3, help="random eps")

    # sac
    parser.add_argument("--reward_scale", type=float, default=1.0, help="reward scale")

    # ppo
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--value_loss_coeff", type=float, default=0.5)
    parser.add_argument("--action_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_loss_coeff", type=float, default=1e-4)
    parser.add_argument("--rollout_length", type=int, default=1000)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # observation normalization
    parser.add_argument("--ob_norm", type=str2bool, default=False)
    parser.add_argument("--max_ob_norm_step", type=int, default=int(1e7))
    parser.add_argument(
        "--clip_obs", type=float, default=200, help="the clip range of observation"
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=5,
        help="the clip range after normalization of observation",
    )

    # training
    parser.add_argument("--is_train", type=str2bool, default=True)
    parser.add_argument(
        "--num_batches",
        type=int,
        default=50,
        help="the times to update the network per epoch",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="the sample batch size"
    )
    parser.add_argument("--max_grad_norm", type=float, default=100)
    parser.add_argument("--max_global_step", type=int, default=int(5e6))
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--init_ckpt_path", type=str, default=None)

    # encoder
    parser.add_argument(
        "--encoder_type", type=str, default="mlp", choices=["mlp", "cnn"]
    )
    parser.add_argument("--encoder_image_size", type=int, default=84)
    parser.add_argument("--encoder_conv_dim", type=int, default=32)
    parser.add_argument("--encoder_mlp_dim", nargs="+", default=[128, 128])
    parser.add_argument("--encoder_kernel_size", nargs="+", default=[3, 3, 3, 3])
    parser.add_argument("--encoder_stride", nargs="+", default=[2, 1, 1, 1])
    parser.add_argument("--encoder_conv_output_dim", type=int, default=50)
    args, unparsed = parser.parse_known_args()
    if args.encoder_type == "cnn":
        parser.set_defaults(screen_width=100, screen_height=100)

    # log
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--evaluate_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=200)
    parser.add_argument("--log_root_dir", type=str, default="log")
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="set it True if you want to use wandb",
    )

    # evaluation
    parser.add_argument("--ckpt_num", type=int, default=None)
    parser.add_argument("--num_eval", type=int, default=10)
    parser.add_argument(
        "--save_rollout",
        type=str2bool,
        default=False,
        help="save rollout information during evaluation",
    )
    parser.add_argument(
        "--num_record_samples",
        type=int,
        default=1,
        help="number of trajectories to collect during eval",
    )

    # misc
    parser.add_argument("--run_prefix", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--debug", type=str2bool, default=False)

    # il
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=10000,
        help="training epoch for behavior cloning",
    )
    parser.add_argument(
        "--lr_bc", type=float, default=1e-3, help="learning rate for bc"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="how much of dataset to leave for validation set"
    )
    parser.add_argument(
        "--sched_lambda", type=float, default=None, help="learning rate scheduler factor"
    )
    parser.add_argument("--demo_path", type=str, default=None, help="path to demos")
    parser.add_argument("--gail_entropy_loss_coeff", type=float, default=1e-3)
    parser.add_argument(
        "--eval_on_train_set",
        type=str2bool,
        default=False,
        help="set on to evaluate on initial positions from training set",
    )

    return parser


def get_env_specific_argument(env):
    f = None

    if env == "FurnitureCursorToyTableEnv":
        import config.furniture_cursor_toytable as f

    elif env == "FurnitureSawyerToyTableEnv":
        import config.furniture_sawyer_toytable as f

    elif env == "FurnitureSawyerPlaceEnv":
        import config.furniture_sawyer_place as f

    elif env == "FurnitureSawyerPickEnv":
        import config.furniture_sawyer_pick as f

    return f


def argparser():
    """
    Directly parses the arguments
    """
    parser = create_parser()
    args, unparsed = parser.parse_known_args()

    return args, unparsed
