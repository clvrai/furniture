import argparse

from util import str2bool, str2list

def create_parser(env=None):
    """
    Creates the argparser.  Use this to add additional arguments
    to the parser later.
    """
    parser = argparse.ArgumentParser(
        'IKEA Furniture Assembly Environment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # environment
    parser.add_argument('--env', type=str, default='FurnitureBaxterEnv',
                        choices=['FurnitureBaxterEnv',
                                 'FurnitureSawyerEnv',
                                 'FurnitureCursorEnv',
                                 'FurnitureBaxterBlockEnv',
                                 'FurnitureCursorToyTableEnv',
                                 'FurnitureSawyerToyTableEnv'],
                        help='Environment name')
    parser.add_argument('--env_args', type=str, default=None)

    args, unparsed = parser.parse_known_args()
    # furniture
    import config.furniture as furniture_config
    furniture_config.add_argument(parser)
    if 'FurnitureCursorToyTableEnv' in [env, args.env]:
        import config.furniture_cursor_toytable as f
        f.add_argument(parser)
    elif 'FurnitureSawyerToyTableEnv' in [env, args.env]:
        import config.furniture_sawyer_toytable as f
        f.add_argument(parser)

    # training algorithm
    parser.add_argument('--algo', type=str, default='sac',
                        choices=['sac', 'ppo', 'ddpg', 'bc', 'gail'])
    parser.add_argument('--policy', type=str, default='mlp',
                        choices=['mlp', 'manual'])

    # vanilla rl
    parser.add_argument('--rl_hid_size', type=int, default=128)
    parser.add_argument('--rl_activation', type=str, default='relu',
                        choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--tanh_policy', type=str2bool, default=True)

    # for ddpg
    parser.add_argument('--noise_eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random_eps', type=float, default=0.3, help='random eps')


    # observation normalization
    parser.add_argument('--ob_norm', type=str2bool, default=True)
    parser.add_argument('--max_ob_norm_step', type=int, default=int(1e7))
    parser.add_argument('--clip_obs', type=float, default=200, help='the clip range of observation')
    parser.add_argument('--clip_range', type=float, default=5, help='the clip range after normalization of observation')

    # off-policy rl
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='the learning rate of the actor')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.995, help='the average coefficient')

    # training
    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--num_batches', type=int, default=50,
                        help='the times to update the network per epoch')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='the sample batch size')
    parser.add_argument('--max_grad_norm', type=float, default=100)
    parser.add_argument('--max_global_step', type=int, default=int(5e6))
    parser.add_argument('--gpu', type=int, default=None)

    # sac
    parser.add_argument('--reward_scale', type=float, default=1.0, help='reward scale')

    # ppo
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--value_loss_coeff', type=float, default=0.5)
    parser.add_argument('--action_loss_coeff', type=float, default=1.0)
    parser.add_argument('--entropy_loss_coeff', type=float, default=1e-4)
    parser.add_argument('--rollout_length', type=int, default=1000)
    parser.add_argument('--gae_lambda', type=float, default=0.95)

    # log
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--evaluate_interval', type=int, default=10)
    parser.add_argument('--ckpt_interval', type=int, default=200)
    parser.add_argument('--log_root_dir', type=str, default='log')
    parser.add_argument('--wandb', type=str2bool, default=False,
                        help='set it True if you want to use wandb')

    # evaluation
    parser.add_argument('--ckpt_num', type=int, default=None)
    parser.add_argument('--num_eval', type=int, default=10)
    parser.add_argument('--save_rollout', type=str2bool, default=False,
                        help='save rollout information during evaluation')
    parser.add_argument('--record', type=str2bool, default=True,
                        help='enable video recording')
    parser.add_argument('--record_caption', type=str2bool, default=True)
    parser.add_argument('--num_record_samples', type=int, default=1,
                        help='number of trajectories to collect during eval')
    parser.add_argument('--save_qpos', type=str2bool, default=False,
                        help='save entire qpos history of success rollouts to file (for idle primitive training)')
    parser.add_argument('--save_success_qpos', type=str2bool, default=False,
                        help='save later segment of success rollouts to file (for moving and placing primitie trainings)')

    # misc
    parser.add_argument('--prefix', type=str, default='test')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--debug', type=str2bool, default=False)
    return parser


def argparser():
    """
    Directly parses the arguments
    """
    parser = create_parser()
    args, unparsed = parser.parse_known_args()
    args.env_args_str = args.env_args

    return args, unparsed
