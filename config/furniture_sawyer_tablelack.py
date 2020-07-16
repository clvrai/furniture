from util import str2bool


def add_argument(parser):
    parser.set_defaults(max_episode_steps=200)

    parser.add_argument("--pos_threshold", type=float, default=0.01)
    parser.add_argument("--rot_threshold", type=float, default=0.05)

    # environment offsets
    parser.add_argument("--above_leg_z", type=float, default=0.1)

    # reward coefficients
    parser.add_argument("--ctrl_penalty_coef", type=float, default=1)
    parser.add_argument("--eef_leg_rot_dist_coef", type=float, default=1)
    parser.add_argument("--eef_leg_pos_dist_coef", type=float, default=1)
