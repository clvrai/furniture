from util import str2bool


def add_argument(parser):
    parser.set_defaults(max_episode_steps=200)

    parser.add_argument("--pos_threshold", type=float, default=0.015)
    parser.add_argument("--rot_threshold", type=float, default=0.05)
    parser.add_argument("--discrete_grip", type=str2bool, default=True)

    # environment offsets
    parser.add_argument("--above_leg_z", type=float, default=0.05)

    # reward coefficients
    parser.add_argument("--diff_rew", type=str2bool, default=False)
    parser.add_argument("--ctrl_penalty_coef", type=float, default=0.1)
    parser.add_argument("--touch_coef", type=float, default=0.3)
    parser.add_argument("--gripper_penalty_coef", type=float, default=0.1)
    parser.add_argument("--rot_dist_coef", type=float, default=0.05)
    parser.add_argument("--pos_dist_coef", type=float, default=1)
    parser.add_argument("--align_rot_dist_coef", type=float, default=0.3)
    parser.add_argument("--fine_align_rot_dist_coef", type=float, default=0.6)
    parser.add_argument("--fine_pos_dist_coef", type=float, default=5)

    parser.add_argument("--phase_bonus", type=float, default=5)
