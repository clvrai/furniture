from ..util import str2bool


def add_argument(parser):
    parser.set_defaults(max_episode_steps=150)
    parser.set_defaults(control_type="impedance")
    parser.set_defaults(furniture_name="table_lack_0825")
    parser.set_defaults(unity=False)

    parser.set_defaults(auto_align=False)
    parser.set_defaults(alignment_pos_dist=0.02)
    parser.set_defaults(alignment_rot_dist_up=0.99)
    parser.set_defaults(alignment_rot_dist_forward=0.99)
    parser.set_defaults(alignment_project_dist=0.)

    # reward coefficients
    ## common rewards
    parser.add_argument("--diff_rew", type=str2bool, default=True)
    parser.add_argument("--phase_bonus", type=float, default=2000)
    parser.add_argument("--eef_forward_dist_coef", type=float, default=2)
    parser.add_argument("--eef_up_dist_coef", type=float, default=4)
    parser.add_argument("--eef_rot_threshold", type=float, default=0.95)
    parser.add_argument("--gripper_penalty_coef", type=float, default=5)
    parser.add_argument("--move_other_part_penalty_coef", type=float, default=50)

    parser.add_argument("--early_termination", type=str2bool, default=True)

    ## init_eef
    parser.add_argument("--init_eef_pos_dist_coef", type=float, default=100)

    ## move_eef_above_leg
    parser.add_argument("--move_eef_pos_dist_coef", type=float, default=100)

    ## lower_eef_to_leg
    parser.add_argument("--lower_eef_pos_dist_coef", type=float, default=1000)

    ## grasp_leg
    parser.add_argument("--grasp_dist_coef", type=float, default=200)

    ## lift_leg
    parser.add_argument("--lift_z_dist_coef", type=float, default=500)
    parser.add_argument("--lift_xy_dist_coef", type=float, default=250)

    ## align_leg
    parser.add_argument("--align_pos_dist_coef", type=float, default=100)
    parser.add_argument("--align_rot_dist_coef", type=float, default=50)
    parser.add_argument("--align_pos_threshold", type=float, default=0.2)
    parser.add_argument("--align_rot_threshold", type=float, default=0.85)

    ## move_leg
    parser.add_argument("--move_pos_dist_coef", type=float, default=300)
    parser.add_argument("--move_rot_dist_coef", type=float, default=50)
    parser.add_argument("--move_pos_threshold", type=float, default=0.06)
    parser.add_argument("--move_rot_threshold", type=float, default=0.85)

    ## move_leg_fine
    parser.add_argument("--move_fine_pos_dist_coef", type=float, default=500)
    parser.add_argument("--move_fine_rot_dist_coef", type=float, default=200)
    parser.add_argument("--aligned_bonus_coef", type=float, default=10)
