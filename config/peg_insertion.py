from util import str2bool


def add_argument(parser):
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=30,
        help="max number of steps for an episode",
    )
    parser.add_argument(
        "--task", type=str, default="insert", choices=["insert", "remove"]
    )
    parser.add_argument("--sparse_rew", type=str2bool, default=False)
    parser.add_argument(
        "--robot_ob",
        type=str2bool,
        default=True,
        help="includes agent state in observation",
    )

    # reward config
    parser.add_argument("--peg_to_point_rew_coeff", type=float, default=5)
    parser.add_argument("--success_rew", type=float, default=1)
    parser.add_argument("--control_penalty_coeff", type=float, default=0.0001)
    parser.add_argument("--goal_pos_threshold", type=float, default=0.2)

    # demo loading
    parser.add_argument('--demo_dir', type=str, default='demos',
                        help='path to demo folder')
    parser.add_argument('--record_demo', type=str2bool, default=False,
                        help='enable demo recording')