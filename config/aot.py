from util import str2bool


def add_argument(parser):
    # reset config
    parser.add_argument("--max_failed_reset", type=int, default=1)
    parser.add_argument("--max_reset_episode_steps", type=int, default=99)
    parser.add_argument("--reset_kl_penalty", type=str2bool, default=False)
    parser.add_argument("--kl_penalty_coeff", type=float, default=10)
    parser.add_argument("--zero_qvel", type=str2bool, default=True, help="Zero the robot qvel on forward start")

    # aot config
    parser.add_argument("--status_quo_baseline", type=str2bool, default=False)
    parser.add_argument("--use_aot", type=str2bool, default=False)
    parser.add_argument("--aot_num_episodes", type=int, default=10)
    parser.add_argument("--aot_num_timepairs", type=int, default=50)
    parser.add_argument("--aot_num_batches", type=int, default=300)
    parser.add_argument("--lr_aot", type=float, default=3e-4)
    parser.add_argument("--aot_reg_coeff", type=float, default=0.5)
    parser.add_argument("--aot_weight_decay", type=float, default=0.005)
    parser.add_argument("--aot_hid_size", type=int, default=128)
    parser.add_argument("--aot_rew_coeff", type=float, default=0.1)
    parser.add_argument("--aot_succ_rew", type=float, default=20)
    parser.add_argument("--aot_success_buffer", type=str2bool, default=False)
    parser.add_argument("--aot_success_buffer_size", type=int, default=1e6)
    parser.add_argument("--aot_num_succ_episodes", type=int, default=10)
    parser.add_argument("--aot_num_succ_timepairs", type=int, default=50)

    parser.add_argument("--aot_ensemble", type=int, default=None)
    parser.add_argument("--ensemble_sampler", type=str, default="min", choices=["min", "mean", "max", "median", "meanvar", "softmin", "softmax"])
    parser.add_argument("--var_coeff", type=float, default=1)
    parser.add_argument("--reversible_state_type", type=str, default="obj_pose", choices=["obj_pose", "obj_position"])

    # safety config
    parser.add_argument("--safe_forward", type=str2bool, default=False)
    parser.add_argument("--safety_threshold", type=int, default=float("-inf"))
    parser.add_argument("--num_safety_actions", type=int, default=10)
