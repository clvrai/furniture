from util import str2bool, str2list


def add_argument(parser):
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=99,
        help="max number of steps for an episode",
    )
    parser.add_argument(
        "--task", type=str, default="forward", choices=["forward", "remove"]
    )
    parser.add_argument("--sparse_rew", type=str2bool, default=False)
    parser.add_argument(
        "--robot_ob",
        type=str2bool,
        default=True,
        help="includes agent state in observation",
    )
    parser.add_argument(
        "--lfd", type=str2bool, default=False, help="use demonstrations to learn task",
    )

    parser.add_argument("--action_noise", type=float, default=None)
    parser.add_argument("--wrist_noise", type=float, default=0.1)
    parser.add_argument("--body_noise", type=float, default=0.1)

    parser.add_argument("--frame_skip", type=int, default=5)

    # reward config
    parser.add_argument("--use_diff_rew", type=str2bool, default=False)
    parser.add_argument("--success_rew", type=float, default=100)
    parser.add_argument("--control_penalty_coeff", type=float, default=0.001)
    parser.add_argument("--goal_pos_threshold", type=float, default=0.03)
    parser.add_argument("--start_pos_threshold", type=float, default=0.03)
    parser.add_argument("--goal_quat_threshold", type=float, default=0.1)
    parser.add_argument("--sparse_remove_rew", type=str2bool, default=False)
    parser.add_argument("--obj_to_point_coeff", type=float, default=10)
    parser.add_argument("--reset_obj_to_point_coeff", type=float, default=10)

    # demo loading
    parser.add_argument(
        "--demo_dir", type=str, default="demos", help="path to demo folder"
    )
    parser.add_argument(
        "--record_demo", type=str2bool, default=False, help="enable demo recording"
    )

    # SILO argparse
    train_arg = parser.add_argument_group("Train")
    env_arg = parser.add_argument_group("Environment")
    push_arg = parser.add_argument_group("RobotPush")

    train_arg.add_argument("--gpu", type=int, default=None)
    train_arg.add_argument("--method", type=str, default="rl", choices=["rl"])
    train_arg.add_argument("--algo", type=str, default="ddpg", choices=["ddpg", "sac"])
    train_arg.add_argument(
        "--meta_algo", type=str, default="dqn", choices=["dqn", "double_dqn"]
    )

    train_arg.add_argument("--resume_on_same_name", type=str2bool, default=True)
    # optimization
    train_arg.add_argument("--lr_start", type=float, default=1e-3)
    train_arg.add_argument("--lr_decay_step", type=int, default=1000000)
    train_arg.add_argument("--lr_decay_rate", type=float, default=1.00)
    train_arg.add_argument("--max_grad_norm", type=float, default=100)
    train_arg.add_argument("--max_global_step", type=int, default=1e7)
    train_arg.add_argument("--is_train", type=str2bool, default=True)
    train_arg.add_argument("--activation", type=str, default="relu")
    train_arg.add_argument("--mini_batch_size", type=int, default=64)

    train_arg.add_argument("--discount_factor", type=float, default=0.99)

    train_arg.add_argument("--adv_norm", type=str2bool, default=False)
    train_arg.add_argument("--gae_lambda", type=float, default=0.95)

    train_arg.add_argument("--ob_norm", type=str2bool, default=True)

    # network
    train_arg.add_argument("--use_batch_norm", type=str2bool, default=False)
    train_arg.add_argument("--kernel_size", type=str2list, default="8,3,3")
    train_arg.add_argument("--stride", type=str2list, default="4,2,2")
    train_arg.add_argument("--conv_dim", type=int, default=32)
    train_arg.add_argument("--hid_size", type=int, default=256)
    train_arg.add_argument("--tcn_dim", type=int, default=32)
    train_arg.add_argument("--goal_estimator_path", type=str, default=None)
    train_arg.add_argument("--goal_threshold", type=float, default=0.03)
    train_arg.add_argument(
        "--goal_criteria", type=str, default="iou", choices=["l2", "iou", "bb_offset"]
    )

    # meta policy
    train_arg.add_argument(
        "--meta_agent",
        type=str,
        default="policy",
        choices=["policy", "random", "sequential"],
    )
    train_arg.add_argument("--meta_window", type=int, default=5)
    train_arg.add_argument(
        "--meta_reward_decay",
        type=float,
        default=0.99,
        help="decaying reward when skipping frames",
    )
    train_arg.add_argument(
        "--meta_reward",
        type=float,
        default=1.0,
        help="reward for meta policy for acheiving a frame",
    )
    train_arg.add_argument(
        "--completion_bonus",
        type=float,
        default=0,
        help="additional reward for task completion",
    )
    train_arg.add_argument(
        "--time_penalty_coeff",
        type=float,
        default=0,
        help="Multiplied with the length of the episode",
    )
    train_arg.add_argument("--binary_q", type=str2bool, default=False)

    # her
    train_arg.add_argument(
        "--num_batches",
        type=int,
        default=50,
        help="the times to update the network per epoch",
    )
    train_arg.add_argument(
        "--replay_strategy", type=str, default="future", help="the HER strategy"
    )
    train_arg.add_argument(
        "--replace_future", type=int, default=0.8, help="ratio to be replace"
    )
    train_arg.add_argument(
        "--buffer_size", type=int, default=int(1e5), help="the size of the buffer"
    )
    train_arg.add_argument(
        "--clip_obs", type=float, default=200, help="the clip range of observation"
    )
    train_arg.add_argument(
        "--clip_range",
        type=float,
        default=5,
        help="the clip range after normalization of observation",
    )
    train_arg.add_argument(
        "--batch_size", type=int, default=256, help="the sample batch size"
    )
    train_arg.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor"
    )
    train_arg.add_argument("--action_l2", type=float, default=0.1, help="l2 reg")
    train_arg.add_argument(
        "--lr_actor", type=float, default=3e-4, help="the learning rate of the actor"
    )
    train_arg.add_argument(
        "--lr_critic", type=float, default=3e-4, help="the learning rate of the critic"
    )
    train_arg.add_argument(
        "--polyak", type=float, default=0.995, help="the average coefficient"
    )

    # ddpg
    train_arg.add_argument("--noise_eps", type=float, default=0.2, help="noise eps")
    train_arg.add_argument("--random_eps", type=float, default=0.3, help="random eps")

    # soft actor-critic
    train_arg.add_argument(
        "--reward_scale", type=float, default=1.0, help="reward scale"
    )

    # dqn
    train_arg.add_argument(
        "--epsilon_decay",
        type=float,
        default=0.005,
        help="decaying epsilon for epsilon-greedy every update",
    )
    train_arg.add_argument(
        "--use_per",
        type=str2bool,
        default=False,
        help="use prioritized experience replay",
    )
    train_arg.add_argument("--per_eps", type=float, default=1e-6)

    # log
    train_arg.add_argument(
        "--wandb_api_key", type=str, default="612ffd4fad3a25888b3995b752de10aca45efe4e"
    )
    train_arg.add_argument("--log_interval", type=int, default=1)
    train_arg.add_argument("--evaluate_interval", type=int, default=100)
    train_arg.add_argument("--ckpt_interval", type=int, default=100)
    train_arg.add_argument("--prefix", type=str, default="test")
    train_arg.add_argument("--log_dir", type=str, default="logs")
    train_arg.add_argument("--data_dir", type=str, default="remove_demos")
    train_arg.add_argument("--load_path", type=str, default="")
    train_arg.add_argument("--ckpt_num", type=int, default=None)
    train_arg.add_argument("--num_eval", type=int, default=10)
    train_arg.add_argument("--record", type=str2bool, default=True)
    train_arg.add_argument("--caption", type=str2bool, default=True)
    train_arg.add_argument("--debug", type=str2bool, default=False)
    train_arg.add_argument("--seed", type=int, default=123, help="random seed")

    # additional inputs
    env_arg.add_argument(
        "--env", type=str, default="PusherEnv", choices=["PusherEnv"],
    )
    env_arg.add_argument(
        "--train_mode",
        type=int,
        default=1,
        help="0: normal training" "1: training single step",
    )
    # env setup
    push_arg.add_argument("--fixed_episode", type=str2bool, default=False)
    push_arg.add_argument(
        "--control_mode", type=str, default="task_space", choices=["task_space"]
    )
    push_arg.add_argument("--simulated", type=str2bool, default=False)
    push_arg.add_argument("--terminate_on_collision", type=str2bool, default=True)
    push_arg.add_argument("--collision_penalty", type=float, default=0)
    push_arg.add_argument("--obstacle_penalty", type=float, default=0)
    env_arg.add_argument(
        "--goal_type",
        type=str,
        default="state_obj",
        choices=[
            "state_obj_robot",
            "state_obj_rot_robot",
            "state_obj",
            "state_obj_rot",
            "tcn",
            "detector2d",
            "detector3d",
            "detector_box",
        ],
    )
    train_arg.add_argument("--max_ob_norm_step", type=int, default=10000000)
    env_arg.add_argument("--num_seeds", type=int, default=9)
    env_arg.add_argument(
        "--action_repeat", type=int, default=5, help="action repeat for the robots"
    )
