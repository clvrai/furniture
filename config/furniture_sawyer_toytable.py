from util import str2bool


def add_argument(parser):
    parser.set_defaults(max_episode_steps=200)

    parser.add_argument(
        "--site_dist_rew", type=float, default=10, help="rew for dist between sites"
    )
    parser.add_argument(
        "--grip_dist_rew", type=float, default=10, help="rew for dist between sites"
    )
    parser.add_argument(
        "--site_up_rew",
        type=float,
        default=10,
        help="rew for angular dist between site up vecs",
    )
    parser.add_argument(
        "--grip_up_rew",
        type=float,
        default=2,
        help="rew for angular dist between grip and site up vecs",
    )
    parser.add_argument(
        "--connect_rew", type=float, default=25, help="rew for connecting"
    )
    parser.add_argument(
        "--success_rew", type=float, default=5000, help="rew for successful connect"
    )
    parser.add_argument(
        "--pick_rew", type=float, default=10, help="rew for successful pick"
    )
    parser.add_argument(
        "--aligned_rew", type=float, default=10, help="rew for successful pick and rot"
    )
    parser.add_argument(
        "--ctrl_penalty", type=float, default=0.00001, help="penalty for moving"
    )
    parser.add_argument(
        "--grip_penalty", type=float, default=0.001, help="penalty for wrong grip dir"
    )
    parser.add_argument(
        "--grip_z_offset", type=float, default=0.06, help="z offset for grip site"
    )
    parser.add_argument(
        "--topsite_z_offset",
        type=float,
        default=0.07,
        help="z offset for above top site",
    )
    parser.add_argument(
        "--hold_duration", type=int, default=1, help="number of frames to hold leg"
    )
    parser.add_argument(
        "--discretize_grip",
        type=str2bool,
        default=False,
        help="make grip dimension discrete action",
    )
    parser.add_argument(
        "--xy_dist_rew", type=float, default=50, help="rew for successful xy dist"
    )
    parser.add_argument(
        "--z_dist_rew", type=float, default=50, help="rew for successful z"
    )
