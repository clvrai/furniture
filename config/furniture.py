from util import str2bool, str2intlist
from env.models import furniture_names, furniture_ids, background_names


def add_argument(parser):
    """
    Adds a list of arguments to argparser for the furniture assembly environment.
    """
    # unity
    parser.add_argument('--unity', type=str2bool, default=True,
                        help='Use Unity or MuJoCo for rendering')
    parser.add_argument('--unity_editor', type=str2bool, default=False,
                        help='Use bundle or editor for Unity')
    parser.add_argument('--virtual_display', type=str, default=None,
                        help="Specify virtual display for rendering if you use (e.g. ':0' or ':1')")
    parser.add_argument('--port', type=int, default=1050,
                        help='Port for MuJoCo-Unity plugin')
    parser.add_argument('--background', type=str, default='Industrial',
                        choices=background_names+['Random'],
                        help='Unity background scene, see README for descriptions')
    parser.add_argument('--quality', type=int, default=4,
                        help='Graphics quality of Unity rendering (0~5). -1 represents the best quality')

    # mujoco simulation
    parser.add_argument('--control_type', type=str, default='ik',
                        choices=['ik', 'impedance', 'torque'],
                        help='control type of agent')
    parser.add_argument('--control_freq', type=int, default=10,
                        help='frequency of physic solver steps')
    parser.add_argument('--rescale_actions', type=str2bool, default=True,
                        help='rescale actions to [-1,1] and normalize to the control range')
    parser.add_argument('--move_speed', type=float, default=0.025,
                        help='step size of move actions')
    parser.add_argument('--rotate_speed', type=float, default=11.25,
                        help='step size of rotate actions')

    parser.add_argument('--furniture_id', type=int, default=1,
                        choices=furniture_ids,
                        help='id of furniture model to load')
    parser.add_argument('--furniture_name', type=str, default=None,
                        choices=furniture_names+['Random'],
                        help='name of furniture model to load')
    parser.add_argument('--fix_init', type=str2bool, default=False,
                        help='fixed furniture initialization across episode')
    parser.add_argument('--load_demo', type=str, default=None,
                        help='path to pickle file of demo to load')
    parser.add_argument('--demo_dir', type=str, default='demos',
                        help='path to demo folder')
    parser.add_argument('--record_demo', type=str2bool, default=False,
                        help='enable demo recording')
    parser.add_argument('--preassembled', type=str2intlist, default=[],
                        help='list of weld equality ids to activate at start')
    parser.add_argument('--max_episode_steps', type=int, default=100,
                        help='max number of steps for an episode')

    # observations
    parser.add_argument('--furn_init_randomness', type=float, default=0.01, help='initial randomness of furn')
    parser.add_argument('--init_randomness', type=float, default=0.001, help='initial randomness of scene')
    parser.add_argument('--robot_ob', type=str2bool, default=True,
                        help='includes agent state in observation')
    parser.add_argument('--object_ob', type=str2bool, default=True,
                        help='includes object pose in observation')
    parser.add_argument('--visual_ob', type=str2bool, default=False,
                        help='includes camera image in observation')
    parser.add_argument('--subtask_ob', type=str2bool, default=False,
                        help='includes subtask (furniture part id) in observation')
    parser.add_argument('--depth_ob', type=str2bool, default=False,
                        help='includes depth mapping for camera')
    parser.add_argument('--segmentation_ob', type=str2bool, default=False,
                        help='includes object segmentation for camera')
    parser.add_argument('--screen_width', type=int, default=500,
                        help='width of camera image')
    parser.add_argument('--screen_height', type=int, default=500,
                        help='height of camera image')
    parser.add_argument('--camera_ids', type=str2intlist, default=[0],
                        help='MuJoCo camera id list')
    parser.add_argument('--render', type=str2bool, default=False,
                        help='whether to render camera')

    # vr controller
    parser.add_argument('--wrist_only', type=str2bool, default=False,
                        help='fix rotation to robot wrist')

    # debug
    parser.add_argument('--render_agent', type=str2bool, default=True,
                        help='renders the agent in the scene')
    parser.add_argument('--no_collision', type=str2bool, default=False,
                        help='turns off agent collision')
    parser.add_argument('--assembled', type=str2bool, default=False,
                        help='initialize with assembled furniture')
    


def get_default_config():
    """
    Gets default configurations for the furniture assembly environment.
    """
    import argparse
    from util import str2bool

    parser = argparse.ArgumentParser("Default Configuration for IKEA Furniture Assembly Environment")
    add_argument(parser)

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)

    config = parser.parse_args([])
    return config
