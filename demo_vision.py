"""
Vision demo for the IKEA furniture assembly environment.
It will show the user the various observation options available
to the environment. The video generation can be RAM heavy, so
decrease --screen_width and --screen_height if it crashes.
"""

import argparse
import pickle

import numpy as np

from env import make_env
from env.models import furniture_names, background_names
import env.image_utils as I
from util import str2bool
from util.video_recorder import VideoRecorder


# available agents
agent_names = ['Baxter', 'Sawyer', 'Cursor']

# available furnitures
furniture_names

# available background scenes
background_names


def argsparser():
    """
    Returns argument parser for furniture assembly environment.
    """
    parser = argparse.ArgumentParser("Demo for IKEA Furniture Assembly Environment")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)

    import config.furniture as furniture_config
    furniture_config.add_argument(parser)

    args = parser.parse_args()
    return args


def main(args):
    """
    Inputs type of agent, observation types and simulates the environment.
    """
    print("The observation tutorial will show you the various observation configurations available.")

    background_name = background_names[1]

    # load demo file for playback
    demo = args.load_demo = input('Input path to demo file, such as demos/Sawyer_7.pkl: ')
    if demo == '':
        demo = args.load_demo = 'demos/Sawyer_7.pkl'

    agent_name, furniture_id = demo.split('/')[-1].split('.')[0].split('_')
    agent_name = agent_name[0].upper() + agent_name[1:]
    furniture_id = int(furniture_id)
    furniture_name = furniture_names[furniture_id]

    # choose robot observation
    print()
    print("Include robot observation?\n")
    try:
        s = input("Put 1 for True or 0 for False: ")
        k = int(s) == 1
    except:
        print("Input is not valid. Use 0 by default.")
        k = False

    args.robot_ob = k

    # choose furniture observation
    print()
    print("Include furniture observation?\n")
    try:
        s = input("Put 1 for True or 0 for False: ")
        k = int(s) == 1
    except:
        print("Input is not valid. Use 0 by default.")
        k = False

    args.object_ob = k

    # choose segmentation
    print()
    print("Use segmentation?\n")
    try:
        s = input("Put 1 for True or 0 for False: ")
        k = int(s) == 1
    except:
        print("Input is not valid. Use 0 by default.")
        k = False

    use_seg = k

    # choose depth
    print()
    print("Use depth map?\n")
    try:
        s = input("Put 1 for True or 0 for False: ")
        k = int(s) == 1
    except:
        print("Input is not valid. Use 0 by default.")
        k = False

    use_depth = k

    # set parameters for the environment (env, furniture_id, background)
    env_name = 'Furniture{}Env'.format(agent_name)
    args.env = env_name
    args.furniture_id = furniture_id
    args.background = background_name

    print()
    print("Creating environment (robot: {}, furniture: {}, background: {})".format(
        env_name, furniture_name, background_name))


    # make environment with rgb, depth map, and segmentation
    args.depth_ob = True
    args.segmentation_ob = True

    # make environment following arguments
    env = make_env(env_name, args)
    ob = env.reset(args.furniture_id, args.background)

    # tell user about environment observation space
    print('-' * 80)
    print('Observation configuration:')
    print(f"Robot ob: {args.robot_ob}, Furniture ob: {args.object_ob}")
    print(f"Depth Map: {use_depth}, Segmentation Map: {use_seg}")
    print()
    print("Observation Space:\n")
    print("The observation space is a dictionary. For furniture (object) observations, it is "+
          "a multiple of 7 because each part has 3 dims for position and 4 dims for quaternion. "+
          "The robot_ob is dependent on the agent, and contains position, velocity, or angles of "+
          "the current robot.\n")
    print(env.observation_space)
    print()
    input("Type anything to record an episode's visual observations")


    # run the trajectory, save the video
    rgb_frames = []
    depth_frames = []
    seg_frames = []

    # load demo from pickle file
    with open(env._load_demo, 'rb') as f:
        demo = pickle.load(f)
        all_qpos = demo['qpos']

    # playback first 100 frames
    for qpos in all_qpos:
        # set furniture part positions
        for i, body in enumerate(env._object_names):
            pos = qpos[body][:3]
            quat = qpos[body][3:]
            env._set_qpos(body, pos, quat)
            env._stop_object(body, gravity=0)
        # set robot positions
        if env._agent_type == 'Sawyer':
            env.sim.data.qpos[env._ref_joint_pos_indexes] = qpos['sawyer_qpos']
            env.sim.data.qpos[env._ref_gripper_joint_pos_indexes] = qpos['l_gripper']
        elif env._agent_type == 'Baxter':
            env.sim.data.qpos[env._ref_joint_pos_indexes] = qpos['baxter_qpos']
            env.sim.data.qpos[env._ref_gripper_right_joint_pos_indexes] = qpos['r_gripper']
            env.sim.data.qpos[env._ref_gripper_left_joint_pos_indexes] = qpos['l_gripper']
        elif env._agent_type == 'Cursor':
            env._set_pos('cursor0', qpos['cursor0'])
            env._set_pos('cursor1', qpos['cursor1'])

        env.sim.forward()
        env._update_unity()

        img, depth = env.render('rgbd_array')
        seg = I.color_segmentation(env.render('segmentation'))
        rgb_frames.append(img)
        depth_frames.append(depth)
        seg_frames.append(seg)

    env.close()

    # concatenate available observation frames together and render video
    wide_frames = []
    L = max(len(rgb_frames), len(rgb_frames), len(seg_frames))
    for l in range(L):
        rgb = rgb_frames[l]
        f = [rgb * 255]
        if use_depth:
            depth = depth_frames[l]
            f.append(depth * 255)
        if use_seg:
            seg = seg_frames[l]
            f.append(seg)
        wide = np.concatenate(f, axis=1)
        wide_frames.append(wide)

    vr = VideoRecorder()
    vr._frames = wide_frames
    vr.save_video('observations.mp4')

if __name__ == '__main__':
    args = argsparser()
    main(args)



