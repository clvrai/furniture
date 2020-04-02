""" Define base environment class FurnitureEnv. """

import logging
import os
import pickle
import time
from collections import OrderedDict

import numpy as np
from pyquaternion import Quaternion
from scipy.interpolate import interp1d

import env.transform_utils as T
from env.action_spec import ActionSpec
from env.base import EnvMeta
from env.image_utils import color_segmentation
from env.mjcf_utils import xml_path_completion
from env.models import (
    background_names,
    furniture_name2id,
    furniture_names,
    furniture_xmls,
)
from env.models.grippers import gripper_factory
from env.models.objects import MujocoXMLObject
from env.unity_interface import UnityInterface
from util.demo_recorder import DemoRecorder
from util.logger import logger
from util.video_recorder import VideoRecorder

try:
    import mujoco_py
except ImportError as e:
    raise Exception("{}. (need to install mujoco_py)".format(e))

np.set_printoptions(suppress=True)


class FurnitureEnv(metaclass=EnvMeta):
    """
    Base class for IKEA furniture assembly environment.
    """

    name = "furniture"

    def __init__(self, config):
        """
        Initializes class with the configuration.
        """
        self._config = config
        # default env config
        self._env_config = {
            "max_episode_steps": config.max_episode_steps,
            "success_reward": 100,
            "ctrl_reward": 1e-3,
            "init_randomness": config.init_randomness,
            "furn_init_randomness": config.furn_init_randomness,
            "unstable_penalty": 100,
            "boundary": 1.5,  # XYZ cube boundary
            "pos_dist": 0.1,
            "rot_dist_up": 0.9,
            "rot_dist_forward": 0.9,
            "project_dist": 0.3,
        }

        self._debug = config.debug
        if logger.getEffectiveLevel() != logging.CRITICAL:
            logger.setLevel(logging.INFO)
        if self._debug:
            logger.setLevel(logging.DEBUG)
        self._rng = np.random.RandomState(config.seed)

        if config.render and not config.unity:
            self._render_mode = "human"
        else:
            self._render_mode = "no"  # ['no', 'human', 'rgb_array']

        self._screen_width = config.screen_width
        self._screen_height = config.screen_height

        self._agent_type = config.agent_type  # ['baxter', 'sawyer', 'cursor']
        self._control_type = config.control_type  # ['ik', 'impedance']
        self._control_freq = config.control_freq  # reduce freq -> longer timestep
        self._rescale_actions = config.rescale_actions

        self._robot_ob = config.robot_ob
        self._object_ob = config.object_ob
        self._visual_ob = config.visual_ob
        self._subtask_ob = config.subtask_ob
        self._segmentation_ob = config.segmentation_ob
        self._depth_ob = config.depth_ob
        self._camera_ids = config.camera_ids
        self._camera_name = "frontview"
        self._is_render = False

        self._furniture_id = None
        self._background = None
        self._pos_init = None
        self._quat_init = None

        self._action_on = False
        self._load_demo = config.load_demo
        self._init_qpos = None
        if self._load_demo:
            with open(self._load_demo, "rb") as f:
                demo = pickle.load(f)
                self._init_qpos = demo["qpos"][-1]

        self._record_demo = config.record_demo
        if self._record_demo:
            self._demo = DemoRecorder(config.demo_dir)

        self._num_connect_steps = 0
        self._gravity_compensation = 0

        self._move_speed = config.move_speed
        self._rotate_speed = config.rotate_speed

        self._preassembled = config.preassembled

        if self._agent_type != "Cursor" and self._control_type == "ik":
            self._min_gripper_pos = np.array([-1.5, -1.5, 0.0])
            self._max_gripper_pos = np.array([1.5, 1.5, 1.5])
            self._action_repeat = 5

        self._viewer = None
        self._unity = None
        if config.unity:
            self._unity = UnityInterface(
                config.port, config.unity_editor, config.virtual_display
            )
            # set to the best quality
            self._unity.set_quality(config.quality)

    @property
    def observation_space(self):
        """
        Returns dict where keys are ob names and values are dimensions.
        """
        ob_space = OrderedDict()
        num_cam = len(self._camera_ids)
        if self._visual_ob:
            ob_space["camera_ob"] = [
                num_cam,
                self._screen_height,
                self._screen_width,
                3,
            ]

        if self._object_ob:
            # can be changed to the desired number depending on the task
            ob_space["object_ob"] = [(3 + 4) * 2]

        if self._subtask_ob:
            ob_space["subtask_ob"] = 2

        return ob_space

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with out grippers).
        """
        raise NotImplementedError

    @property
    def max_episode_steps(self):
        """
        Returns maximum number of steps in an episode.
        """
        return self._env_config["max_episode_steps"]

    @property
    def action_size(self):
        """
        Returns size of the action space.
        """
        return self.action_space.size

    @property
    def action_space(self):
        """
        Returns ActionSpec of action space, see
        action_spec.py for more documentation.
        """
        return ActionSpec(self.dof)

    def reset(self, furniture_id=None, background=None):
        """
        Takes in a furniture_id, and background string.
        Resets the environment, viewer, and internal variables.
        Returns the initial observation.
        """
        self._reset(furniture_id=furniture_id, background=background)
        # reset mujoco viewer
        if self._render_mode == "human" and not self._unity:
            self._viewer = self._get_viewer()
        self._after_reset()

        ob = self._get_obs()
        if self._record_demo:
            self._demo.add(ob=ob)

        return ob

    def _init_random(self, size, name):
        """
        Returns initial random distribution.
        """
        if name == "furniture":
            r = self._env_config["furn_init_randomness"]
        else:
            r = self._env_config["init_randomness"]
        return self._rng.uniform(low=-r, high=r, size=size)

    def _after_reset(self):
        """
        Reset timekeeping and internal state for episode.
        """
        self._episode_reward = 0
        self._episode_length = 0
        self._episode_time = time.time()

        self._terminal = False
        self._success = False
        self._fail = False
        self._unity_updated = False

    def step(self, action):
        """
        Computes the next environment state given @action.
        Stores env state for demonstration if needed.
        Returns observation dict, reward float, done bool, and info dict.
        """
        self._before_step()
        if isinstance(action, list):
            action = {key: val for ac_i in action for key, val in ac_i.items()}
        if isinstance(action, dict):
            action = np.concatenate(
                [action[key] for key in self.action_space.shape.keys()]
            )
        ob, reward, done, info = self._step(action)
        done, info, penalty = self._after_step(reward, done, info)
        reward += penalty
        if self._record_demo:
            self._store_qpos()
            self._demo.add(ob=ob, action=action, reward=reward)
        return ob, reward, done, info

    def _before_step(self):
        """
        Called before step
        """
        self._unity_updated = False

    def _update_unity(self):
        """
        Updates unity rendering with qpos. Call this after you change qpos
        """
        self._unity.set_qpos(self.sim.data.qpos)
        if self._agent_type == "Cursor":
            for cursor_i in range(2):
                cursor_name = "cursor%d" % cursor_i
                cursor_pos = self._get_pos(cursor_name)
                self._unity.set_geom_pos(cursor_name, cursor_pos)

    def _step(self, a):
        """
        Internal step function. Moves agent, updates unity, and then
        returns ob, reward, done, info tuple
        """
        if a is None:
            a = np.zeros(self.dof)

        if self._agent_type == "Cursor":
            self._step_discrete(a.copy())
            self._do_simulation(None)

        elif self._control_type == "ik":
            self._step_continuous(a.copy())

        elif self._control_type == "torque":
            self._step_continuous(a.copy())

        elif self._control_type == "impedance":
            a = self._setup_action(a.copy())
            self._do_simulation(a)

        else:
            raise ValueError

        if self._connected_body1 is not None:
            self.sim.forward()
            self._move_objects_target(
                self._connected_body1,
                self._connected_body1_pos,
                self._connected_body1_quat,
                self._gravity_compensation,
            )
            self._connected_body1 = None
            self.sim.forward()
            self.sim.step()

        ob = self._get_obs()
        done = False
        if (
            self._num_connected == len(self._object_names) - 1
            and len(self._object_names) > 1
        ):
            self._success = True
            done = True

        reward = 0
        info = {}
        return ob, reward, done, info

    def _after_step(self, reward, terminal, info):
        """
        Called after _step, adds additional information and calculate penalty
        """
        step_log = dict(info)
        self._terminal = terminal
        penalty = 0

        if reward is not None:
            self._episode_reward += reward
            self._episode_length += 1

        if self._episode_length == self._env_config["max_episode_steps"] or self._fail:
            self._terminal = True
            if self._fail:
                self._fail = False
                penalty = -self._env_config["unstable_penalty"]

        if self._terminal:
            total_time = time.time() - self._episode_time
            step_log["episode_success"] = int(self._success)
            step_log["episode_reward"] = self._episode_reward + penalty
            step_log["episode_length"] = self._episode_length
            step_log["episode_time"] = total_time
            step_log["episode_unstable"] = penalty

        return self._terminal, step_log, penalty

    def _compute_reward(self):
        """
        Computes the reward at the current step
        """
        reward = self._env_config["success_reward"] * self._num_connected
        # do not terminate
        done = False
        info = {}
        return reward, done, info

    def _ctrl_reward(self, a):
        """
        Control penalty to discourage erratic motions
        """
        if a is None or self._agent_type == "Cursor":
            return 0
        ctrl_reward = -self._env_config["ctrl_reward"] * np.square(a).sum()
        return ctrl_reward

    def _set_camera_position(self, cam_id, cam_pos):
        """
        Sets cam_id camera position to cam_pos
        """
        self.sim.model.cam_pos[cam_id] = cam_pos.copy()

    def _set_camera_rotation(self, cam_id, target_pos):
        """
        Rotates camera to look at target_pos
        """
        cam_pos = self.sim.model.cam_pos[cam_id]
        forward = target_pos - cam_pos
        up = [
            forward[0],
            forward[1],
            (forward[0] ** 2 + forward[1] ** 2) / (-forward[2]),
        ]
        if forward[0] == 0 and forward[1] == 0:
            up = [0, 1, 0]
        q = T.lookat_to_quat(-forward, up)
        self.sim.model.cam_quat[cam_id] = T.convert_quat(q, to="wxyz")

    def _set_camera_pose(self, cam_id, pose):
        """
        Sets unity camera to pose
        """
        self._unity.set_camera_pose(cam_id, pose)

    def _render_callback(self):
        """
        Callback for rendering
        """
        self.sim.forward()

    def render(self, mode="human"):
        """
        Renders the environment. If mode is rgb_array, we render the pixels.
        The pixels can be rgb, depth map, segmentation map
        If the mode is human, we render to the MuJoCo viewer, or for unity,
        do nothing since rendering is handled by Unity.
        """
        self._render_callback()

        # update unity
        if not self._unity_updated and self._unity:
            self._update_unity()
            self._unity_updated = True

        if mode == "rgb_array":
            if self._unity:
                img, _ = self._unity.get_images(self._camera_ids)
            else:
                img = self.sim.render(
                    camera_name=self._camera_name,
                    width=self._screen_width,
                    height=self._screen_height,
                    depth=False,
                )
                img = np.expand_dims(img, axis=0)
            assert len(img.shape) == 4
            img = img[:, ::-1, :, :] / 255.0
            return img

        elif mode == "rgbd_array":
            depth = None
            if self._unity:
                img, depth = self._unity.get_images(self._camera_ids, self._depth_ob)
            else:
                camera_obs = self.sim.render(
                    camera_name=self._camera_name,
                    width=self._screen_width,
                    height=self._screen_height,
                    depth=self._depth_ob,
                )
                if self._depth_ob:
                    img, depth = camera_obs
                else:
                    img = camera_obs
            if len(img.shape) == 4:
                img = img[:, ::-1, :, :] / 255.0
            elif len(img.shape) == 3:
                img = img[::-1, :, :] / 255.0

            if depth is not None:
                # depth map is 0 to 1, with 1 being furthest
                # infinite depth is 0, so set to 1
                black_pixels = np.all(depth == [0, 0, 0], axis=-1)
                depth[black_pixels] = [255] * 3
                if len(depth.shape) == 4:
                    depth = depth[:, ::-1, :, :] / 255.0
                elif len(depth.shape) == 3:
                    depth = depth[::-1, :, :] / 255.0

            return img, depth

        elif mode == "segmentation" and self._unity:
            img = self._unity.get_segmentations(self._camera_ids)
            return img

        elif mode == "human" and not self._unity:
            self._get_viewer().render()

        return None

    def _destroy_viewer(self):
        """
        Destroys the current viewer if there is one
        """
        if self._viewer is not None:
            import glfw

            glfw.destroy_window(self._viewer.window)
            self._viewer = None

    def _viewer_reset(self):
        """
        Resets the viewer
        """
        pass

    def _get_viewer(self):
        """
        Returns the viewer instance, or instantiates a new one
        """
        if self._viewer is None:
            self._viewer = mujoco_py.MjViewer(self.sim)
            self._viewer.cam.fixedcamid = self._camera_ids[0]
            self._viewer.cam.type = mujoco_py.generated.const.CAMERA_FIXED
            self._viewer_reset()
        return self._viewer

    def close(self):
        """
        Cleans up the environment
        """
        if self._unity:
            self._unity.disconnect_to_unity()
        self._destroy_viewer()

    def __delete__(self):
        """
        Called to destroy environment
        """
        if self._unity:
            self._unity.disconnect_to_unity()

    def __del__(self):
        """
        Called to destroy environment
        """
        if self._unity:
            self._unity.disconnect_to_unity()

    def _move_cursor(self, cursor_i, move_offset):
        """
        Moves cursor by move_offset amount, takes into account the
        boundary
        """
        cursor_name = "cursor%d" % cursor_i
        cursor_pos = self._get_pos(cursor_name)
        cursor_pos = cursor_pos + move_offset
        boundary = self._env_config["boundary"]
        if (np.abs(cursor_pos) < boundary).all() and cursor_pos[
            2
        ] >= self._move_speed * 0.45:
            self._set_pos(cursor_name, cursor_pos)
            return True
        return False

    def _move_rotate_object(self, obj, move_offset, rotate_offset):
        """
        Used by cursor to move and rotate selected objects
        """
        qpos_base = self._get_qpos(obj)
        target_quat = T.euler_to_quat(rotate_offset, qpos_base[3:])

        part_idx = self._object_name2id[obj]
        old_pos_rot = {}
        for i, obj_name in enumerate(self._object_names):
            if self._find_group(i) == self._find_group(part_idx):
                old_pos_rot[obj_name] = self._get_qpos(obj_name)
                new_pos, new_rot = T.transform_to_target_quat(
                    qpos_base, self._get_qpos(obj_name), target_quat
                )
                new_pos = new_pos + move_offset
                self._set_qpos(obj_name, new_pos, new_rot)

        if self._is_inside(obj):
            return True

        for obj_name, pos_rot in old_pos_rot.items():
            self._set_qpos(obj_name, pos_rot[:3], pos_rot[3:])
        return False

    def _get_bounding_box(self, obj_name):
        """
        Gets the bounding box of the object
        """
        body_ids = []
        part_idx = self._object_name2id[obj_name]
        for i, body_name in enumerate(self._object_names):
            if self._find_group(i) == self._find_group(part_idx):
                body_id = self.sim.model.body_name2id(body_name)
                body_ids.append(body_id)

        body_id = self.sim.model.body_name2id(obj_name)
        min_pos = np.array([0, 0, 0])
        max_pos = np.array([0, 0, 0])
        for i, site in enumerate(self.sim.model.site_names):
            if self.sim.model.site_bodyid[i] in body_ids:
                pos = self._get_pos(site)
                min_pos = np.minimum(min_pos, pos)
                max_pos = np.maximum(max_pos, pos)

        return min_pos, max_pos

    def _is_inside(self, obj_name):
        """
        Determines if object is inside the boundary
        """
        self.sim.forward()
        self.sim.step()
        min_pos, max_pos = self._get_bounding_box(obj_name)
        b = self._env_config["boundary"]
        if (min_pos < np.array([-b, -b, -0.05])).any() or (
            max_pos > np.array([b, b, b])
        ).any():
            return False
        return True

    def _select_object(self, cursor_i):
        """
        Selects an object within cursor_i
        """
        for obj_name in self._object_names:
            is_selected = False
            obj_group = self._find_group(obj_name)
            for selected_obj in self._cursor_selected:
                if selected_obj and obj_group == self._find_group(selected_obj):
                    is_selected = True

            if not is_selected and self.on_collision("cursor%d" % cursor_i, obj_name):
                return obj_name
        return None

    def _step_discrete(self, a):
        """
        Takes a step for the cursor agent
        """
        assert len(a) == 15
        actions = [a[:7], a[7:]]

        for cursor_i in range(2):
            # move
            move_offset = actions[cursor_i][0:3] * self._move_speed
            # rotate
            rotate_offset = actions[cursor_i][3:6] * self._rotate_speed
            # select
            select = actions[cursor_i][6] > 0

            if not select:
                self._cursor_selected[cursor_i] = None

            success = self._move_cursor(cursor_i, move_offset)
            if not success:
                logger.debug("could not move cursor")
                continue
            if self._cursor_selected[cursor_i] is not None:
                success = self._move_rotate_object(
                    self._cursor_selected[cursor_i], move_offset, rotate_offset
                )
                if not success:
                    logger.debug("could not move cursor due to object out of boundary")
                    # reset cursor to original position
                    self._move_cursor(cursor_i, -move_offset)
                    continue

            if select:
                if self._cursor_selected[cursor_i] is None:
                    self._cursor_selected[cursor_i] = self._select_object(cursor_i)

        connect = a[14]
        if connect > 0 and self._cursor_selected[0] and self._cursor_selected[1]:
            logger.debug(
                "try connect ({} and {})".format(
                    self._cursor_selected[0], self._cursor_selected[1]
                )
            )
            self._try_connect(self._cursor_selected[0], self._cursor_selected[1])
        elif self._connect_step > 0:
            self._connect_step = 0

    def _connect(self, site1_id, site2_id):
        """
        Connects two sites together with weld constraint.
        Makes the two objects are within boundaries
        """
        self._connected_sites.add(site1_id)
        self._connected_sites.add(site2_id)
        self._site1_id = site1_id
        self._site2_id = site2_id
        site1 = self.sim.model.site_names[site1_id]
        site2 = self.sim.model.site_names[site2_id]

        logger.debug("**** connect {} and {}".format(site1, site2))

        body1_id = self.sim.model.site_bodyid[site1_id]
        body2_id = self.sim.model.site_bodyid[site2_id]
        body1 = self.sim.model.body_id2name(body1_id)
        body2 = self.sim.model.body_id2name(body2_id)

        # remove collision
        group1 = self._find_group(body1)
        group2 = self._find_group(body2)
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            if body_name in self._object_names:
                group = self._find_group(body_name)
                if group in [group1, group2]:
                    if self.sim.model.geom_contype[geom_id] != 0:
                        self.sim.model.geom_contype[geom_id] = (
                            (1 << 30) - 1 - (1 << (group1 + 1))
                        )
                        self.sim.model.geom_conaffinity[geom_id] = 1 << (group1 + 1)

        # align site
        self._align_connectors(site1, site2, gravity=self._gravity_compensation)

        # move furniture to collision-safe position
        if self._agent_type == "Cursor":
            self._stop_objects()
        self.sim.forward()
        self.sim.step()
        min_pos1, max_pos1 = self._get_bounding_box(body1)
        min_pos2, max_pos2 = self._get_bounding_box(body2)
        min_pos = np.minimum(min_pos1, min_pos2)
        if min_pos[2] < 0:
            offset = [0, 0, -min_pos[2]]
            self._move_rotate_object(body1, offset, [0, 0, 0])
            self._move_rotate_object(body2, offset, [0, 0, 0])

        # activate weld
        if self._agent_type == "Cursor":
            self._stop_objects()
        self.sim.forward()
        self.sim.step()

        self._activate_weld(body1, body2)

        self._num_connected += 1
        # release cursor
        if self._agent_type == "Cursor":
            self._cursor_selected[1] = None

        self._connected_body1 = body1
        self._connected_body1_pos = self._get_qpos(body1)[:3]
        self._connected_body1_quat = self._get_qpos(body1)[3:]

        # set next subtask
        self._get_next_subtask()

    def _try_connect(self, part1=None, part2=None):
        """
        Attempts to connect 2 parts. If they are correctly aligned,
        then we interpolate the 2 parts towards the target position and orientation
        for smoother visual connection.
        """
        if part1 is not None:
            body1_ids = [
                self.sim.model.body_name2id(obj_name)
                for obj_name in self._object_names
                if self._find_group(obj_name) == self._find_group(part1)
            ]
        else:
            body1_ids = [
                self.sim.model.body_name2id(obj_name) for obj_name in self._object_names
            ]

        if part2 is not None:
            body2_ids = [
                self.sim.model.body_name2id(obj_name)
                for obj_name in self._object_names
                if self._find_group(obj_name) == self._find_group(part2)
            ]
        else:
            body2_ids = [
                self.sim.model.body_name2id(obj_name) for obj_name in self._object_names
            ]

        sites1 = []
        sites2 = []
        for j, site in enumerate(self.sim.model.site_names):
            if "conn_site" in site:
                if self.sim.model.site_bodyid[j] in body1_ids:
                    sites1.append((j, site))
                if self.sim.model.site_bodyid[j] in body2_ids:
                    sites2.append((j, site))

        if len(sites1) == 0 or len(sites2) == 0:
            return False

        for i, (id1, id2) in enumerate(
            zip(self.sim.model.eq_obj1id, self.sim.model.eq_obj2id)
        ):
            if id1 in (body1_ids + body2_ids) and id2 in (body1_ids + body2_ids):
                break
        else:
            return False

        # site bookkeeping
        site_bodyid = self.sim.model.site_bodyid
        body_names = self.sim.model.body_names

        for site1_id, site1_name in sites1:
            site1_pairs = site1_name.split(",")[0].split("-")
            for site2_id, site2_name in sites2:
                site2_pairs = site2_name.split(",")[0].split("-")
                # first check if already connected
                if (
                    site1_id in self._connected_sites
                    or site2_id in self._connected_sites
                ):
                    continue
                if site1_pairs == site2_pairs[::-1]:
                    if self._is_aligned(site1_name, site2_name):
                        logger.debug(
                            f"connect {site1_name} and {site2_name}, {self._connect_step}/{self._num_connect_steps}"
                        )
                        if self._connect_step < self._num_connect_steps:
                            # set target as site2 pos
                            site1_pos_quat = self._site_xpos_xquat(site1_name)
                            site1_quat = self._target_connector_xquat
                            target_pos = site1_pos_quat[:3]
                            body2id = site_bodyid[site2_id]
                            part2 = body_names[body2id]
                            part2_qpos = self._get_qpos(part2).copy()
                            site2_pos_quat = self._site_xpos_xquat(site2_name)
                            site2_pos = site2_pos_quat[:3]
                            body_pos, body_rot = T.transform_to_target_quat(
                                site2_pos_quat, part2_qpos, site1_quat
                            )
                            body_pos += target_pos - site2_pos
                            if self._connect_step == 0:
                                # generate rotation interpolations
                                self.next_rot = []
                                for f in range(self._num_connect_steps):
                                    step = (f + 1) * 1 / (self._num_connect_steps)
                                    q = T.quat_slerp(part2_qpos[3:], body_rot, step)
                                    self.next_rot.append(q)

                                # generate pos interpolation
                                x = [0, 1]
                                y = [part2_qpos[:3], body_pos]
                                f = interp1d(x, y, axis=0)
                                xnew = np.linspace(
                                    1 / self._num_connect_steps,
                                    0.9,
                                    self._num_connect_steps,
                                )
                                self.next_pos = f(xnew)

                            next_pos, next_rotation = (
                                self.next_pos[self._connect_step],
                                self.next_rot[self._connect_step],
                            )
                            self._move_objects_target(
                                part2, next_pos, list(next_rotation)
                            )
                            self._connect_step += 1
                            return False
                        else:
                            self._connect(site1_id, site2_id)
                            self._connect_step = 0
                            self.next_pos = self.next_rot = None
                            return True

        self._connect_step = 0
        return False

    def _site_xpos_xquat(self, site):
        """
        Gets the site's position and quaternion
        """
        site_id = self.sim.model.site_name2id(site)
        site_xpos = self.sim.data.get_site_xpos(site).copy()
        site_quat = self.sim.model.site_quat[site_id].copy()
        body_id = self.sim.model.site_bodyid[site_id]
        body_quat = self.sim.data.body_xquat[body_id].copy()

        site_xquat = list(Quaternion(body_quat) * Quaternion(site_quat))
        return np.hstack([site_xpos, site_xquat])

    def _is_aligned(self, connector1, connector2):
        """
        Checks if two sites are connected or not, given the site names, and
        returns possible rotations
        """
        site1_xpos = self._site_xpos_xquat(connector1)
        site2_xpos = self._site_xpos_xquat(connector2)

        allowed_angles = [x for x in connector1.split(",")[1:-1] if x]
        for i in range(len(allowed_angles)):
            allowed_angles[i] = float(allowed_angles[i])

        up1 = self._get_up_vector(connector1)
        up2 = self._get_up_vector(connector2)
        forward1 = self._get_forward_vector(connector1)
        forward2 = self._get_forward_vector(connector2)
        pos_dist = T.l2_dist(site1_xpos[:3], site2_xpos[:3])
        rot_dist_up = T.cos_dist(up1, up2)
        rot_dist_forward = T.cos_dist(forward1, forward2)

        project1_2 = np.dot(up1, T.unit_vector(site2_xpos[:3] - site1_xpos[:3]))
        project2_1 = np.dot(up2, T.unit_vector(site1_xpos[:3] - site2_xpos[:3]))

        logger.debug(
            f"pos_dist: {pos_dist}"
            + f"rot_dist_up: {rot_dist_up}"
            + f"rot_dist_forward: {rot_dist_forward}"
            + f"project: {project1_2}, {project2_1}"
        )

        max_rot_dist_forward = rot_dist_forward
        if len(allowed_angles) == 0:
            is_rot_forward_aligned = True
            cos = T.cos_dist(forward1, forward2)
            forward1_rotated_pos = T.rotate_vector_cos_dist(forward1, up1, cos, 1)
            forward1_rotated_neg = T.rotate_vector_cos_dist(forward1, up1, cos, -1)
            rot_dist_forward_pos = T.cos_dist(forward1_rotated_pos, forward2)
            rot_dist_forward_neg = T.cos_dist(forward1_rotated_neg, forward2)
            if rot_dist_forward_pos > rot_dist_forward_neg:
                forward1_rotated = forward1_rotated_pos
            else:
                forward1_rotated = forward1_rotated_neg
            max_rot_dist_forward = max(rot_dist_forward_pos, rot_dist_forward_neg)
            self._target_connector_xquat = T.convert_quat(
                T.lookat_to_quat(up1, forward1_rotated), "wxyz"
            )
        else:
            is_rot_forward_aligned = False
            for angle in allowed_angles:
                forward1_rotated = T.rotate_vector(forward1, up1, angle)
                rot_dist_forward = T.cos_dist(forward1_rotated, forward2)
                max_rot_dist_forward = max(max_rot_dist_forward, rot_dist_forward)
                if rot_dist_forward > self._env_config["rot_dist_forward"]:
                    is_rot_forward_aligned = True
                    self._target_connector_xquat = T.convert_quat(
                        T.lookat_to_quat(up1, forward1_rotated), "wxyz"
                    )
                    break

        if (
            pos_dist < self._env_config["pos_dist"]
            and rot_dist_up > self._env_config["rot_dist_up"]
            and is_rot_forward_aligned
            and abs(project1_2) > self._env_config["project_dist"]
            and abs(project2_1) > self._env_config["project_dist"]
        ):
            return True

        # connect two parts if they are very close to each other
        if (
            pos_dist < 0.03
            and rot_dist_up > self._env_config["rot_dist_up"]
            and is_rot_forward_aligned
        ):
            return True

        if pos_dist >= self._env_config["pos_dist"]:
            logger.debug(
                "(connect) two parts are too far ({} >= {})".format(
                    pos_dist, self._env_config["pos_dist"]
                )
            )
        elif rot_dist_up <= self._env_config["rot_dist_up"]:
            logger.debug(
                "(connect) misaligned ({} <= {})".format(
                    rot_dist_up, self._env_config["rot_dist_up"]
                )
            )
        elif not is_rot_forward_aligned:
            logger.debug(
                "(connect) aligned, but rotate a connector ({} <= {})".format(
                    max_rot_dist_forward, self._env_config["rot_dist_forward"]
                )
            )
        else:
            logger.debug("(connect) misaligned. move connectors to align the axis")
        return False

    def _move_objects_target(self, obj, target_pos, target_quat, gravity=1):
        """
        Moves objects toward target position and quaternion
        """
        qpos_base = self._get_qpos(obj)
        translation = target_pos - qpos_base[:3]
        self._move_objects_translation_quat(obj, translation, target_quat, gravity)

    def _move_objects_translation_quat(self, obj, translation, target_quat, gravity=1):
        """
        Moves objects with translation and target quaternion
        """
        obj_id = self._object_name2id[obj]
        qpos_base = self._get_qpos(obj)
        for i, obj_name in enumerate(self._object_names):
            if self._find_group(i) == self._find_group(obj_id):
                new_pos, new_rot = T.transform_to_target_quat(
                    qpos_base, self._get_qpos(obj_name), target_quat
                )
                new_pos = new_pos + translation
                self._set_qpos(obj_name, new_pos, new_rot)
                self._stop_object(obj_name, gravity=gravity)

    def _align_connectors(self, connector1, connector2, gravity=1):
        """
        Move connector2 to connector 1
        """
        site1_xpos = self._site_xpos_xquat(connector1)
        site1_xpos[3:] = self._target_connector_xquat
        self._move_site_to_target(connector2, site1_xpos, gravity)

    def _move_site_to_target(self, site, target_qpos, gravity=1):
        """
        Move target site towards target quaternion / position
        """
        qpos_base = self._site_xpos_xquat(site)
        target_quat = target_qpos[3:]

        site_id = self.sim.model.site_name2id(site)
        body_id = self.sim.model.site_bodyid[site_id]
        body_name = self.sim.model.body_names[body_id]
        body_qpos = self._get_qpos(body_name)
        new_pos, new_quat = T.transform_to_target_quat(
            qpos_base, body_qpos, target_quat
        )
        new_site_pos, new_site_quat = T.transform_to_target_quat(
            body_qpos, qpos_base, new_quat
        )
        translation = target_qpos[:3] - new_site_pos
        self._move_objects_translation_quat(body_name, translation, new_quat, gravity)

    def _bounded_d_pos(self, d_pos, pos):
        """
        Clips d_pos to the gripper limits
        """
        min_action = self._min_gripper_pos - pos
        max_action = self._max_gripper_pos - pos
        return np.clip(d_pos, min_action, max_action)

    def _step_continuous(self, action):
        """
        Step function for continuous control, like Sawyer and Baxter
        """
        connect = action[-1]
        if self._control_type == "ik":
            for qvel_addr in self._ref_joint_vel_indexes:
                self.sim.data.qvel[qvel_addr] = 0.0
            self.sim.forward()

            if self._agent_type in ["Sawyer", "Panda", "Jaco"]:
                action[:3] = action[:3] * self._move_speed
                action[:3] = [-action[1], action[0], action[2]]
                gripper_pos = self.sim.data.get_body_xpos("right_hand")
                d_pos = self._bounded_d_pos(action[:3], gripper_pos)
                self._initial_right_hand_quat = T.euler_to_quat(
                    action[3:6] * self._rotate_speed, self._initial_right_hand_quat
                )
                d_quat = T.quat_multiply(
                    T.quat_inverse(self._right_hand_quat), self._initial_right_hand_quat
                )
                gripper_dis = action[-2]
                action = np.concatenate([d_pos, d_quat, [gripper_dis]])

            elif self._agent_type == "Baxter":
                action[:3] = action[:3] * self._move_speed
                action[:3] = [-action[1], action[0], action[2]]
                action[6:9] = action[6:9] * self._move_speed
                action[6:9] = [-action[7], action[6], action[8]]
                right_gripper_pos = self.sim.data.get_body_xpos("right_hand")
                right_d_pos = self._bounded_d_pos(action[:3], right_gripper_pos)
                self._initial_right_hand_quat = T.euler_to_quat(
                    action[3:6] * self._rotate_speed, self._initial_right_hand_quat
                )
                right_d_quat = T.quat_multiply(
                    T.quat_inverse(self._right_hand_quat), self._initial_right_hand_quat
                )

                right_gripper_dis = action[-3]
                left_gripper_pos = self.sim.data.get_body_xpos("left_hand")
                left_d_pos = self._bounded_d_pos(action[6:9], left_gripper_pos)
                self._initial_left_hand_quat = T.euler_to_quat(
                    action[9:12] * self._rotate_speed, self._initial_left_hand_quat
                )
                left_d_quat = T.quat_multiply(
                    T.quat_inverse(self._left_hand_quat), self._initial_left_hand_quat
                )
                left_gripper_dis = action[-2]
                action = np.concatenate(
                    [
                        right_d_pos,
                        right_d_quat,
                        left_d_pos,
                        left_d_quat,
                        [right_gripper_dis, left_gripper_dis],
                    ]
                )

            input_1 = self._make_input(action[:7], self._right_hand_quat)
            if self._agent_type in ["Sawyer", "Panda"]:
                velocities = self._controller.get_control(**input_1)
                low_action = np.concatenate([velocities, action[7:8]])
            elif self._agent_type == "Jaco":
                velocities = self._controller.get_control(**input_1)
                low_action = np.concatenate([velocities] + [action[7:]] * 3)
            elif self._agent_type == "Baxter":
                input_2 = self._make_input(action[7:14], self._left_hand_quat)
                velocities = self._controller.get_control(input_1, input_2)
                low_action = np.concatenate([velocities, action[14:16]])
            else:
                raise Exception(
                    "Only Sawyer, Panda, Jaco, Baxter robot environments are supported for IK "
                    "control currently."
                )

            # keep trying to reach the target in a closed-loop
            ctrl = self._setup_action(low_action)
            for i in range(self._action_repeat):
                self._do_simulation(ctrl)

                if i + 1 < self._action_repeat:
                    velocities = self._controller.get_control()
                    if self._agent_type in ["Sawyer", "Panda"]:
                        low_action = np.concatenate([velocities, action[7:]])
                    elif self._agent_type == "Jaco":
                        low_action = np.concatenate([velocities] + [action[7:]] * 3)
                    elif self._agent_type == "Baxter":
                        low_action = np.concatenate([velocities, action[14:]])
                    ctrl = self._setup_action(low_action)

        elif self._control_type == "torque":
            self._do_simulation(action)

        if connect > 0:
            num_hands = 2 if self._agent_type == "Baxter" else 1
            for i in range(num_hands):
                touch_left_finger = {}
                touch_right_finger = {}
                for body_id in self._object_body_ids:
                    touch_left_finger[body_id] = False
                    touch_right_finger[body_id] = False

                for j in range(self.sim.data.ncon):
                    c = self.sim.data.contact[j]
                    body1 = self.sim.model.geom_bodyid[c.geom1]
                    body2 = self.sim.model.geom_bodyid[c.geom2]
                    if (
                        c.geom1 in self.l_finger_geom_ids[i]
                        and body2 in self._object_body_ids
                    ):
                        touch_left_finger[body2] = True
                    if (
                        body1 in self._object_body_ids
                        and c.geom2 in self.l_finger_geom_ids[i]
                    ):
                        touch_left_finger[body1] = True

                    if (
                        c.geom1 in self.r_finger_geom_ids[i]
                        and body2 in self._object_body_ids
                    ):
                        touch_right_finger[body2] = True
                    if (
                        body1 in self._object_body_ids
                        and c.geom2 in self.r_finger_geom_ids[i]
                    ):
                        touch_right_finger[body1] = True

                for body_id in self._object_body_ids:
                    if touch_left_finger[body_id] and touch_right_finger[body_id]:
                        logger.debug("try connect")
                        result = self._try_connect(self.sim.model.body_id2name(body_id))
                        if result:
                            return
                        break

    def _make_input(self, action, old_quat):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat.
        """
        return {
            "dpos": action[:3],
            # IK controller takes an absolute orientation in robot base frame
            "rotation": T.quat2mat(T.quat_multiply(old_quat, action[3:7])),
        }

    def _get_obs(self):
        """
        Returns observation dictionary
        """
        state = OrderedDict()

        # visual obs
        if self._visual_ob:
            camera_obs, depth_obs = self.render("rgbd_array")
            state["camera_ob"] = camera_obs
            if depth_obs is not None:
                state["depth_ob"] = depth_obs

        if self._segmentation_ob:
            segmentation_obs = self.render("segmentation")
            state["segmentation_ob"] = segmentation_obs

        # object states
        if self._object_ob:
            obj_states = OrderedDict()
            for i, obj_name in enumerate(self._object_names):
                if i in [self._subtask_part1, self._subtask_part2]:
                    obj_pos = self._get_pos(obj_name)
                    obj_quat = self._get_quat(obj_name)
                    obj_states["{}_pos".format(obj_name)] = obj_pos
                    obj_states["{}_quat".format(obj_name)] = obj_quat

            if self._subtask_part1 == -1:
                obj_states["dummy"] = np.zeros(14)

            state["object_ob"] = np.concatenate(
                [x.ravel() for _, x in obj_states.items()]
            )

        # part ids
        if self._subtask_ob:
            state["subtask_ob"] = np.array(
                [self._subtask_part1 + 1, self._subtask_part2 + 1]
            )

        return state

    def _place_objects(self):
        """
        Returns the randomly distributed initial positions and rotations of furniture parts.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        pos_init, _ = self.mujoco_model.place_objects()
        quat_init = []
        for i, body in enumerate(self._object_names):
            rotate = self._rng.randint(0, 10, size=3)
            quat_init.append(list(T.euler_to_quat(rotate)))
        return pos_init, quat_init

    def _reset(self, furniture_id=None, background=None):
        """
        Internal reset function that resets the furniture and agent
        Randomly resets furniture by disabling robot collision, spreading
        parts around, and then reenabling collision.
        """
        if self._config.furniture_name == "Random":
            furniture_id = self._rng.randint(len(furniture_xmls))
        if self._furniture_id is None or (
            self._furniture_id != furniture_id and furniture_id is not None
        ):
            # construct mujoco xml for furniture_id
            if furniture_id is None:
                self._furniture_id = self._config.furniture_id
            else:
                self._furniture_id = furniture_id
            self._reset_internal()

        # reset simulation data and clear buffers
        self.sim.reset()

        # store robot's contype, conaffinity (search MuJoCo XML API for details)
        # disable robot collision
        robot_col = {}
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name not in self._object_names and self.mujoco_robot.is_robot_part(
                geom_name
            ):
                robot_col[geom_name] = (
                    self.sim.model.geom_contype[geom_id],
                    self.sim.model.geom_conaffinity[geom_id],
                )
                self.sim.model.geom_contype[geom_id] = 0
                self.sim.model.geom_conaffinity[geom_id] = 0

        # initialize collision for non-mesh geoms
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name in self._object_names and "collision" in geom_name:
                self.sim.model.geom_contype[geom_id] = 1
                self.sim.model.geom_conaffinity[geom_id] = 1

        # initialize group
        self._object_group = list(range(len(self._object_names)))

        # initialize member variables
        self._connect_step = 0
        self._connected_sites = set()
        self._connected_body1 = None
        self._connected_body1_pos = None
        self._connected_body1_quat = None
        self._num_connected = 0
        if self._agent_type == "Cursor":
            self._cursor_selected = [None, None]

        # initialize weld constraints
        eq_obj1id = self.sim.model.eq_obj1id
        eq_obj2id = self.sim.model.eq_obj2id
        p = self._preassembled  # list of weld equality ids to activate
        if len(p) > 0:
            for eq_id in p:
                self.sim.model.eq_active[eq_id] = 1
                object_body_id1 = eq_obj1id[eq_id]
                object_body_id2 = eq_obj2id[eq_id]
                object_name1 = self._object_body_id2name[object_body_id1]
                object_name2 = self._object_body_id2name[object_body_id2]
                self._merge_groups(object_name1, object_name2)
        elif eq_obj1id is not None:
            for i, (id1, id2) in enumerate(zip(eq_obj1id, eq_obj2id)):
                self.sim.model.eq_active[i] = 1 if self._config.assembled else 0

        self._do_simulation(None)
        # stablize furniture pieces
        for _ in range(100):
            for obj_name in self._object_names:
                self._stop_object(obj_name, gravity=0)
            self.sim.forward()
            self.sim.step()

        logger.debug("*** furniture initialization ***")
        # load demonstration from filepath, initialize furniture and robot
        if self._load_demo is not None:
            pos_init = []
            quat_init = []
            for body in self._object_names:
                qpos = self._init_qpos[body]
                pos_init.append(qpos[:3])
                quat_init.append(qpos[3:])
            if self._agent_type in ["Sawyer", "Panda", "Jaco"]:
                if (
                    "l_gripper" in self._init_qpos
                    and "r_gripper" not in self._init_qpos
                    and "qpos" in self._init_qpos
                ):
                    self.sim.data.qpos[self._ref_joint_pos_indexes] = self._init_qpos[
                        "qpos"
                    ]
                    self.sim.data.qpos[
                        self._ref_gripper_joint_pos_indexes
                    ] = self._init_qpos["l_gripper"]
            elif self._agent_type == "Baxter":
                if (
                    "l_gripper" in self._init_qpos
                    and "r_gripper" in self._init_qpos
                    and "qpos" in self._init_qpos
                ):
                    self.sim.data.qpos[self._ref_joint_pos_indexes] = self._init_qpos[
                        "qpos"
                    ]
                    self.sim.data.qpos[
                        self._ref_gripper_right_joint_pos_indexes
                    ] = self._init_qpos["r_gripper"]
                    self.sim.data.qpos[
                        self._ref_gripper_left_joint_pos_indexes
                    ] = self._init_qpos["l_gripper"]
            elif self._agent_type == "Cursor":
                if "cursor0" in self._init_qpos and "cursor1" in self._init_qpos:
                    self._set_pos("cursor0", self._init_qpos["cursor0"])
                    self._set_pos("cursor1", self._init_qpos["cursor1"])
            # enable robot collision
            for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
                body_name = self.sim.model.body_names[body_id]
                geom_name = self.sim.model.geom_id2name(geom_id)
                if (
                    body_name not in self._object_names
                    and self.mujoco_robot.is_robot_part(geom_name)
                ):
                    contype, conaffinity = robot_col[geom_name]
                    self.sim.model.geom_contype[geom_id] = contype
                    self.sim.model.geom_conaffinity[geom_id] = conaffinity
        else:
            if self._config.fix_init and self._pos_init is not None:
                pos_init = self._pos_init
                quat_init = self._quat_init
            else:
                pos_init, quat_init = self._place_objects()
            self._pos_init = pos_init
            self._quat_init = quat_init

        # set furniture positions
        for i, body in enumerate(self._object_names):
            logger.debug(f"{body} {pos_init[i]} {quat_init[i]}")
            if self._config.assembled:
                self._object_group[i] = 0
            else:
                self._set_qpos(body, pos_init[i], quat_init[i])

        if self._load_demo is not None:
            self.sim.forward()
        else:
            # stablize furniture pieces
            for _ in range(100):
                self._slow_objects()
                self.sim.forward()
                self.sim.step()

            for _ in range(2):
                for obj_name in self._object_names:
                    self._stop_object(obj_name, gravity=0)
                for i in range(100):
                    self.sim.forward()
                    self.sim.step()

            # set initial pose of an agent
            self._initialize_robot_pos()

            # enable robot collision
            for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
                body_name = self.sim.model.body_names[body_id]
                geom_name = self.sim.model.geom_id2name(geom_id)
                if (
                    body_name not in self._object_names
                    and self.mujoco_robot.is_robot_part(geom_name)
                ):
                    contype, conaffinity = robot_col[geom_name]
                    self.sim.model.geom_contype[geom_id] = contype
                    self.sim.model.geom_conaffinity[geom_id] = conaffinity

            # stablize furniture pieces
            for _ in range(100):
                self._slow_objects()
                self.sim.forward()
                self.sim.step()

            for body in self._object_names:
                self._stop_object(body, gravity=0)
            for _ in range(500):
                # gravity compensation
                if self._agent_type != "Cursor":
                    self.sim.data.qfrc_applied[
                        self._ref_joint_vel_indexes
                    ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

                # set initial pose of an agent
                self._initialize_robot_pos()

                self.sim.forward()
                self.sim.step()

        # store qpos of furniture and robot
        if self._record_demo:
            self._store_qpos()

        if self._agent_type in ["Sawyer", "Panda", "Jaco", "Baxter"]:
            self._initial_right_hand_quat = self._right_hand_quat
            if self._agent_type == "Baxter":
                self._initial_left_hand_quat = self._left_hand_quat

            if self._control_type == "ik":
                # set up ik controller
                self._controller.sync_state()

        # set next subtask
        self._get_next_subtask()

        # set object positions in unity
        if self._unity:
            if background is None and self._background is None:
                background = self._config.background
            if self._config.background == "Random":
                background = self._rng.choice(background_names)
            if background and background != self._background:
                self._background = background
                self._unity.set_background(background)

    def _initialize_robot_pos(self):
        """
        Initializes robot posision with random noise perturbation
        """
        noise = self._init_random(self.mujoco_robot.init_qpos.shape, "agent")
        if self._agent_type in ["Sawyer", "Panda", "Jaco"]:
            self.sim.data.qpos[self._ref_joint_pos_indexes] = (
                self.mujoco_robot.init_qpos + noise
            )
            self.sim.data.qpos[
                self._ref_gripper_joint_pos_indexes
            ] = -self.gripper.init_qpos  # open

        elif self._agent_type == "Baxter":
            self.sim.data.qpos[self._ref_joint_pos_indexes] = (
                self.mujoco_robot.init_qpos + noise
            )
            self.sim.data.qpos[
                self._ref_gripper_right_joint_pos_indexes
            ] = -self.gripper_right.init_qpos
            self.sim.data.qpos[
                self._ref_gripper_left_joint_pos_indexes
            ] = self.gripper_left.init_qpos

        elif self._agent_type == "Cursor":
            self._set_pos("cursor0", [-0.2, 0.0, self._move_speed / 2])
            self._set_pos("cursor1", [0.2, 0.0, self._move_speed / 2])

    def _store_qpos(self):
        """
        Stores current qposition for demonstration
        """
        if self._agent_type in ["Sawyer", "Panda", "Jaco"]:
            qpos = {
                "qpos": self.sim.data.qpos[self._ref_joint_pos_indexes],
                "l_gripper": self.sim.data.qpos[self._ref_gripper_joint_pos_indexes],
            }
        elif self._agent_type == "Baxter":
            qpos = {
                "r_gripper": self.sim.data.qpos[
                    self._ref_gripper_right_joint_pos_indexes
                ],
                "l_gripper": self.sim.data.qpos[
                    self._ref_gripper_left_joint_pos_indexes
                ],
                "qpos": self.sim.data.qpos[self._ref_joint_pos_indexes],
            }
        elif self._agent_type == "Cursor":
            qpos = {
                "cursor0": self._get_pos("cursor0"),
                "cursor1": self._get_pos("cursor1"),
            }
        qpos.update({x: self._get_qpos(x) for x in self._object_names})
        self._demo.add(qpos=qpos)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # instantiate simulation from MJCF model
        self._load_model_robot()
        self._load_model_arena()
        self._load_model_object()
        self._load_model()

        # write xml for unity viewer
        if self._unity:
            self._unity.change_model(
                self.mujoco_model.get_xml(),
                self._camera_ids[0],
                self._screen_width,
                self._screen_height,
            )

        logger.debug(self.mujoco_model.get_xml())

        # construct mujoco model from xml
        self.mjpy_model = self.mujoco_model.get_model(mode="mujoco_py")
        self.sim = mujoco_py.MjSim(self.mjpy_model)
        self.initialize_time()

        self._is_render = self._visual_ob or self._render_mode != "no"
        if self._is_render:
            self._destroy_viewer()
            if self._camera_ids[0] == 0:
                # front view
                self._set_camera_position(self._camera_ids[0], [0.0, -0.7, 0.5])
                self._set_camera_rotation(self._camera_ids[0], [0.0, 0.0, 0.0])
            elif self._camera_ids[0] == 1:
                # side view
                self._set_camera_position(self._camera_ids[0], [-2.5, 0.0, 0.5])
                self._set_camera_rotation(self._camera_ids[0], [0.0, 0.0, 0.0])

        # additional housekeeping
        self._sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0

        # necessary to refresh MjData
        self.sim.forward()

        # setup mocap for ik control
        if self._control_type == "ik" and self._agent_type != "Cursor":
            import env.models

            if self._agent_type == "Sawyer":
                from env.controllers import SawyerIKController as IKController
            elif self._agent_type == "Baxter":
                from env.controllers import BaxterIKController as IKController
            elif self._agent_type == "Panda":
                from env.controllers import PandaIKController as IKController
            elif self._agent_type == "Jaco":
                from env.controllers import JacoIKController as IKController
            else:
                raise ValueError

            self._controller = IKController(
                bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )

    def _load_model_robot(self):
        """
        Loads sawyer, baxter, or cursor
        """
        use_torque = self._control_type == "torque"
        if self._agent_type == "Sawyer":
            from env.models.robots import Sawyer

            self.mujoco_robot = Sawyer(use_torque=use_torque)
            self.gripper = gripper_factory("TwoFingerGripper")
            self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper)
            self.mujoco_robot.set_base_xpos([0, 0.65, -0.7])
            self.mujoco_robot.set_base_xquat([1, 0, 0, -1])

        elif self._agent_type == "Panda":
            from env.models.robots import Panda

            self.mujoco_robot = Panda(use_torque=use_torque)
            self.gripper = gripper_factory("PandaGripper")
            self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper)
            self.mujoco_robot.set_base_xpos([0, 0.65, -0.7])
            self.mujoco_robot.set_base_xquat([1, 0, 0, -1])

        elif self._agent_type == "Jaco":
            from env.models.robots import Jaco

            self.mujoco_robot = Jaco(use_torque=use_torque)
            self.gripper = gripper_factory("JacoGripper")
            self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper)
            self.mujoco_robot.set_base_xpos([0, 0.65, -0.7])
            self.mujoco_robot.set_base_xquat([1, 0, 0, -1])

        elif self._agent_type == "Baxter":
            from env.models.robots import Baxter

            self.mujoco_robot = Baxter(use_torque=use_torque)
            self.gripper_right = gripper_factory("TwoFingerGripper")
            self.gripper_left = gripper_factory("LeftTwoFingerGripper")
            self.gripper_right.hide_visualization()
            self.gripper_left.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper_right)
            self.mujoco_robot.add_gripper("left_hand", self.gripper_left)
            self.mujoco_robot.set_base_xpos([0, 0.65, -0.7])
            self.mujoco_robot.set_base_xquat([1, 0, 0, -1])

        elif self._agent_type == "Cursor":
            from env.models.robots import Cursor

            self.mujoco_robot = Cursor()
            self.mujoco_robot.set_size(self._move_speed / 2)
            self.mujoco_robot.set_xpos([0, 0, self._move_speed / 2])

        # hide an agent
        if not self._config.render_agent:
            for x in self.mujoco_robot.worldbody.findall(".//geom"):
                x.set("rgba", "0 0 0 0")

        # no collision with an agent
        if self._config.no_collision:
            for x in self.mujoco_robot.worldbody.findall(".//geom"):
                x.set("conaffinity", "0")
                x.set("contype", "0")

    def _load_model_arena(self):
        """
        Loads the arena XML
        """
        floor_full_size = (1.0, 1.0)
        floor_friction = (2.0, 0.005, 0.0001)
        from env.models.arenas import FloorArena

        self.mujoco_arena = FloorArena(
            floor_full_size=floor_full_size, floor_friction=floor_friction
        )

    def _load_model_object(self):
        """
        Loads the object XMLs
        """
        # load models for objects
        path = xml_path_completion(furniture_xmls[self._furniture_id])
        logger.debug("load furniture %s" % path)
        objects = MujocoXMLObject(path, self._debug)
        part_names = objects.get_children_names()

        # furniture pieces
        lst = []
        for part_name in part_names:
            lst.append((part_name, objects))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)
        self.mujoco_equality = objects.equality

    def _load_model(self):
        """
        Loads the Task, which is composed of arena, robot, objects, equality
        """
        # task includes arena, robot, and objects of interest
        from env.models.tasks import FloorTask

        self.mujoco_model = FloorTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.mujoco_equality,
            self._rng,
        )

    def save_demo(self):
        """
        Saves the demonstration into a file
        """
        agent = self._agent_type
        furniture_name = furniture_names[self._furniture_id]
        self._demo.save(agent + "_" + furniture_name)

    def key_callback(self, window, key, scancode, action, mods):
        """
        Key listener for MuJoCo viewer
        """
        import glfw

        if action != glfw.RELEASE:
            return
        elif key == glfw.KEY_SPACE:
            action = "sel"
        elif key == glfw.KEY_ENTER:
            action = "des"
        elif key == glfw.KEY_W:
            action = "m_f"
        elif key == glfw.KEY_S:
            action = "m_b"
        elif key == glfw.KEY_E:
            action = "m_u"
        elif key == glfw.KEY_Q:
            action = "m_d"
        elif key == glfw.KEY_A:
            action = "m_l"
        elif key == glfw.KEY_D:
            action = "m_r"
        elif key == glfw.KEY_I:
            action = "r_f"
        elif key == glfw.KEY_K:
            action = "r_b"
        elif key == glfw.KEY_O:
            action = "r_u"
        elif key == glfw.KEY_U:
            action = "r_d"
        elif key == glfw.KEY_J:
            action = "r_l"
        elif key == glfw.KEY_L:
            action = "r_r"
        elif key == glfw.KEY_C:
            action = "connect"
        elif key == glfw.KEY_1:
            action = "switch1"
        elif key == glfw.KEY_2:
            action = "switch2"
        elif key == glfw.KEY_R:
            action = "record"
        elif key == glfw.KEY_T:
            action = "screenshot"
        elif key == glfw.KEY_Y:
            action = "save"
        elif key == glfw.KEY_ESCAPE:
            self.reset()
            return
        else:
            return

        logger.info("Input action: %s" % action)
        self.action = action
        self._action_on = True

    def key_input_unity(self):
        """
        Key input for unity If adding new keys,
        make sure to add keys to whitelist in MJTCPInterace.cs
        """
        key = self._unity.get_input()
        if key == "None":
            return
        elif key == "Space":
            action = "sel"
        elif key == "Return":
            action = "des"
        elif key == "W":
            action = "m_f"
        elif key == "S":
            action = "m_b"
        elif key == "E":
            action = "m_u"
        elif key == "Q":
            action = "m_d"
        elif key == "A":
            action = "m_l"
        elif key == "D":
            action = "m_r"
        elif key == "I":
            action = "r_f"
        elif key == "K":
            action = "r_b"
        elif key == "O":
            action = "r_u"
        elif key == "U":
            action = "r_d"
        elif key == "J":
            action = "r_l"
        elif key == "L":
            action = "r_r"
        elif key == "C":
            action = "connect"
        elif key == "Alpha1":
            action = "switch1"
        elif key == "Alpha2":
            action = "switch2"
        elif key == "R":
            action = "record"
        elif key == "T":
            action = "screenshot"
        elif key == "Y":
            action = "save"
        elif key == "Escape":
            self.reset()
            return
        else:
            return

        logger.info("Input action: %s" % action)
        self.action = action
        self._action_on = True

    def run_demo(self, config):
        """
        Since we save all qpos, just play back qpos
        """
        if config.furniture_name is not None:
            config.furniture_id = furniture_name2id[config.furniture_name]
        self.reset(config.furniture_id, config.background)
        if self._config.record:
            video_prefix = (
                self._agent_type + "_" + furniture_names[config.furniture_id] + "_"
            )
            if self._record_demo:
                vr = VideoRecorder(video_prefix=video_prefix, demo_dir=config.demo_dir)
            else:
                vr = VideoRecorder(video_prefix=video_prefix)
            vr.capture_frame(self.render("rgb_array")[0])
        with open(self._load_demo, "rb") as f:
            demo = pickle.load(f)
            all_qpos = demo["qpos"]
        try:
            for qpos in all_qpos:
                # set furniture part positions
                for i, body in enumerate(self._object_names):
                    pos = qpos[body][:3]
                    quat = qpos[body][3:]
                    self._set_qpos(body, pos, quat)
                    self._stop_object(body, gravity=0)
                # set robot positions
                if self._agent_type in ["Sawyer", "Panda", "Jaco"]:
                    self.sim.data.qpos[self._ref_joint_pos_indexes] = qpos["qpos"]
                    self.sim.data.qpos[self._ref_gripper_joint_pos_indexes] = qpos[
                        "l_gripper"
                    ]
                elif self._agent_type == "Baxter":
                    self.sim.data.qpos[self._ref_joint_pos_indexes] = qpos["qpos"]
                    self.sim.data.qpos[
                        self._ref_gripper_right_joint_pos_indexes
                    ] = qpos["r_gripper"]
                    self.sim.data.qpos[self._ref_gripper_left_joint_pos_indexes] = qpos[
                        "l_gripper"
                    ]
                elif self._agent_type == "Cursor":
                    self._set_pos("cursor0", qpos["cursor0"])
                    self._set_pos("cursor1", qpos["cursor1"])

                self.sim.forward()
                self._update_unity()
                if self._config.record:
                    vr.capture_frame(self.render("rgb_array")[0])
        finally:
            vr.close()

    def get_vr_input(self, controller):
        c = self.vr.devices[controller]
        if controller not in self.vr.devices:
            print("Lost track of ", controller)
            return None, None
        pose = c.get_pose_euler()
        state = c.get_controller_inputs()
        if pose is None or state is None:
            print("Lost track of pose ", controller)
            return None, None
        return np.asarray(pose), state

    def run_vr(self, config):
        """
        Runs the environment with HTC Vive support
        """
        from util.triad_openvr import triad_openvr

        self.vr = triad_openvr.triad_openvr()
        self.vr.print_discovered_objects()

        if config.furniture_name is not None:
            config.furniture_id = furniture_name2id[config.furniture_name]
        ob = self.reset(config.furniture_id, config.background)

        if config.render:
            self.render()

        # set initial pose of controller as origin
        origin_1, _ = self.get_vr_input("controller_1")
        origin_2 = None
        if self._agent_type == "Baxter":
            origin_2, _ = self.get_vr_input("controller_2")

        cursor_idx = 0
        flag = [-1, -1]
        t = 0
        while True:
            # get pose of the vr
            p1, s1 = self.get_vr_input("controller_1")
            if self._agent_type == "Baxter":
                p2, s2 = self.get_vr_input("controller_2")

            # check if controller is connected
            if p1 is None or s1 is None:
                time.sleep(0.1)
                continue
            if self._agent_type == "Baxter" and p2 is None or s2 is None:
                time.sleep(0.1)
                continue

            d_p1 = p1 - origin_1
            # clamp rotation
            r1 = d_p1[-3:]
            r1[np.abs(r1) < 0.0] = 0
            d_p1[-3:] = r1
            # remap xyz translation
            d_p1[1], d_p1[2] = d_p1[2], d_p1[1]
            d_p1[1] = -d_p1[1]
            # remap xyz rotation
            # wrist rotation is 3
            d_p1[3] = d_p1[4]
            if config.wrist_only:
                d_p1[[4, 5]] = 0
            origin_1 = p1

            if self._agent_type == "Baxter":
                d_p2 = p2 - origin_2
                # clamp rotation
                r2 = d_p2[-3:]
                r2[np.abs(r2) < 0.0] = 0
                d_p2[-3:] = r2
                # remap xyz translation
                d_p2[1], d_p2[2] = d_p2[2], d_p2[1]
                d_p2[1] = -d_p2[1]
                # remap xyz rotation
                d_p2[3] = d_p2[4]
                if config.wrist_only:
                    d_p2[[4, 5]] = 0
                origin_2 = p2

            if config.render:
                self.render()

            action = np.zeros((8,))
            # action_2 = np.zeros((8,))

            states = [s1, s2] if self._agent_type == "Baxter" else [s1]
            reset = False
            for cursor_idx, s in enumerate(states):
                # select
                if s["trigger"] > 0.01:
                    flag[cursor_idx] = 1
                else:
                    flag[cursor_idx] = -1

                # connect
                if s["trackpad_pressed"] != 0:
                    action[7] = 1

                # reset
                if s["grip_button"] != 0:
                    t = 0
                    flag = [-1, -1]
                    self.reset(config.furniture_id, config.background)
                    reset = True
                    break

            if reset:
                continue

            # action space is 7 dim per controller (3 xyz, 3 euler, 1 sel)
            # and then one more dim for connect
            if self._agent_type == "Cursor":
                action = np.hstack(
                    [
                        d_p1[:6],
                        [flag[0]],
                        np.zeros_like(action[:6]),
                        [flag[1], action[7]],
                    ]
                )
            elif self._agent_type in ["Sawyer", "Panda", "Jaco"]:
                action[:6] = d_p1[:6]
                action = action[:8]
                action[6] = flag[0]
            elif self._agent_type == "Baxter":
                action = np.hstack([d_p1[:6], d_p2[:6], [flag[0], flag[1], action[7]]])

            ob, reward, done, info = self.step(action)
            if config.debug:
                print("\rAction: " + str(action[:6]), end="")
            t += 1

    def run_manual(self, config):
        """ 
        Run the environment under manual (keyboard) control
        """
        if config.furniture_name is not None:
            config.furniture_id = furniture_name2id[config.furniture_name]
        ob = self.reset(config.furniture_id, config.background)

        if config.render:
            self.render()

        from util.video_recorder import VideoRecorder

        vr = None
        if self._config.record:
            video_prefix = (
                self._agent_type + "_" + furniture_names[config.furniture_id] + "_"
            )
            if self._record_demo:
                vr = VideoRecorder(video_prefix=video_prefix, demo_dir=config.demo_dir)
            else:
                vr = VideoRecorder(video_prefix=video_prefix)
            vr.capture_frame(self.render("rgb_array")[0])
        else:
            self.render()
        if not config.unity:
            # override keyboard callback function of viewer
            import glfw

            glfw.set_key_callback(self._viewer.window, self.key_callback)

        cursor_idx = 0
        flag = [-1, -1]
        t = 0
        try:
            while True:
                if config.unity:
                    self.key_input_unity()

                if config.render:
                    self.render()

                if not self._action_on:
                    time.sleep(0.1)
                    continue

                action = np.zeros((8,))

                if self.action == "switch1":
                    cursor_idx = 0
                    self._action_on = False
                    continue
                if self.action == "switch2":
                    cursor_idx = 1
                    self._action_on = False
                    continue

                # pick
                if self.action == "sel":
                    flag[cursor_idx] = 1
                if self.action == "des":
                    flag[cursor_idx] = -1

                # connect
                if self.action == "connect":
                    action[7] = 1

                # move
                if self.action == "m_f":
                    action[1] = 1
                if self.action == "m_b":
                    action[1] = -1
                if self.action == "m_u":
                    action[2] = 1
                if self.action == "m_d":
                    action[2] = -1
                if self.action == "m_l":
                    action[0] = -1
                if self.action == "m_r":
                    action[0] = 1
                # rotate
                if self.action == "r_f":
                    action[4] = 1
                if self.action == "r_b":
                    action[4] = -1
                if self.action == "r_u":
                    action[5] = 1
                if self.action == "r_d":
                    action[5] = -1
                if self.action == "r_l":
                    action[3] = -1
                if self.action == "r_r":
                    action[3] = 1

                if self.action == "record":
                    pass
                    # no longer needed to save, each frame is appeneded by default if self._config.record==True
                    #     if self._config.record:
                    #     vr.save_video('video.mp4')

                if self._agent_type == "Cursor":
                    if cursor_idx:
                        action = np.hstack(
                            [
                                np.zeros_like(action[:6]),
                                [flag[0]],
                                action[:6],
                                [flag[1], action[7]],
                            ]
                        )
                    else:
                        action = np.hstack(
                            [
                                action[:6],
                                [flag[0]],
                                np.zeros_like(action[:6]),
                                [flag[1], action[7]],
                            ]
                        )
                elif self._control_type == "ik":
                    if self._agent_type in ["Sawyer", "Panda", "Jaco"]:
                        action = action[:8]
                        action[6] = flag[0]
                    elif self._agent_type == "Baxter":
                        if cursor_idx:
                            action = np.hstack(
                                [np.zeros(6), action[:6], [flag[0], flag[1], action[7]]]
                            )
                        else:
                            action = np.hstack(
                                [action[:6], np.zeros(6), [flag[0], flag[1], action[7]]]
                            )

                ob, reward, done, info = self.step(action)
                logger.info(f"Action: {action}")

                if self._config.record:
                    vr.capture_frame(self.render("rgb_array")[0])
                else:
                    self.render("rgb_array")
                if self.action == "screenshot":
                    import imageio

                    img, depth = self.render("rgbd_array")

                    if len(img.shape) == 4:
                        img = np.concatenate(img)
                        if depth is not None:
                            depth = np.concatenate(depth)

                    imageio.imwrite("camera_ob.png", (img * 255).astype(np.uint8))
                    if self._segmentation_ob:
                        seg = self.render("segmentation")
                        if len(seg.shape) == 4:
                            seg = np.concatenate(seg)
                        color_seg = color_segmentation(seg)
                        imageio.imwrite("segmentation_ob.png", color_seg)

                    if self._depth_ob:
                        imageio.imwrite("depth_ob.png", (depth * 255).astype(np.uint8))

                if self.action == "save" and self._record_demo:
                    self.save_demo()

                self._action_on = False
                t += 1
                if done:
                    t = 0
                    flag = [-1, -1]
                    if self._record_demo:
                        self.save_demo()
                    self.reset(config.furniture_id, config.background)
                    if self._config.record:
                        # print('capture_frame3')
                        vr.capture_frame(self.render("rgb_array")[0])
                    else:
                        self.render("rgb_array")
        finally:
            if self._config.record:
                vr.close()

    def _get_reference(self):
        """
        Store ids / keys of objects, connector sites, and collision data in the scene
        """
        self._object_body_id = {}
        self._object_body_id2name = {}
        for obj_str in self.mujoco_objects.keys():
            self._object_body_id[obj_str] = self.sim.model.body_name2id(obj_str)
            self._object_body_id2name[self.sim.model.body_name2id(obj_str)] = obj_str

        # for checking distance to / contact with objects we want to pick up
        self._object_body_ids = list(map(int, self._object_body_id.values()))

        # information of objects
        self._object_names = list(self.mujoco_objects.keys())
        self._object_name2id = {k: i for i, k in enumerate(self._object_names)}
        self._object_group = list(range(len(self._object_names)))
        self._object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self._object_names
        ]

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

    def _get_next_subtask(self):
        eq_obj1id = self.sim.model.eq_obj1id
        if eq_obj1id is not None:
            for i, (id1, id2) in enumerate(
                zip(self.sim.model.eq_obj1id, self.sim.model.eq_obj2id)
            ):
                object_name1 = self._object_body_id2name[id1]
                object_name2 = self._object_body_id2name[id2]
                if self._find_group(object_name1) != self._find_group(object_name2):
                    self._subtask_part1 = self._object_name2id[object_name1]
                    self._subtask_part2 = self._object_name2id[object_name2]
                    return
        self._subtask_part1 = -1
        self._subtask_part2 = -1

    def _find_group(self, idx):
        """
        Finds the group of the object
        """
        if isinstance(idx, str):
            idx = self._object_name2id[idx]
        if self._object_group[idx] == idx:
            return idx
        self._object_group[idx] = self._find_group(self._object_group[idx])
        return self._object_group[idx]

    def _merge_groups(self, idx1, idx2):
        """
        Merges two groups into one
        """
        if isinstance(idx1, str):
            idx1 = self._object_name2id[idx1]
        if isinstance(idx2, str):
            idx2 = self._object_name2id[idx2]
        p_idx1 = self._find_group(idx1)
        p_idx2 = self._find_group(idx2)
        self._object_group[p_idx1] = p_idx2

    def _activate_weld(self, part1, part2):
        """
        Turn on weld constraint between two parts
        """
        for i, (id1, id2) in enumerate(
            zip(self.sim.model.eq_obj1id, self.sim.model.eq_obj2id)
        ):
            p1 = self.sim.model.body_id2name(id1)
            p2 = self.sim.model.body_id2name(id2)
            if p1 in [part1, part2] and p2 in [part1, part2]:
                # setup eq_data
                self.sim.model.eq_data[i] = T.rel_pose(
                    self._get_qpos(p1), self._get_qpos(p2)
                )
                self.sim.model.eq_active[i] = 1
                self._merge_groups(part1, part2)

    def _stop_object(self, obj_name, gravity=1):
        """
        Stops object by removing force and velocity. If gravity=1, then
        it compensates for gravity.
        """
        body_id = self.sim.model.body_name2id(obj_name)
        self.sim.data.xfrc_applied[body_id] = [
            0,
            0,
            -gravity
            * self.sim.model.opt.gravity[-1]
            * self.sim.model.body_mass[body_id],
            0,
            0,
            0,
        ]
        qvel_addr = self.sim.model.get_joint_qvel_addr(obj_name)
        self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = [0] * (
            qvel_addr[1] - qvel_addr[0]
        )

    def _stop_objects(self, gravity=1):
        """
        Stops all objects selected by cursor
        """
        selected_idx = []
        for obj_name in self._cursor_selected:
            if obj_name is not None:
                selected_idx.append(self._find_group(obj_name))
        for obj_name in self._object_names:
            if self._find_group(obj_name) in selected_idx:
                self._stop_object(obj_name, gravity)

    def _slow_object(self, obj_name):
        """
        Slows object by clipping qvelocity
        """
        body_id = self.sim.model.body_name2id(obj_name)
        self.sim.data.xfrc_applied[body_id] = [
            0,
            0,
            -self.sim.model.opt.gravity[-1] * self.sim.model.body_mass[body_id],
            0,
            0,
            0,
        ]
        qvel_addr = self.sim.model.get_joint_qvel_addr(obj_name)
        self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = np.clip(
            self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]], -0.2, 0.2
        )

    def _slow_objects(self):
        """
        Slow all objects
        """
        for obj_name in self._object_names:
            self._slow_object(obj_name)

    def initialize_time(self):
        """
        Initializes the time constants used for simulation.
        """
        self._cur_time = 0
        self._model_timestep = self.sim.model.opt.timestep
        self._control_timestep = 1.0 / self._control_freq

    def _do_simulation(self, a):
        """
        Take multiple physics simulation steps, bounded by self._control_timestep
        """
        try:
            self.sim.forward()

            if self.sim.data.ctrl is not None:
                self.sim.data.ctrl[:] = 0 if a is None else a

            if self._agent_type == "Cursor":
                # gravity compensation
                selected_idx = []
                for obj_name in self._cursor_selected:
                    if obj_name is not None:
                        selected_idx.append(self._find_group(obj_name))
                for obj_name in self._object_names:
                    if self._find_group(obj_name) in selected_idx:
                        self._stop_object(obj_name, gravity=1)
                    else:
                        self._stop_object(obj_name, gravity=0)

            self.sim.forward()
            for i in range(int(self._control_timestep / self._model_timestep)):
                self.sim.step()
                self._cur_time += self._model_timestep

            self._cur_time += self._control_timestep

            if self._agent_type == "Cursor":
                # gravity compensation
                for obj_name in self._object_names:
                    if self._find_group(obj_name) in selected_idx:
                        self._stop_object(obj_name, gravity=1)

        except Exception as e:
            logger.warn(
                "[!] Warning: Simulation is unstable. The episode is terminated."
            )
            logger.warn(e)
            self.reset()
            self._fail = True

    def set_state(self, qpos, qvel):
        """
        Sets the qpos and qvel of the MuJoCo sim
        """
        assert qpos.shape == (self.sim.model.nq,) and qvel.shape == (self.sim.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    def _get_cursor_pos(self, name=None):
        """
        Returns the cursor positions
        """
        if self._agent_type in ["Sawyer", "Panda", "Jaco", "Baxter"]:
            return self.sim.data.site_xpos[self.eef_site_id]
        elif self._agent_type == "Cursor":
            if name:
                return self._get_pos(name)
            else:
                return np.hstack([self._get_pos("cursor0"), self._get_pos("cursor1")])
        else:
            return None

    def _get_pos(self, name):
        """
        Get the position of a site, body, or geom
        """
        if name in self.sim.model.body_names:
            return self.sim.data.get_body_xpos(name).copy()
        if name in self.sim.model.geom_names:
            return self.sim.data.get_geom_xpos(name).copy()
        if name in self.sim.model.site_names:
            return self.sim.data.get_site_xpos(name).copy()
        raise ValueError

    def _set_pos(self, name, pos):
        """
        Set the position of a body or geom
        """
        if name in self.sim.model.body_names:
            body_idx = self.sim.model.body_name2id(name)
            self.sim.model.body_pos[body_idx] = pos[:].copy()
            return
        if name in self.sim.model.geom_names:
            geom_idx = self.sim.model.geom_name2id(name)
            self.sim.model.geom_pos[geom_idx][0:3] = pos[:].copy()
            return
        raise ValueError

    def _get_quat(self, name):
        """
        Get the quaternion of a body, geom, or site
        """
        if name in self.sim.model.body_names:
            body_idx = self.sim.model.body_name2id(name)
            return self.sim.data.body_xquat[body_idx].copy()
        if name in self.sim.model.geom_names:
            geom_idx = self.sim.model.geom_name2id(name)
            return self.sim.model.geom_quat[geom_idx].copy()
        if name in self.sim.model.site_names:
            site_idx = self.sim.model.site_name2id(name)
            return self.sim.model.site_quat[site_idx].copy()
        raise ValueError

    def _set_quat(self, name, quat):
        """
        Set the quaternion of a body
        """
        if name in self.sim.model.body_names:
            body_idx = self.sim.model.body_name2id(name)
            self.sim.model.body_quat[body_idx][0:4] = quat[:]
            return
        raise ValueError

    def _get_left_vector(self, name):
        """
        Get the left vector of a geom, or site
        """
        if name in self.sim.model.geom_names:
            return self.sim.data.get_geom_xmat(name)[:, 0].copy()
        if name in self.sim.model.site_names:
            return self.sim.data.get_site_xmat(name)[:, 0].copy()
        raise ValueError

    def _get_forward_vector(self, name):
        """
        Get the forward vector of a geom, or site
        """
        if name in self.sim.model.geom_names:
            return self.sim.data.get_geom_xmat(name)[:, 1].copy()
        if name in self.sim.model.site_names:
            return self.sim.data.get_site_xmat(name)[:, 1].copy()
        raise ValueError

    def _get_up_vector(self, name):
        """
        Get the up vector of a geom, or site
        """
        if name in self.sim.model.geom_names:
            return self.sim.data.get_geom_xmat(name)[:, 2].copy()
        if name in self.sim.model.site_names:
            return self.sim.data.get_site_xmat(name)[:, 2].copy()
        raise ValueError

    def _get_distance(self, name1, name2):
        """
        Get the distance vector of a body, geom, or site
        """
        pos1 = self._get_pos(name1)
        pos2 = self._get_pos(name2)
        return np.linalg.norm(pos1 - pos2)

    def _get_size(self, name):
        """
        Get the size of a body
        """
        body_idx1 = self.sim.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
            if body_idx1 == body_idx2:
                return self.sim.model.geom_size[geom_idx, :].copy()
        raise ValueError

    def _set_size(self, name, size):
        """
        Set the size of a body
        """
        body_idx1 = self.sim.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.sim.model.geom_size[geom_idx, :] = size
                return
        raise ValueError

    def _get_geom_type(self, name):
        """
        Get the type of a geometry
        """
        body_idx1 = self.sim.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
            if body_idx1 == body_idx2:
                return self.sim.model.geom_type[geom_idx].copy()

    def _set_geom_type(self, name, geom_type):
        """
        Set the type of a geometry
        """
        body_idx1 = self.sim.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.sim.model.geom_type[geom_idx] = geom_type

    def _get_qpos(self, name):
        """
        Get the qpos of a joint
        """
        object_qpos = self.sim.data.get_joint_qpos(name)
        return object_qpos.copy()

    def _set_qpos(self, name, pos, rot=[1, 0, 0, 0]):
        """
        Set the qpos of a joint
        """
        object_qpos = self.sim.data.get_joint_qpos(name)
        assert object_qpos.shape == (7,)
        object_qpos[:3] = pos
        object_qpos[3:] = rot
        self.sim.data.set_joint_qpos(name, object_qpos)

    def _set_qpos0(self, name, qpos):
        """
        Set the qpos0
        """
        qpos_addr = self.sim.model.get_joint_qpos_addr(name)
        self.sim.model.qpos0[qpos_addr[0] : qpos_addr[1]] = qpos

    def _set_color(self, name, color):
        """
        Set the color
        """
        body_idx1 = self.sim.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.sim.model.geom_rgba[geom_idx, 0 : len(color)] = color

    def _mass_center(self):
        """
        Get the mass center
        """
        mass = np.expand_dims(self.sim.model.body_mass, axis=1)
        xpos = self.sim.data.xipos
        return np.sum(mass * xpos, 0) / np.sum(mass)

    def on_collision(self, ref_name, body_name=None):
        """
        Checks if there is collision
        """
        mjcontacts = self.sim.data.contact
        ncon = self.sim.data.ncon
        for i in range(ncon):
            ct = mjcontacts[i]
            g1 = self.sim.model.geom_id2name(ct.geom1)
            g2 = self.sim.model.geom_id2name(ct.geom2)
            if g1 is None or g2 is None:
                continue  # geom_name can be None
            if body_name is not None:
                if (g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0) and (
                    g1.find(body_name) >= 0 or g2.find(body_name) >= 0
                ):
                    return True
            else:
                if g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0:
                    return True
        return False

    # inverse kinematics
    @property
    def _joint_positions(self):
        """
        Returns a numpy array of joint positions.
        Sawyer robots have 7 joints and positions are in rotation angles.
        """
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """
        Returns a numpy array of joint velocities.
        Sawyer robots have 7 joints and velocities are angular velocities.
        """
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    def _robot_jpos_getter(self):
        return np.array(self._joint_positions)

    def _setup_action(self, action):
        if self._rescale_actions:
            action = np.clip(action, -1, 1)

        arm_action = action[: self.mujoco_robot.dof]
        if self._agent_type in ["Sawyer", "Panda", "Jaco"]:
            gripper_action_in = action[
                self.mujoco_robot.dof : self.mujoco_robot.dof + self.gripper.dof
            ]
            gripper_action_actual = self.gripper.format_action(gripper_action_in)
            action = np.concatenate([arm_action, gripper_action_actual])

        elif self._agent_type == "Baxter":
            last = self.mujoco_robot.dof  # Degrees of freedom in arm, i.e. 14
            gripper_right_action_in = action[last : last + self.gripper_right.dof]
            last = last + self.gripper_right.dof
            gripper_left_action_in = action[last : last + self.gripper_left.dof]
            gripper_right_action_actual = self.gripper_right.format_action(
                gripper_right_action_in
            )
            gripper_left_action_actual = self.gripper_left.format_action(
                gripper_left_action_in
            )
            action = np.concatenate(
                [arm_action, gripper_right_action_actual, gripper_left_action_actual]
            )

        if self._rescale_actions:
            # rescale normalized action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            applied_action = bias + weight * action
        else:
            applied_action = action

        # gravity compensation
        self.sim.data.qfrc_applied[
            self._ref_joint_vel_indexes
        ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

        return applied_action

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_quat(self):
        """
        Returns eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._right_hand_orn)

    @property
    def _left_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("left_hand")

    @property
    def _left_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._left_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _left_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._left_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _left_hand_quat(self):
        """
        Returns eef orientation of left hand in base from of robot.
        """
        return T.mat2quat(self._left_hand_orn)
