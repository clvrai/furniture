""" Wrapper class for MuJoCo-Unity plugin. """

import os
import subprocess
import time
import atexit
import glob
from sys import platform

import numpy as np

from env.mjremote import mjremote
from util.logger import logger


class UnityInterface(object):
    """
    Mujoco-Unity interface (wrapper for mjremote.py).
    """

    def __init__(self, port, unity_editor, virtual_display):
        """
        Opens @port to connect to unity.

        Args:
            port: port number to connect to Unity
            unity_editor: opens a Unity app if False, otherwise connect to Unity editor.
            virtual_display: virtual display number if needed
        """
        self._port = port
        self._unity_editor = unity_editor
        self._virtual_display = virtual_display
        self._remote = mjremote()
        self.proc1 = None

        os.makedirs('unity-xml', exist_ok=True)
        self._unity_xml_path = 'unity-xml/temp-{}.xml'.format(port)

        if not unity_editor:
            self._launch_unity(port)

        logger.info("Unity remote connecting to {}".format(port))
        while True:
            try:
                self._remote.connect(port=port)
                time.sleep(0.5)
                break
            except Exception as e:
                print("now connecting to {}".format(port))
                print(e)
                time.sleep(1)

        print("Unity remote connected to {}".format(port))

    def change_model(self, xml, camera_id, screen_width, screen_height):
        """
        Changes the mujoco scene rendered in Unity.

        Args:
            xml: path to mujoco xml.
            camera_id: id of the camera for rendering.
            screen_width: width of screen for rendering.
            screen_height: height of screen for rendering.
        """
        with open(self._unity_xml_path, 'w') as f:
            f.write(xml)

        full_path = os.path.abspath(self._unity_xml_path)
        self._remote.changeworld(full_path)
        self._remote.setcamera(camera_id)
        self._remote.setresolution(screen_width, screen_height)
        logger.debug(f"Size of qpos:{self._remote.nqpos} Size of mocap: {self._remote.nmocap}" +
                     f" No. of camera: {self._remote.ncamera}" +
                     f" Size of image w = {self._remote.width} h ={self._remote.height}")

    def get_images(self, camera_ids, render_depth=False):
        """
        Gets multiple rendered image from Unity.

        Args:
            camera_ids: cameras ids to get
            render_depth: returns depth image if True
        """
        n_camera = len(camera_ids)
        b_img = bytearray(n_camera * 3 * self._remote.height*self._remote.width)
        self._remote.getimages(b_img, camera_ids)
        img = np.reshape(b_img, (n_camera, self._remote.height, self._remote.width, 3))
        if render_depth:
            b_img = bytearray(n_camera*3*self._remote.height*self._remote.width)
            self._remote.getdepthimages(b_img, camera_ids)
            depth = np.reshape(b_img, (n_camera, self._remote.height, self._remote.width, 3))
        else:
            depth = None
        return img, depth


    def get_segmentations(self, camera_ids):
        """
        Gets segmentation maps from Unity.
        Args:
            camera_ids: camera_ids to get
        """
        n_camera = len(camera_ids)
        b_img = bytearray(n_camera*self._remote.height*self._remote.width*3)
        self._remote.getsegmentationimages(b_img, camera_ids)
        img = np.reshape(b_img, (n_camera, self._remote.height, self._remote.width, 3))
        img = img[:, ::-1, :, :]
        return img

    def get_input(self):
        """ Gets a key input from Unity. """
        return self._remote.getinput()

    def set_qpos(self, qpos):
        """ Changes qpos of the scene. """
        self._remote.setqpos(qpos)

    def set_camera_pose(self, cam_id, pose):
        """Sets xyz, wxyz of camera pose. """
        self._remote.setcamerapose(cam_id, pose)

    def set_geom_pos(self, name, pos):
        """ Changes position of geometry of the scene. """
        self._remote.setgeompos(name, pos)

    def set_background(self, background):
        """ Changes the background of the scene. """
        self._remote.setbackground(background)

    def set_quality(self, quality):
        """ Changes the graphics quality. """
        self._remote.setquality(quality)

    def disconnect_to_unity(self):
        """ Closes the connection between Unity. """
        self._remote.close()
        self.close()

    def _launch_unity(self, port):
        """
        Launches a unity app in ./binary/ and connects to the @port.
        """
        atexit.register(self.close)
        cwd = os.getcwd()
        file_name = 'binary/Furniture'
        file_name = (file_name.strip()
                     .replace('.app', '').replace('.exe', '').replace('.x86_64', '').replace('.x86', ''))
        true_filename = os.path.basename(os.path.normpath(file_name))
        logger.info('The true file name is {}'.format(true_filename))

        launch_string = None
        if platform == "linux" or platform == "linux2":
            candidates = glob.glob(os.path.join(cwd, file_name) + '.x86_64')
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(cwd, file_name) + '.x86')
            if len(candidates) == 0:
                candidates = glob.glob(file_name + '.x86_64')
            if len(candidates) == 0:
                candidates = glob.glob(file_name + '.x86')

        elif platform == 'darwin':
            candidates = glob.glob(os.path.join(cwd, file_name + '.app', 'Contents', 'MacOS', true_filename))
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(file_name + '.app', 'Contents', 'MacOS', true_filename))
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(cwd, file_name + '.app', 'Contents', 'MacOS', '*'))
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(file_name + '.app', 'Contents', 'MacOS', '*'))

        if len(candidates) > 0:
            launch_string = candidates[0]

        logger.info("This is the launch string {}".format(launch_string))
        assert launch_string is not None, 'Cannot find unity app {}'.format(
            launch_string)

        new_env = os.environ.copy()
        if self._virtual_display is not None:
            new_env["DISPLAY"] = self._virtual_display

        os.makedirs('unity-log', exist_ok=True)
        # Launch Unity environment
        self.proc1 = subprocess.Popen(
            " ".join([launch_string, "-logFile", "./unity-log/log" +
             str(port) + ".txt", '--port', str(port)]),
            shell=True, env=new_env)

    def __delete__(self):
        """ Closes the connection between Unity. """
        self.disconnect_to_unity()

    def close(self):
        """ Kills the unity app. """
        if self.proc1 is not None:
            self.proc1.kill()

