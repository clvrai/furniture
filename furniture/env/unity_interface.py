""" Wrapper class for MuJoCo-Unity plugin. """

import os
import subprocess
import signal
import time
import atexit
import glob
from sys import platform
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import gdown

from .mjremote import mjremote
from ..util.logger import logger


APP_GDRIVE_ID = {
    "linux": "1LCpLtwMov1pd5XesOsTD-PfOQtbvJZWb",
    "linux2": "1LCpLtwMov1pd5XesOsTD-PfOQtbvJZWb",
    "darwin": "1nk1zJTYU5CI76r8sCiHJJAjQg28EgwsC",
    "win32": "1mr4RQMxcArHCU8Pj2nDZl5EihHfFNkjo",
}


APP_FILE_NAME = {
    "linux": "ubuntu_binary.zip",
    "linux2": "ubuntu_binary.zip",
    "darwin": "mac_binary.zip",
    "win32": "windows_binary.zip",
}


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

        os.makedirs("unity-xml", exist_ok=True)
        self._unity_xml_path = "unity-xml/temp-{}.xml".format(port)

        if not unity_editor:
            self._launch_unity(port)

        logger.info("Unity remote connecting to {}".format(port))
        while True:
            try:
                self._remote.connect(port=port)
                time.sleep(0.5)
                break
            except Exception as e:
                logger.info("now connecting to {}".format(port))
                logger.info(e)
                time.sleep(1)

        logger.info("Unity remote connected to {}".format(port))

    def change_model(
        self, xml=None, xml_path=None, camera_id=0, screen_width=500, screen_height=500
    ):
        """
        Changes the mujoco scene rendered in Unity.

        Args:
            xml: mujoco xml.
            xml_path: path to mujoco xml.
            camera_id: id of the camera for rendering.
            screen_width: width of screen for rendering.
            screen_height: height of screen for rendering.
        """
        if xml is not None:
            with open(self._unity_xml_path, "w") as f:
                f.write(xml)
            full_path = os.path.abspath(self._unity_xml_path)
        else:
            full_path = os.path.abspath(xml_path)
        self._remote.changeworld(full_path)
        self._remote.setcamera(camera_id)
        self._remote.setresolution(screen_width, screen_height)
        self._camera_id = camera_id
        logger.debug(
            f"Size of qpos:{self._remote.nqpos} Size of mocap: {self._remote.nmocap}"
            + f" No. of camera: {self._remote.ncamera}"
            + f" Size of image w = {self._remote.width} h ={self._remote.height}"
        )

    def get_images(self, camera_ids=None, render_depth=False):
        """
        Gets multiple rendered image from Unity.

        Args:
            camera_ids: cameras ids to get
            render_depth: returns depth image if True
        """
        if camera_ids is None:
            camera_ids = [self._camera_id]
        n_camera = len(camera_ids)
        b_img = bytearray(n_camera * 3 * self._remote.height * self._remote.width)
        self._remote.getimages(b_img, camera_ids)
        img = np.reshape(b_img, (n_camera, self._remote.height, self._remote.width, 3))
        if render_depth:
            b_img = bytearray(n_camera * 3 * self._remote.height * self._remote.width)
            self._remote.getdepthimages(b_img, camera_ids)
            depth = np.reshape(
                b_img, (n_camera, self._remote.height, self._remote.width, 3)
            )
        else:
            depth = None
        return img, depth

    def get_segmentations(self, camera_ids=None):
        """
        Gets segmentation maps from Unity.
        Args:
            camera_ids: camera_ids to get
        """
        if camera_ids is None:
            camera_ids = [self._camera_id]
        n_camera = len(camera_ids)
        b_img = bytearray(n_camera * self._remote.height * self._remote.width * 3)
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

    def set_camera_id(self, cam_id):
        """Sets camera id. """
        self._remote.setcamera(cam_id)

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

    def _download_unity(self):
        """ Downloads Unity app from Google Drive. """
        url = "https://drive.google.com/uc?id=" + APP_GDRIVE_ID[platform]
        # os.makedirs("binary", exist_ok=True)
        # zip_path = os.path.join("binary", APP_FILE_NAME[platform])
        zip_path = APP_FILE_NAME[platform]
        if os.path.exists(zip_path):
            logger.info("%s is already downloaded.", zip_path)

            with ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall()
        else:
            logger.info("Downloading Unity app from %s", url)
            gdown.cached_download(url, zip_path, postprocess=gdown.extractall)

        if platform == "darwin":
            import stat

            os.chmod("binary/Furniture.app/Contents/MacOS/Furniture", stat.S_IEXEC)

    def _find_unity_path(self):
        """ Finds path to Unity app. """
        cwd = os.getcwd()
        file_name = "binary/Furniture"
        true_filename = "Furniture"

        launch_string = None
        if platform == "linux" or platform == "linux2":
            candidates = glob.glob(os.path.join(cwd, file_name) + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(cwd, file_name) + ".x86")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86")

        elif platform == "darwin":
            candidates = glob.glob(
                os.path.join(
                    cwd, file_name + ".app", "Contents", "MacOS", true_filename
                )
            )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(file_name + ".app", "Contents", "MacOS", true_filename)
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(cwd, file_name + ".app", "Contents", "MacOS", "*")
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(file_name + ".app", "Contents", "MacOS", "*")
                )

        elif platform == "win32":
            candidates = glob.glob(os.path.join(cwd, file_name) + ".exe")

        if len(candidates) > 0:
            launch_string = candidates[0]
        return launch_string

    def _launch_unity(self, port):
        """
        Launches a unity app in ./binary/ and connects to the @port.
        """
        atexit.register(self.close)

        launch_string = self._find_unity_path()
        if launch_string is None:
            self._download_unity()
            launch_string = self._find_unity_path()

        logger.info("This is the launch string {}".format(launch_string))
        assert launch_string is not None, "Cannot find unity app {}".format(
            launch_string
        )

        new_env = os.environ.copy()
        if self._virtual_display is not None:
            new_env["DISPLAY"] = self._virtual_display

        os.makedirs("unity-log", exist_ok=True)

        # Launch Unity environment
        if platform == "win32":
            self.proc1 = subprocess.Popen(
                " ".join(
                    [
                        launch_string,
                        "-logFile",
                        "./unity-log/log" + str(port) + ".txt",
                        "--port",
                        str(port),
                    ]
                ),
                shell=False,
                env=new_env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            self.proc1 = subprocess.Popen(
                " ".join(
                    [
                        launch_string,
                        "-logFile",
                        "./unity-log/log" + str(port) + ".txt",
                        "--port",
                        str(port),
                    ]
                ),
                shell=True,
                env=new_env,
                preexec_fn=os.setsid,
            )

    def __delete__(self):
        """ Closes the connection between Unity. """
        self.disconnect_to_unity()

    def close(self):
        """ Kills the unity app. """
        if self.proc1 is not None:
            if platform == "win32":
                self.proc1.send_signal(signal.CTRL_BREAK_EVENT)
                self.proc1.kill()
            else:
                os.killpg(os.getpgid(self.proc1.pid), signal.SIGTERM)
            self.proc1 = None
