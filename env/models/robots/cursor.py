import numpy as np
from env.models.robots.robot import Robot
from env.mjcf_utils import xml_path_completion, array_to_string


class Cursor(Robot):
    """Cursor is an artificial agent with a simple action space."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/cursor/robot.xml"))

    def set_xpos(self, pos):
        """Places the cursor on position @pos."""
        node = self.worldbody.find("./body[@name='cursor0']")
        node.set("pos", array_to_string(pos))

        node = self.worldbody.find("./body[@name='cursor1']")
        node.set("pos", array_to_string(pos))

    def set_size(self, size):
        """Change the size of cursors."""
        node = self.worldbody.find("./body/geom[@name='cursor0']")
        node.set("size", array_to_string([size] * 3))
        node.set("margin", array_to_string([size]))

        node = self.worldbody.find("./body/geom[@name='cursor1']")
        node.set("size", array_to_string([size] * 3))
        node.set("margin", array_to_string([size]))

    def is_robot_part(self, body_name):
        """Checks if name is part of robot"""
        return body_name in ['cursor0', 'cursor1']

    @property
    def dof(self):
        return 14

    @property
    def joints(self):
        return []

    @property
    def init_qpos(self):
        return np.array([])
