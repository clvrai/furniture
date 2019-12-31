import numpy as np
from env.models.robots.robot import Robot
from env.mjcf_utils import xml_path_completion, array_to_string


class Baxter(Robot):
    """Baxter is a hunky bimanual robot designed by Rethink Robotics."""

    def __init__(self, use_torque=False):
        path = "robots/baxter/robot.xml"
        if use_torque:
            path = "robots/baxter/robot_torque.xml"

        super().__init__(xml_path_completion(path))

        self.bottom_offset = np.array([0, 0, -0.913])
        self.left_hand = self.worldbody.find(".//body[@name='left_hand']")

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_xquat(self, quat):
        """Places the robot on position @quat."""
        node = self.worldbody.find("./body[@name='base']")
        node.set("quat", array_to_string(quat))

    def is_robot_part(self, geom_name):
        """Checks if name is part of robot"""
        arm_parts = geom_name in ['right_l2_geom2', 'right_l3_geom2', 'right_l4_geom2', 'right_l5_geom2', 'right_l6_geom2']
        arm_parts = arm_parts or geom_name in ['left_l2_geom2', 'left_l3_geom2', 'left_l4_geom2', 'left_l5_geom2', 'left_l6_geom2']
        gripper_parts = geom_name in ['l_finger_g0', 'l_finger_g1', 'l_fingertip_g0', 'r_finger_g0', 'r_finger_g1', 'r_fingertip_g0']
        gripper_parts = gripper_parts or geom_name in ['l_g_l_finger_g0', 'l_g_l_finger_g1', 'l_g_l_fingertip_g0', 'l_g_r_finger_g0', 'l_g_r_finger_g1', 'l_g_r_fingertip_g0']
        return arm_parts or gripper_parts

    @property
    def dof(self):
        return 14

    @property
    def joints(self):
        out = []
        for s in ["right_", "left_"]:
            out.extend(s + a for a in ["s0", "s1", "e0", "e1", "w0", "w1", "w2"])
        return out

    @property
    def init_qpos(self):
        return np.array([ 0.814, -0.44, -0.07, 0.5, 0, 1.641, -1.57629266,
                         -0.872, -0.39, 0.07, 0.5, 0, 1.641, -1.57629197])

        # Arms ready to work on the table
        return np.array([
            0.535, -0.093, 0.038, 0., 0., 1.960, -1.297,
            -0.518, -0.026, -0.076, 0., 0., 1.641, -0.158])

        # Arms fully extended
        return np.zeros(14)

        # Arms half extended
        return np.array([
            0.752, -0.038, -0.021, 0.161, 0.348, 2.095, -0.531,
            -0.585, -0.117, -0.037, 0.164, -0.536, 1.543, 0.204])

        # Arm within small range
        return np.array([
            1.1, -0.8, 0, 1.7, 0.0, 0.7, 0.4,
            -0.8, -0.8, 0, 1.7, 0.0, 0.7, 0])

