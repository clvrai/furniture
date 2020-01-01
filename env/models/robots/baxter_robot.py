import numpy as np
from env.models.robots.robot import Robot
from env.mjcf_utils import xml_path_completion, array_to_string


class Baxter(Robot):
    """Baxter is a hunky bimanual robot designed by Rethink Robotics."""

    def __init__(
        self,
        use_torque=False,
        xml_path="robots/baxter/robot.xml",
    ):
        if use_torque:
            xml_path = "robots/baxter/robot_torque.xml"
        super().__init__(xml_path_completion(xml_path))

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

    @property
    def contact_geoms(self):
        return [
            "right_upper_shoulder_collision",
            "right_lower_shoulder_collision",
            "right_upper_elbow_collision",
            "right_lower_elbow_collision",
            "right_upper_forearm_collision",
            "right_lower_forearm_collision",
            "right_wrist_collision",
            "left_upper_shoulder_collision",
            "left_lower_shoulder_collision",
            "left_upper_elbow_collision",
            "left_lower_elbow_collision",
            "left_upper_forearm_collision",
            "left_lower_forearm_collision",
        ]
