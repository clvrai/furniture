import numpy as np
from env.models.robots.robot import Robot
from env.mjcf_utils import xml_path_completion, array_to_string


class Sawyer(Robot):
    """Sawyer is a witty single-arm robot designed by Rethink Robotics."""

    def __init__(
        self,
        use_torque=False,
        xml_path="robots/sawyer/robot.xml",
    ):
        if use_torque:
            xml_path = "robots/sawyer/robot_torque.xml"
        super().__init__(xml_path_completion(xml_path))

        self.bottom_offset = np.array([0, 0, -0.913])

    def set_base_xpos(self, pos):
        """
        Places the robot on position @pos.
        """
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_xquat(self, quat):
        """
        Places the robot on position @quat.
        """
        node = self.worldbody.find("./body[@name='base']")
        node.set("quat", array_to_string(quat))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return ["right_j{}".format(x) for x in range(7)]

    @property
    def init_qpos(self):
        # return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])
        # 0: base, 1: 1st elbow, 3: 2nd elbow 5: 3rd elbow
        return np.array([-0.28, -0.60, 0.00, 1.86, 0.00, 0.3, 1.57])

    @property
    def contact_geoms(self):
        return ["right_l{}_collision".format(x) for x in range(2, 7)]

