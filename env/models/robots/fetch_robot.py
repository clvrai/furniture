import numpy as np

from env.mjcf_utils import array_to_string, xml_path_completion
from env.models.robots.robot import Robot


class Fetch(Robot):
    """Fetch is a witty single-arm robot designed by Fetch Robotics."""

    def __init__(
        self, use_torque=False, xml_path="robots/fetch/robot.xml",
    ):
        if use_torque:
            xml_path = "robots/fetch/robot_torque.xml"
        super().__init__(xml_path_completion(xml_path))

        self.bottom_offset = np.array([-0.3, -0.4, -0.213])
        self.bottom_offset = np.array([-0.3, -0.4, -0.7])

        self._init_qpos = np.array([0, -0.70, 0.00, 1.0, 0.00, 1.45, 0])
        self._init_qpos = np.array([0, 0.0, 0.00, 0.54, 0.0, 0.95, 0])

        self._model_name = "fetch"

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

    def set_joint_damping(
        self, damping=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
    ):
        """Set joint damping """

        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("damping", array_to_string(np.array([damping[i]])))

    def set_joint_frictionloss(
        self, friction=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
    ):
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("frictionloss", array_to_string(np.array([friction[i]])))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return self._joints

    @property
    def init_qpos(self):
        # return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])
        # 0: base, 1: 1st elbow, 3: 2nd elbow 5: 3rd elbow
        return self._init_qpos

    @init_qpos.setter
    def init_qpos(self, init_qpos):
        self._init_qpos = init_qpos

    @property
    def _link_body(self):
        return [
            "base_link",
            "torso_lift_link",
            "head_pan_link",
            "head_tilt_link",
            "shoulder_pan_link",
            "shoulder_lift_link",
            "upperarm_roll_link",
            "elbow_flex_link",
            "forearm_roll_link",
            "wrist_flex_link",
            "wrist_roll_link",
            "estop_link",
            "laser_link",
        ]

    @property
    def _joints(self):
        return [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

    @property
    def contact_geoms(self):
        return [
            "base_link_collision",
            "torso_lift_link_collision",
            "head_pan_link_collision",
            "head_tilt_link_collision",
            "shoulder_pan_link_collision",
            "shoulder_lift_link_collision",
            "upperarm_roll_link_collision",
            "elbow_flex_link_collision",
            "forearm_roll_link_collision",
            "wrist_flex_link_collision",
            "wrist_roll_link_collision",
            "estop_link_collision",
            "laser_link_collision",
        ]
