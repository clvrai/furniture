"""
Gripper for Kinova's Jaco (has three fingers).
"""
import numpy as np
from env.mjcf_utils import xml_path_completion
from env.models.grippers.gripper import Gripper


class JacoGripperBase(Gripper):
    """
    Gripper for Kinova's Jaco (has three fingers).
    """

    def __init__(self):
        super().__init__(xml_path_completion("grippers/jaco_gripper.xml"))

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0])

    @property
    def joints(self):
        return ["jaco_joint_finger_1", "jaco_joint_finger_2", "jaco_joint_finger_3"]

    @property
    def dof(self):
        return 3

    @property
    def visualization_sites(self):
        return ["grip_site", "grip_site_cylinder"]

    @property
    def contact_geoms(self):
        return [
            "jaco_link_finger_geom_1",
            "jaco_link_finger_geom_2",
            "jaco_link_finger_geom_3",
        ]

    @property
    def left_finger_geoms(self):
        return [
            "jaco_link_finger_geom_1",
        ]

    @property
    def right_finger_geoms(self):
        return [
            "jaco_link_finger_geom_2",
        ]


class JacoGripper(JacoGripperBase):
    pass
