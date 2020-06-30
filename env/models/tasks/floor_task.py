from env.models.tasks import Task, UniformRandomSampler
from env.mjcf_utils import new_joint, array_to_string
from env.xml_adjusting.rescale import *
import env.transform_utils as T
import numpy as np

class FloorTask(Task):
    """
    Creates MJCF model of a floor task.

    A floor task consists of one robot interacting with a variable number of
    objects placed on the floor. This class combines the robot, the floor
    arena, and the objetcts into a single MJCF model.
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_objects, \
                mujoco_equality, furn_xyz_rand, furn_rot_rand, rng, init_qpos):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
            mujoco_objects: a list of MJCF models of physical objects
            initializer: placement sampler to initialize object positions.
        """
        super().__init__()

        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)
        self.merge_equality(mujoco_equality)
        initializer = UniformRandomSampler(rng, use_radius=False, \
           r_xyz=furn_xyz_rand, r_rot=furn_rot_rand, init_qpos=init_qpos)

        self.initializer = initializer
        self.initializer.setup(mujoco_objects, (0, -0.05, 0), (0.7, 0.7, 0))
        self.rng = rng
        self.legal_pos = None

    def resize_objects(self, resize_factor):
        self.tree = rescale(self.tree, self.root, resize_factor, write=False)
        self.root = self.tree.getroot()

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        self.objects = []  # xml manifestation
        self.targets = []  # xml manifestation
        self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free", damping="0.0001"))
            self.objects.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

    def merge_equality(self, equality):
        """Adds equality constraints  to the MJCF model."""
        for one_equality in equality:
            self.equality.append(one_equality)

    def place_objects(self):
        """Places objects randomly until no collisions or max iterations hit."""
        return self.initializer.sample()

