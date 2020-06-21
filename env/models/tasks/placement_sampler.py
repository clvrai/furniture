import collections
import numpy as np
from collections import namedtuple

from env.models.base import RandomizationError


class ObjectPositionSampler:
    """Base class of object placement sampler."""

    def __init__(self):
        pass

    def setup(self, mujoco_objects, table_top_offset, table_size):
        """
        Args:
            Mujoco_objcts(MujocoObject * n_obj): object to be placed
            table_top_offset(float * 3): location of table top center
            table_size(float * 3): x,y,z-FULLsize of the table
        """
        self.mujoco_objects = mujoco_objects
        self.n_obj = len(self.mujoco_objects)
        self.table_top_offset = table_top_offset
        self.table_size = table_size

    def sample(self):
        """
        Args:
            object_index: index of the current object being sampled
        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        raise NotImplementedError


class UniformRandomSampler(ObjectPositionSampler):
    """Places all objects within the table uniformly random."""

    def __init__(
        self,
        x_range=None,
        y_range=None,
        use_radius=True,
        z_rotation=None,
        rng=None,
        use_xml_init=False
    ):
        """
        Args:
            x_range(float * 2): override the x_range used to uniformly place objects
                    if None, default to x-range of table
            y_range(float * 2): override the y_range used to uniformly place objects
                    if None default to y-range of table
            x_range and y_range are both with respect to (0,0) = center of table.
            use_radius:
                True: The center of object is at position:
                     [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
                False:
                    [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
            z_rotation:
                None: Add uniform random random z-rotation
                iterable (a,b): Uniformly randomize rotation angle between a and b (in radians)
                value: Add fixed angle z-rotation
            use_xml_init:
                True: use xml initial positions as initial positions
                False: randomly sample initial positions from range +-self.table_size/2
        """
        self.x_range = x_range
        self.y_range = y_range
        self.use_radius = use_radius
        self.z_rotation = z_rotation
        self.rng = rng
        self._use_xml_init = use_xml_init
        self.Point = namedtuple('Point', ['x', 'y', 'z'])


    def setup(self, mujoco_objects, table_top_offset, table_size):
        """
        Note: overrides superclass implementation.

        Args:
            Mujoco_objcts(MujocoObject * n_obj): object to be placed
            table_top_offset(float * 3): location of table top center
            table_size(float * 3): x,y,z-FULLsize of the table
        """
        self.mujoco_objects = mujoco_objects  # should be a dictionary - (name, mjcf)
        self.n_obj = len(self.mujoco_objects)
        self.table_top_offset = table_top_offset
        self.table_size = table_size
        self.init_pos = dict()
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            if self._use_xml_init:
                obj = obj_mjcf.worldbody.find("./body[@name='%s']" % obj_name)
                obj_pos = [float(x) for x in obj.attrib['pos'].split(' ')]
                self.init_pos[obj_name] = self.Point(obj_pos[0], obj_pos[1], obj_pos[2])
            else:
                self.init_pos[obj_name] = self.Point(0, 0, 0)
        if self._use_xml_init is not True:
            orig_xrange, orig_yrange = self.x_range, self.y_range
            self.x_range, self.y_range = None, None
            init_pos, _ = self.sample()
            for obj_name, pos in init_pos.items():
                self.init_pos[obj_name] = self.Point(pos[0], pos[1], pos[2])
            self.x_range, self.y_range = orig_xrange, orig_yrange


    def sample_x(self, obj_horiz_rad):
        x_range = self.x_range
        if x_range is None:
            x_range = [-self.table_size[0] / 2, self.table_size[0] / 2]
        minimum = min(x_range)
        maximum = max(x_range)
        if self.use_radius:
            minimum += obj_horiz_rad
            maximum -= obj_horiz_rad
        return self.rng.uniform(high=maximum, low=minimum)

    def sample_y(self, obj_horiz_rad):
        y_range = self.y_range
        if y_range is None:
            y_range = [-self.table_size[0] / 2, self.table_size[0] / 2]
        minimum = min(y_range)
        maximum = max(y_range)
        if self.use_radius:
            minimum += obj_horiz_rad
            maximum -= obj_horiz_rad
        return self.rng.uniform(high=maximum, low=minimum)

    def sample_quat(self):
        if self.z_rotation is None:
            rot_angle = self.rng.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.z_rotation, collections.Iterable):
            rot_angle = self.rng.uniform(
                high=max(self.z_rotation), low=min(self.z_rotation)
            )
        else:
            rot_angle = self.z_rotation
        return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]

    # def _collision_check(obj_x, obj_y, horiz_rad, placed_objects):


    def sample(self):
        pos_arr = {}
        quat_arr = {}
        placed_objects = []
        index = 0
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            horiz_rad = obj_mjcf.get_horizontal_radius(obj_name)
            bottom_offset = obj_mjcf.get_bottom_offset(obj_name)
            success = False
            for i in range(10000):  # 1000 retries
                obj_x = self.init_pos[obj_name].x + self.sample_x(horiz_rad)
                obj_y = self.init_pos[obj_name].y + self.sample_y(horiz_rad)
                # objects cannot overlap
                location_valid = True
                for x, y, r in placed_objects:
                    if (
                        np.linalg.norm([obj_x - x, obj_y - y], 2)
                        <= r + horiz_rad
                    ):
                        location_valid = False
                        break
                if location_valid:
                    # location is valid, put the object down
                    pos = (
                        self.table_top_offset
                        - bottom_offset
                        + np.array([obj_x, obj_y, 0])
                    )
                    placed_objects.append((obj_x, obj_y, horiz_rad))
                    # random z-rotation

                    quat = self.sample_quat()

                    quat_arr[obj_name] = quat
                    pos_arr[obj_name] = pos
                    success = True
                    break
            if not success:
                raise RandomizationError("Cannot place all objects on the desk")
            index += 1

        return pos_arr, quat_arr


class UniformRandomPegsSampler(ObjectPositionSampler):
    """Places all objects on top of the table uniformly random."""

    def __init__(
        self,
        x_range=None,
        y_range=None,
        z_range=None,
        use_radius=True,
        z_rotation=True,
    ):
        """
        Args:
            x_range(float * 2): override the x_range used to uniformly place objects
                    if None, default to x-range of table
            y_range(float * 2): override the y_range used to uniformly place objects
                    if None default to y-range of table
            x_range and y_range are both with respect to (0,0) = center of table.
            use_radius:
                True: The center of object is at position:
                     [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
                False:
                    [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
            z_rotation:
                Add random z-rotation
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.use_radius = use_radius
        self.z_rotation = z_rotation

    def sample_x(self, obj_horiz_rad, x_range=None):
        if x_range is None:
            x_range = self.x_range
            if x_range is None:
                x_range = [-self.table_size[0] / 2, self.table_size[0] / 2]
        minimum = min(x_range)
        maximum = max(x_range)
        if self.use_radius:
            minimum += obj_horiz_rad
            maximum -= obj_horiz_rad
        return self.rng.uniform(high=maximum, low=minimum)

    def sample_y(self, obj_horiz_rad, y_range=None):
        if y_range is None:
            y_range = self.y_range
            if y_range is None:
                y_range = [-self.table_size[0] / 2, self.table_size[0] / 2]
        minimum = min(y_range)
        maximum = max(y_range)
        if self.use_radius:
            minimum += obj_horiz_rad
            maximum -= obj_horiz_rad
        return self.rng.uniform(high=maximum, low=minimum)

    def sample_z(self, obj_horiz_rad, z_range=None):
        if z_range is None:
            z_range = self.z_range
            if z_range is None:
                z_range = [0, 1]
        minimum = min(z_range)
        maximum = max(z_range)
        if self.use_radius:
            minimum += obj_horiz_rad
            maximum -= obj_horiz_rad
        return self.rng.uniform(high=maximum, low=minimum)

    def sample_quat(self):
        if self.z_rotation:
            rot_angle = self.rng.uniform(high=2 * np.pi, low=0)
            return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]
        else:
            return [1, 0, 0, 0]

    def sample(self):
        pos_arr = []
        quat_arr = []
        placed_objects = []

        for obj_name, obj_mjcf in self.mujoco_objects.items():
            horiz_rad = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            success = False
            for i in range(5000):  # 1000 retries
                if obj_name.startswith("SquareNut"):
                    x_range = [
                        -self.table_size[0] / 2 + horiz_rad,
                        -horiz_rad,
                    ]
                    y_range = [horiz_rad, self.table_size[0] / 2]
                else:
                    x_range = [
                        -self.table_size[0] / 2 + horiz_rad,
                        -horiz_rad,
                    ]
                    y_range = [-self.table_size[0] / 2, -horiz_rad]
                obj_x = self.sample_x(horiz_rad, x_range=x_range)
                obj_y = self.sample_y(horiz_rad, y_range=y_range)
                obj_z = self.sample_z(0.01)
                # objects cannot overlap
                location_valid = True
                pos = (
                    self.table_top_offset
                    - bottom_offset
                    + np.array([obj_x, obj_y, obj_z])
                )

                for pos2, r in placed_objects:
                    if (
                        np.linalg.norm(pos - pos2, 2) <= r + horiz_rad
                        and abs(pos[2] - pos2[2]) < 0.021
                    ):
                        location_valid = False
                        break
                if location_valid:
                    # location is valid, put the object down
                    placed_objects.append((pos, horiz_rad))
                    # random z-rotation

                    quat = self.sample_quat()

                    quat_arr.append(quat)
                    pos_arr.append(pos)
                    success = True
                    break

                # bad luck, reroll
            if not success:
                raise RandomizationError("Cannot place all objects on the desk")

        return pos_arr, quat_arr

    def setup(self, mujoco_objects, table_top_offset, table_size):
        """
        Note: overrides superclass implementation.

        Args:
            Mujoco_objcts(MujocoObject * n_obj): object to be placed
            table_top_offset(float * 3): location of table top center
            table_size(float * 3): x,y,z-FULLsize of the table
        """
        self.mujoco_objects = mujoco_objects  # should be a dictionary - (name, mjcf)
        self.n_obj = len(self.mujoco_objects)
        self.table_top_offset = table_top_offset
        self.table_size = table_size
