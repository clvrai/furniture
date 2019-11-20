import numpy as np

from env.models.base import MujocoXML
from env.mjcf_utils import xml_path_completion
from env.mjcf_utils import array_to_string, string_to_array
from env.mjcf_utils import new_geom, new_body, new_joint


class Arena(MujocoXML):
    """Base arena class."""

    def set_origin(self, offset):
        """Applies a constant offset to all objects."""
        offset = np.array(offset)
        for node in self.worldbody.findall("./*[@pos]"):
            cur_pos = string_to_array(node.get("pos"))
            new_pos = cur_pos + offset
            node.set("pos", array_to_string(new_pos))

    def add_pos_indicator(self):
        """Adds a new position indicator."""
        body = new_body(name="pos_indicator")
        body.append(
            new_geom(
                "sphere",
                [0.03],
                rgba=[1, 0, 0, 0.5],
                group=1,
                contype="0",
                conaffinity="0",
            )
        )
        body.append(new_joint(type="free", name="pos_indicator"))
        self.worldbody.append(body)


class TableArena(Arena):
    """Workspace that contains an empty table."""

    def __init__(
        self, table_full_size=(0.8, 0.8, 0.8), table_friction=(1, 0.005, 0.0001)
    ):
        """
        Args:
            table_full_size: full dimensions of the table
            friction: friction parameters of the table
        """
        super().__init__(xml_path_completion("arenas/table_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")

        self.configure_location()

    def configure_location(self):
        self.bottom_pos = np.array([0, 0, 0])
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.center_pos = self.bottom_pos + np.array([0, 0, self.table_half_size[2]])
        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        self.table_top.set(
            "pos", array_to_string(np.array([0, 0, self.table_half_size[2]]))
        )

    @property
    def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height


class FloorArena(Arena):
    """Workspace that contains an empty floor."""

    def __init__(
        self, floor_full_size=(1., 1.), floor_friction=(1, 0.005, 0.0001)
    ):
        """
        Args:
            floor_full_size: full dimensions of the floor
            friction: friction parameters of the floor
        """
        super().__init__(xml_path_completion("arenas/floor_arena.xml"))

        self.floor_full_size = np.array([floor_full_size[0], floor_full_size[1], 0.125])
        self.floor_half_size = self.floor_full_size / 2
        self.floor_friction = floor_friction

        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.floor.set("size", array_to_string(self.floor_half_size))
        self.floor.set("friction", array_to_string(self.floor_friction))

