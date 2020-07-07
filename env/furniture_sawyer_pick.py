import numpy as np

from env.furniture_sawyer import FurnitureSawyerEnv
from env.models import background_names, furniture_name2id, furniture_xmls
from util.logger import logger


class FurnitureSawyerPickEnv(FurnitureSawyerEnv):
    """
    Sawyer environment for placing a block onto table.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.furniture_id = furniture_name2id["placeblock"]

        super().__init__(config)
        # default values for rew function
        self._env_config.update(
            {
                "success_rew": config.success_rew,
                "pick_rew": config.pick_rew,
                "ctrl_penalty": config.ctrl_penalty,
                "hold_duration": config.hold_duration,
                "rand_start_range": config.rand_start_range,
                "rand_block_range": config.rand_block_range,
            }
        )
        self._gravity_compensation = 1
        # requires multiple connection actions to make connection between two
        # parts.
        self._num_connect_steps = 0
        self._discretize_grip = config.discretize_grip

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        # discretize gripper action
        if self._discretize_grip:
            a = a.copy()
            a[-2] = -1 if a[-2] < 0 else 1

        ob, _, done, _ = super(FurnitureSawyerEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)

        # for i, body in enumerate(self._object_names):
        #     pose = self._get_qpos(body)
        #     logger.debug(f"{body} {pose[:3]} {pose[3:]}")

        # info["ac"] = a

        return ob, reward, done, info

    def _reset(self, furniture_id=None, background=None):
        """
        Initialize robot at starting point of demonstration.
        Take a random step to increase diversity of starting states.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        """
        Resets simulation. Initialize with block in robot hand.
        Take a random step to increase diversity of starting states.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        self._phase = 1
        if self._config.furniture_name == "Random":
            furniture_id = self._rng.randint(len(furniture_xmls))
        if self._furniture_id is None or (
            self._furniture_id != furniture_id and furniture_id is not None
        ):
            # construct mujoco xml for furniture_id
            if furniture_id is None:
                self._furniture_id = self._config.furniture_id
            else:
                self._furniture_id = furniture_id
            self._reset_internal()

        # reset simulation data and clear buffers
        self.sim.reset()

        # store robot's contype, conaffinity (search MuJoCo XML API for details)
        # disable robot collision
        robot_col = {}
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name not in self._object_names and self.mujoco_robot.is_robot_part(
                geom_name
            ):
                robot_col[geom_name] = (
                    self.sim.model.geom_contype[geom_id],
                    self.sim.model.geom_conaffinity[geom_id],
                )
                self.sim.model.geom_contype[geom_id] = 0
                self.sim.model.geom_conaffinity[geom_id] = 0

        # initialize collision for non-mesh geoms
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name in self._object_names and "collision" in geom_name:
                self.sim.model.geom_contype[geom_id] = 1
                self.sim.model.geom_conaffinity[geom_id] = 1

        # initialize group
        self._object_group = list(range(len(self._object_names)))

        # initialize member variables
        self._connect_step = 0
        self._connected_sites = set()
        self._connected_body1 = None
        self._connected_body1_pos = None
        self._connected_body1_quat = None
        self._num_connected = 0
        if self._agent_type == "Cursor":
            self._cursor_selected = [None, None]

        self._do_simulation(None)
        # stablize furniture pieces
        for _ in range(100):
            for obj_name in self._object_names:
                self._stop_object(obj_name, gravity=0)
            self.sim.forward()
            self.sim.step()

        logger.debug("*** furniture initialization ***")
        # load demonstration from filepath, initialize furniture and robot
        self._init_qpos = {
            "qpos": [
                -0.39438281,
                -0.54315495,
                0.33605859,
                1.64807696,
                -0.56130881,
                0.56099085,
                2.12105571,
            ],
            "l_gripper": [0.00373706, -0.00226879],
            "1_block_l": [
                0.04521311,
                0.04596679,
                0.11724173,
                0.51919501,
                0.52560512,
                0.47367611,
                0.47938163,
            ],
        }
        pos_init = {}
        quat_init = {}

        for body in self._object_names:
            qpos = self._init_qpos[body]
            pos_init[body] = (qpos[:3])
            quat_init[body] = (qpos[3:])
        if self._agent_type in ["Sawyer", "Panda", "Jaco"]:
            if (
                "l_gripper" in self._init_qpos
                and "r_gripper" not in self._init_qpos
                and "qpos" in self._init_qpos
            ):
                self.sim.data.qpos[self._ref_joint_pos_indexes["right"]] = self._init_qpos[
                    "qpos"
                ]
                self.sim.data.qpos[
                    self._ref_gripper_joint_pos_indexes["right"]
                ] = self._init_qpos["l_gripper"]

        # enable robot collision
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name not in self._object_names and self.mujoco_robot.is_robot_part(
                geom_name
            ):
                contype, conaffinity = robot_col[geom_name]
                self.sim.model.geom_contype[geom_id] = contype
                self.sim.model.geom_conaffinity[geom_id] = conaffinity

        # set furniture positions
        for body in self._object_names:
            logger.debug(f"{body} {pos_init[body]} {quat_init[body]}")
            self._set_qpos(body, pos_init[body], quat_init[body])

        self.sim.forward()

        # store qpos of furniture and robot
        if self._record_demo:
            self._store_qpos()

        if self._agent_type in ["Sawyer", "Panda", "Jaco", "Baxter"]:
            self._initial_right_hand_quat = self._right_hand_quat
            if self._control_type == "ik":
                # set up ik controller
                self._controller.sync_state()

        # set next subtask
        self._get_next_subtask()

        # set object positions in unity
        if self._unity:
            if background is None and self._background is None:
                background = self._config.background
            if self._config.background == "Random":
                background = self._rng.choice(background_names)
            if background and background != self._background:
                self._background = background
                self._unity.set_background(background)

        # take some random step away from starting state
        action = np.zeros((8,))
        r = self._env_config["rand_start_range"]
        action[:3] = self._rng.uniform(-r, r, size=3)
        action[6] = 1  # grip block
        self._step_continuous(action)
        super()._reset(furniture_id, background)

    def _place_objects(self):
        """
        Returns fixed initial position and rotations of the toy table.
        The first case has the table top on the left and legs on the right.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        pos_init = {'1_block_l': [0.04521311, 0.04596679, 0.11724173]}
        quat_init = {'1_block_l': [0.51919501, 0.52560512, 0.47367611, 0.47938163]}

        return pos_init, quat_init

    def _ctrl_reward(self, action):
        if self._config.control_type == "ik":
            a = np.linalg.norm(action[:6])
        elif self._config.control_type == "impedance":
            a = np.linalg.norm(action[:7])

        # grasp_offset, grasp_leg, grip_leg
        ctrl_penalty = -self._env_config["ctrl_penalty"] * a
        if self._phase in [
            "move_leg_up",
            "move_leg",
            "connect",
        ]:  # move slower when moving leg
            ctrl_penalty *= 1

        return ctrl_penalty

    def _compute_reward(self, action):
        rew = 0
        done = self._phase == 4
        self._success = self._phase == 4
        info = {}
        return rew, done, info


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureSawyerPickEnv")
    config, unparsed = parser.parse_known_args()

    # generate placing demonstrations
    env = FurnitureSawyerPickEnv(config)
    env.run_manual(config)


if __name__ == "__main__":
    main()
