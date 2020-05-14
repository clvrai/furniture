""" Define cursor environment class FurnitureCursorEnv. """

from collections import OrderedDict

import numpy as np
from tqdm import tqdm

from env.furniture import FurnitureEnv
from env.models import furniture_name2id
from util.logger import logger
from util.video_recorder import VideoRecorder


class FurnitureCursorToyTableDisEnv(FurnitureEnv):
    """
    Cursor environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.agent_type = "Cursor"
        config.furniture_id = furniture_name2id["toy_table"]

        super().__init__(config)
        # default values
        self._env_config.update(
            {
                "pos_dist": 0.1,
                "rot_dist_up": 0.9,
                "rot_dist_forward": 0.9,
                "project_dist": -1,
                "ctrl_penalty": config.ctrl_penalty,
                "rand_block_range": config.rand_block_range,
            }
        )

        # turn on the gravity compensation for selected furniture pieces
        self._gravity_compensation = 1
        self._num_connect_steps = 0

        self._cursor_selected = [None, None]

    @property
    def observation_space(self):
        """
        Returns the observation space.
        """
        ob_space = super().observation_space

        if self._robot_ob:
            ob_space["robot_ob"] = [(3 + 1) * 2]

        return ob_space

    @property
    def dof(self):
        """
        Returns the DoF of the cursor agent.
        """
        assert self._control_type == "ik"
        dof = (3 + 3 + 1) * 2 + 1  # (move, rotate, select) * 2 + connect
        return dof

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """

        ob, _, done, _ = super()._step(a)

        reward, done, info = self._compute_reward(a)

        if self._success:
            logger.info("Success!")

        if self._debug:
            cursor_pos = self._get_cursor_pos()
            logger.debug(f"cursors: {cursor_pos}")
            for i, body in enumerate(self._object_names):
                pose = self._get_qpos(body)
                logger.debug(f"{body} {pose[:3]} {pose[3:]}")

        return ob, reward, done, info

    def _reset(self, furniture_id=None, background=None):
        """
        Resets simulation.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        super()._reset(furniture_id, background)

        # set two bodies for picking or assemblying
        id1 = self.sim.model.eq_obj1id[0]
        id2 = self.sim.model.eq_obj2id[0]
        self._target_body1 = self.sim.model.body_id2name(id1)
        self._target_body2 = self.sim.model.body_id2name(id2)

        self._phase = 1

    def _place_objects(self):
        """
        Returns fixed initial position and rotations of the toy table.
        The first case has the table top on the left and legs on the right.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        pos_init = [
            [-0.1106984, -0.16384359, 0.05900397],
            [0.09076961, 0.04358102, 0.17718201],
        ]
        quat_init = [
            [-0.00003682, 0.70149268, 0.71195775, -0.03200301],
            [-0.00003672, 0.70149266, 0.71195776, -0.03200302],
        ]

        return pos_init, quat_init

    def _get_obs(self):
        """
        Returns the current observation.
        """
        state = super()._get_obs()

        # proprioceptive features
        if self._robot_ob:
            robot_states = OrderedDict()
            robot_states["cursor_pos"] = self._get_cursor_pos()
            robot_states["cursor_state"] = np.array(
                [
                    self._cursor_selected[0] is not None,
                    self._cursor_selected[1] is not None,
                ]
            )

            state["robot_ob"] = np.concatenate(
                [x.ravel() for _, x in robot_states.items()]
            )

        return state

    def _initialize_robot_pos(self):
        """
        Initializes cursor position to be on top of parts
        """
        self._set_pos("cursor0", [-0.35, -0.125, 0.0425])
        self._set_pos("cursor1", [0.09076961, 0.04358102, 0.17718201])

    def _compute_reward(self, action):
        rew = 0
        done = False
        info = {}
        return rew, done, info

    def generate_demos(self, num_demos):
        """
        1. Move leg to some point above
        2. Move leg to some intermediate point
        3. Move leg to table and rotate leg at same time
        """
        cfg = self._config

        for i in tqdm(range(num_demos)):
            done = False
            ob = self.reset(cfg.furniture_id, cfg.background)
            if cfg.render:
                self.render()
            vr = None
            if cfg.record:
                vr = VideoRecorder()
                vr.capture_frame(self.render("rgb_array")[0])
            cursor_pos = self._get_cursor_pos("cursor1")
            above_pos = cursor_pos + [0, 0, 0.15]
            rotate_steps = 0
            rotate_limit = 5
            step = 0
            while not done:
                action = np.zeros((15,))
                # keep left and right hand closed, disconnect
                action[-2] = action[-1] = 1
                cursor_pos = self._get_cursor_pos("cursor1")
                if self._phase == 1:
                    d = (above_pos - cursor_pos)[-1]
                    if np.linalg.norm(d) > 0.005:
                        # move left cursor up
                        action[9] = d * 12
                    else:
                        self._phase = 2
                        midpoint = [0.4, 0.04358102, 0.32718201]
                        r = self._env_config["rand_block_range"]
                        midpoint[:2] += self._rng.uniform(-r, r, size=2)
                        midpoint[2] += self._rng.uniform(0, r, size=None)
                elif self._phase == 2:
                    d = midpoint - cursor_pos
                    if np.linalg.norm(d) > 0.005:
                        # move left cursor to midpoint
                        action[7:10] = d * 12
                    else:
                        self._phase = 3

                elif self._phase == 3:
                    if rotate_steps < rotate_limit:
                        # rotate leg
                        action[-4] = -1.5
                        rotate_steps += 1
                    else:
                        self._phase = 4
                        midpoint[-1] = 0.05
                        ground_pos = midpoint

                elif self._phase == 4:
                    d = (ground_pos - cursor_pos)[-1]
                    if np.linalg.norm(d) > 0.005:
                        action[9] = d * 12
                    else:
                        self._phase = 5

                ob, reward, done, info = self.step(action)
                # self.render()
                step += 1
                if cfg.record:
                    vr.capture_frame(self.render("rgb_array")[0])
                if self._phase == 5:
                    done = True
                    self.save_demo()
                    if cfg.record:
                        vr.close(f"toytabledis_{i}.mp4")

            print(f"total steps: {step}")


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureCursorToyTableEnv")
    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Cursor environment
    env = FurnitureCursorToyTableDisEnv(config)
    env.generate_demos(config.num_demos)
    # env.run_manual(config)


if __name__ == "__main__":
    main()
