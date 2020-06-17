""" Define baxter environment class FurnitureBaxterToyTableEnv. """
import numpy as np
import pickle

import env.transform_utils as T
from env.furniture_baxter import FurnitureBaxterEnv
from env.models import furniture_name2id
from util import clamp
from util.logger import logger


class FurnitureBaxterToyTableEnv(FurnitureBaxterEnv):
    """
    Baxter environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.furniture_id = furniture_name2id["toy_table"]

        super().__init__(config)
        config.discretize_grip = True
        self._gravity_compensation = 1
        # requires multiple connection actions to make connection between two
        # parts.
        self._num_connect_steps = 0
        self._discretize_grip = config.discretize_grip
        self.gripped_count = 0

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        # discretize gripper action
        if self._discretize_grip:
            a = a.copy()
            a[-2] = -1 if a[-2] < 0 else 1

        ob, _, done, _ = super(FurnitureBaxterEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)

        info["ac"] = a

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

    def _place_objects(self):
        """
        Returns fixed initial position and rotations of the toy table.
        The first case has the table top on the left and legs on the right.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        pos_init = [
            [-0.1968, -0.0288, 0.03878],
            [0.2, 0.16578, 0.02379],
        ]
        noise = self._init_random(3 * len(pos_init), "furniture")
        for i in range(len(pos_init)):
            for j in range(3):
                pos_init[i][j] += noise[3 * i + j]

        quat_init = [
            [0.099711762, 0.00028753, 0.00037843, 0.07586979],
            [-0.6725, 0.6417, -0.2970, -0.2186],
        ]
        return pos_init, quat_init

    def _ctrl_reward(self, action):
        if self._config.control_type == "ik":
            a = np.linalg.norm(action[:6])
        elif self._config.control_type == "impedance":
            a = np.linalg.norm(action[:7])

        ctrl_penalty = -self._env_config["ctrl_penalty"] * a
        return ctrl_penalty

    def _compute_reward(self, action):
        """
        Two stage reward.
        The first stage gives reward for picking up a table top and a leg.

        The next stage gives reward for bringing the leg connection site close to the table
        connection site.
        """

        info = {}

        top_site_name = "top-leg,,conn_site4"
        up = self._get_up_vector(top_site_name)
        rot_dist_up = T.cos_dist(up, [0, 0, 1])

        done = False
        success_rew = 0
        if rot_dist_up > 0.98:
            done = True
            success_rew = 1
            logger.warning("Success")
            self._success = True

        rew = success_rew
        info["success"] = success_rew
        info["table_up"] = up
        return rew, done, info


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureBaxterToyTableEnv")
    parser.set_defaults(wrist_only=True)
    parser.set_defaults(max_episode_steps=1000)
    config, unparsed = parser.parse_known_args()
    if len(unparsed):
        logger.error("Unparsed argument is detected:\n%s", unparsed)
        return

    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterToyTableEnv(config)
    # env.run_manual(config)
    env.run_vr(config)


if __name__ == "__main__":
    main()
