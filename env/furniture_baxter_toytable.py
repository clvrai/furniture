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
        config.furniture_id = furniture_name2id["toy_table_flip"]

        super().__init__(config)
        config.discretize_grip = True
        self._gravity_compensation = 1
        # requires multiple connection actions to make connection between two
        # parts.
        self._num_connect_steps = 0
        self._discretize_grip = config.discretize_grip

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        a = a.copy()

        # discretize gripper action
        if self._discretize_grip:
            a[-2] = -1 if a[-2] < 0 else 1
            a[-3] = -1 if a[-3] < 0 else 1

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
        self._target_table_pos = [0.2, -0.1, 0.15]

    def _place_objects(self):
        """
        Returns fixed initial position and rotations of the toy table.
        The first case has the table top on the left and legs on the right.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        pos_init = {
            '4_part4': [-0.1968, -0.0288, 0.03878],
            '2_part2': [0.2, 0.16578, 0.02379]
            }

        noise = self._init_random(3 * len(pos_init), "furniture")
        for i, name in enumerate(pos_init):
            for j in range(3):
                pos_init[name][j] += noise[3 * i + j]

        quat_init = {
            '4_part4': [0.099711762, 0.00028753, 0.00037843, 0.07586979],
            '2_part2': [-0.6725, 0.6417, -0.2970, -0.2186]
            }

        return pos_init, quat_init

    def _ctrl_reward(self, action):
        if self._config.control_type in ["ik", "position_orientation"]:
            a = np.linalg.norm(action[:12])
        elif self._config.control_type in ["ik_quaternion"]:
            a = np.linalg.norm(action[:14])
        elif self._config.control_type in ["position"]:
            a = np.linalg.norm(action[:6])
        elif self._config.control_type == "impedance":
            a = np.linalg.norm(action[:14])

        ctrl_reward = -self._env_config["ctrl_reward"] * a
        return ctrl_reward

    def _compute_reward(self, action):
        """
        Two stage reward.
        The first stage gives reward for picking up a table top and a leg.

        The next stage gives reward for bringing the leg connection site close to the table
        connection site.
        """
        info = {}

        ctrl_rew = self._ctrl_reward(action)

        top_site_name = "top-leg,,conn_site4"
        up = self._get_up_vector(top_site_name)
        rot_dist_up = T.cos_dist(up, [0, 0, 1])

        table_pos = self._get_pos("4_part4")
        table_dist = T.l2_dist(table_pos, self._target_table_pos)
        table_rot_rew = 0.1 * (rot_dist_up - 1)

        # r_hand_pos = self.sim.data.site_xpos[self.right_eef_site_id]
        # l_hand_pos = self.sim.data.site_xpos[self.left_eef_site_id]
        r_hand_pos = self._site_xpos_xquat("grip_site")[:3]
        l_hand_pos = self._site_xpos_xquat("l_g_grip_site")[:3]
        r_table_pos = self._site_xpos_xquat("4_part4_right_site")[:3]
        l_table_pos = self._site_xpos_xquat("4_part4_left_site")[:3]
        r_gh_dist = T.l2_dist(r_hand_pos, r_table_pos)
        l_gh_dist = T.l2_dist(l_hand_pos, l_table_pos)
        r_gh_rew = -1.0 * (r_gh_dist if rot_dist_up < 0 else 0)
        l_gh_rew = -1.0 * (l_gh_dist if rot_dist_up < 0 else 0)

        # maximum height = 0.46
        if rot_dist_up < 0:
            lift_rew = (r_table_pos[2] - l_table_pos[2])
        else:
            lift_rew = 0
            if table_dist < 0.4:
                lift_rew = 2.0 * (0.5 - max(r_table_pos[2] - l_table_pos[2], 0))

        done = False
        success_rew = 0
        if rot_dist_up > 0.98 and table_dist < 0.2:
            done = True
            success_rew = 100
            logger.warning("Success")
            self._success = True

        rew = success_rew + ctrl_rew + table_rot_rew + r_gh_rew + l_gh_rew + lift_rew
        info["success"] = success_rew
        info["table_up"] = up
        info["table_rot_rew"] = table_rot_rew
        info["ctrl_rew"] = ctrl_rew
        info["lift_rew"] = lift_rew
        info["r_gh_dist"] = r_gh_dist
        info["l_gh_dist"] = l_gh_dist
        info["r_gh_rew"] = r_gh_rew
        info["l_gh_rew"] = l_gh_rew
        info["r_hand_pos"] = r_hand_pos
        info["l_hand_pos"] = l_hand_pos
        info["r_table_pos"] = r_table_pos
        info["l_table_pos"] = l_table_pos
        info["table_pos"] = table_pos
        info["table_dist"] = table_dist
        return rew, done, info


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureBaxterToyTableEnv")
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
