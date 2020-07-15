from typing import Tuple

import numpy as np

import env.transform_utils as T
from env.furniture_sawyer import FurnitureSawyerEnv
from env.models import furniture_name2id
from util.logger import logger


class FurnitureSawyerTableLackEnv(FurnitureSawyerEnv):
    """
    Sawyer environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.furniture_id = furniture_name2id["table_lack_0825"]

        super().__init__(config)
        # default values for rew function
        self._env_config.update(
            {
                "pos_dist": 0.04,
                "rot_dist_up": 0.95,
                "rot_dist_forward": 0.9,
                "project_dist": -1,
            }
        )
        self._pos_threshold = 0.01
        self._rot_threshold = 0.05
        self._ctrl_penalty_coef = 1
        self._eef_leg_up_dist_coef = 1
        self._eef_leg_left_dist_coef = 1
        self._above_leg_z = 0.1
        # self._gravity_compensation = 1
        # requires multiple connection actions to make connection between two
        # parts.
        self._num_connect_steps = 0
        self._discretize_grip = config.discretize_grip

    def _reset_reward_variables(self):
        self._phases = ["move_eef_above_leg", "lower_eef_to_leg", "done"]
        self._phase_i = 0
        self._current_leg = "1_leg1"
        self._current_leg_site = "leg-table,0,90,180,270,conn_site1"
        self._current_table_site = "table-leg,0,90,180,270,conn_site1"

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        # discretize gripper action
        # if self._discretize_grip:
        #     a = a.copy()
        #     a[-2] = -1 if a[-2] < 0 else 1

        ob, _, done, _ = super(FurnitureSawyerEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)

        # for i, body in enumerate(self._object_names):
        #     pose = self._get_qpos(body)
        #     logger.debug(f"{body} {pose[:3]} {pose[3:]}")

        info["ac"] = a
        return ob, reward, done, info

    def _place_objects(self):
        """
        Returns fixed initial position and rotations of the toy table.
        The first case has the table top on the left and legs on the right.

        Returns:
            xpos: x,y,z position of the objects in world frame
            xquat: quaternion of the objects
        """
        pos_init = {
            "1_leg1": [0.31343275, -0.2 + 0.35850108, 0.01831519],
            "2_leg2": [-0.24890323, -0.2 + 0.43996071, 0.01830461],
            "3_leg3": [-0.25975243, -0.2 + 0.35248785, 0.0183152],
            "4_leg4": [0.31685774, -0.2 + 0.44853931, 0.0183046],
            "5_tabletop": [0.03014604, -0.2 + 0.09554463, 0.01972958],
        }
        noise = self._init_random(3 * len(pos_init), "furniture")
        for i, name in enumerate(pos_init):
            for j in range(3):
                pos_init[name][j] += noise[3 * i + j]

        quat_init = {
            "1_leg1": [0.50863839, -0.50861836, 0.49112556, -0.4913146],
            "2_leg2": [0.51725545, 0.51737851, 0.48189126, 0.48223137],
            "3_leg3": [0.51481971, 0.51462992, 0.48482265, 0.4848337],
            "4_leg4": [0.50010382, -0.50023869, 0.49966084, -0.49999646],
            "5_tabletop": [0.00001242, -0.99988404, -0.0152285, 0.00000484],
        }
        return pos_init, quat_init

    def _compute_reward(self, action) -> Tuple[float, bool, dict]:
        """
        Multistage reward.
        While moving the leg, we need to make sure the grip is stable by measuring
        angular movements.
        At any point, the robot should minimize pose displacement in non-relevant parts.

        Phases:
        move_eef_over_leg: move eef over table leg
        lower_eef_to_leg: lower eef onto the leg
        grasp_leg: grasp the leg
        lift_leg: lift the leg
        move_leg_above_table: move the leg above the table site
        align_leg_above_table:
        """
        reward = 0
        done = False
        info = {}
        phase = self._phases[self._phase_i]

        opp_penalty, opp_info = self._other_parts_penalty()
        ctrl_penalty, ctrl_info = self._ctrl_penalty(action)
        stable_grip_rew, sg_info = self._stable_grip_reward()
        if phase == "move_eef_above_leg":
            phase_reward, phase_info = self._move_eef_above_leg_reward()
            if phase_info["move_eef_above_leg_succ"] and sg_info["stable_grip_succ"]:
                self._phase_i += 1
        elif phase == "lower_eef_to_leg":
            phase_reward, phase_info = self._lower_eef_to_leg_reward()
            if phase_info["lower_eef_to_leg_succ"] and sg_info["stable_grip_succ"]:
                self._phase_i += 1

        reward = opp_penalty + ctrl_penalty + phase_reward
        info = {**info, **opp_info, **ctrl_info, **sg_info, **phase_info}
        return reward, done, info

    def _move_eef_above_leg_reward(self) -> Tuple[float, dict]:
        """
        Moves the eef above the leg and rotates the wrist.
        Negative euclidean distance between eef xy and leg xy.

        Return negative eucl distance
        """
        eef_pos = self._get_cursor_pos()
        leg_pos = self._get_pos(self._current_leg) + [0, 0, self._above_leg_z]
        eef_above_leg_distance = -np.linalg.norm(eef_pos - leg_pos)
        rew = eef_above_leg_distance * self._eef_leg_pos_coef
        info = {"eef_above_leg_distance": eef_above_leg_distance, "eef_leg_rew": rew}
        info["move_eef_above_leg_succ"] = eef_above_leg_distance < self._pos_threshold
        assert rew <= 0
        return rew, info

    def _lower_eef_to_leg_reward(self) -> Tuple[float, dict]:
        """
        Moves the eef over the leg and rotates the wrist.
        Negative euclidean distance between eef xy and leg xy.

        Return negative eucl distance
        """
        eef_pos = self._get_cursor_pos()
        leg_pos = self._get_pos(self._current_leg)
        eef_leg_distance = -np.linalg.norm(eef_pos - leg_pos)
        rew = eef_leg_distance * self._eef_leg_pos_coef
        info = {"eef_leg_distance": eef_leg_distance, "eef_leg_rew": rew}
        info["lower_eef_over_leg_succ"] = eef_leg_distance < self._pos_threshold
        assert rew <= 0
        return rew, info

    def _stable_grip_reward(self) -> Tuple[float, dict]:
        """
        Makes sure the eef and object axes are aligned
        Returns negative angular distance
        """
        # up vector of leg and up vector of grip site should be perpendicular
        eef_up = self._get_up_vector("grip_site")
        leg_up = self._get_up_vector(self._current_leg_site)
        eef_leg_up_dist = T.cos_dist(eef_up, leg_up)
        logger.debug(f"eef_leg_up_dist: {eef_leg_up_dist}")
        eef_leg_up_rew = self._eef_leg_up_dist_coef * -np.abs(eef_leg_up_dist)

        # up vector of leg and left vector of grip site should be parallel (close to -1 or 1)
        eef_left = self._get_left_vector("grip_site")
        eef_leg_left_dist = T.cos_dist(eef_left, leg_up)
        eef_leg_left_rew = (
            np.abs(eef_leg_left_dist) - 1
        ) * self._eef_leg_left_dist_coef
        info = {
            "eef_leg_up_dist": eef_leg_up_dist,
            "eef_leg_up_rew": eef_leg_up_rew,
            "eef_leg_left_dist": eef_leg_left_dist,
            "eef_leg_left_rew": eef_leg_left_rew,
        }
        assert eef_leg_up_rew <= 0 and eef_leg_left_rew <= 0
        rew = eef_leg_up_rew + eef_leg_left_rew
        info["stable_grip_succ"] = (
            eef_leg_up_dist < self._rot_threshold
            and eef_leg_left_dist < self._rot_threshold
        )
        return rew, info

    def _ctrl_penalty(self, action) -> Tuple[float, dict]:
        rew = np.linalg.norm(action[:6]) * -self._ctrl_penalty_coef
        info = {"ctrl_penalty": rew}
        assert rew <= 0
        return rew, info

    def _other_parts_penalty(self) -> Tuple[float, dict]:
        """
        At any point, the robot should minimize pose displacement in non-relevant parts.
        Return negative reward
        """
        rew = 0
        info = {"opp_penalty": rew}
        assert rew <= 0
        return rew, info


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureSawyerToyTableEnv")
    config, unparsed = parser.parse_known_args()
    if len(unparsed):
        logger.error("Unparsed argument is detected:\n%s", unparsed)
        return

    # create an environment and run manual control of Sawyer environment
    env = FurnitureSawyerTableLackEnv(config)
    env.run_manual(config)


if __name__ == "__main__":
    main()
