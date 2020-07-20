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
        config.object_ob_all = False
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
        self._phase_bonus = config.phase_bonus
        self._ctrl_penalty_coef = config.ctrl_penalty_coef
        self._pos_threshold = config.pos_threshold
        self._rot_threshold = config.rot_threshold
        self._rot_dist_coef = config.rot_dist_coef
        self._pos_dist_coef = config.pos_dist_coef
        self._above_leg_z = config.above_leg_z
        self._gripper_penalty_coef = config.gripper_penalty_coef
        self._align_rot_dist_coef = config.align_rot_dist_coef
        self._fine_align_rot_dist_coef = config.fine_align_rot_dist_coef
        self._touch_coef = config.touch_coef
        # requires multiple connection actions to make connection between two
        # parts.
        self._num_connect_steps = 0
        self._discrete_grip = config.discrete_grip
        self._grip_open_phases = set(["move_eef_above_leg", "lower_eef_to_leg"])
        self._phases = ["move_eef_above_leg", "lower_eef_to_leg", "lift_leg"]
        self._phases.extend(["move_eef_over_conn", "align_leg", "lower_leg"])
        self._phases.extend(["align_leg_fine, lower_leg_fine"])

    def _reset_reward_variables(self):
        self._phase_i = 1
        self._current_leg = "0_part0"
        self._current_leg_site = "leg-table,0,90,180,270,conn_site1"
        self._current_table_site = "table-leg,0,90,180,270,conn_site1"
        self._subtask_part1 = self._object_name2id[self._current_leg]
        self._subtask_part2 = self._object_name2id["4_part4"]

    def _reset(self, furniture_id=None, background=None):
        super()._reset(furniture_id, background)
        self._reset_reward_variables()

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        # discretize gripper action
        if self._discrete_grip:
            a = a.copy()
            a[-2] = -1 if a[-2] < 0 else 1

        ob, _, done, _ = super(FurnitureSawyerEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)
        return ob, reward, done, info

    def _place_objects(self):
        """
        Returns fixed initial position and rotations of the toy table.
        The first case has the table top on the left and legs on the right.

        Returns:
            xpos: x,y,z position of the objects in world frame
            xquat: quaternion of the objects
        """
        # pos_init = {
        #     "0_part0": [0.31343275, -0.2 + 0.35850108, 0.01831519],
        #     "1_part1": [-0.24890323, -0.2 + 0.43996071, 0.01830461],
        #     "2_part2": [-0.25975243, -0.2 + 0.35248785, 0.0183152],
        #     "3_part3": [0.31685774, -0.2 + 0.44853931, 0.0183046],
        #     "4_part4": [0.03014604, -0.2 + 0.09554463, 0.01972958],
        # }
        pos_init = {
            "0_part0": [0.0055512, 0.09120562, 0.01831519],
            "1_part1": [-0.24890323, -10.2 + 0.43996071, 0.01830461],
            "2_part2": [-0.25975243, -10.2 + 0.35248785, 0.0183152],
            "3_part3": [0.31685774, -10.2 + 0.44853931, 0.0183046],
            "4_part4": [0.03014604, -0.3 + 0.09554463, 0.01972958],
        }
        noise = self._init_random(3 * len(pos_init), "furniture")
        for i, name in enumerate(pos_init):
            for j in range(3):
                pos_init[name][j] += noise[3 * i + j]

        quat_init = {
            # "0_part0": [0.50863839, -0.50861836, 0.49112556, -0.4913146],
            "0_part0": [0, 0, -0.707, -0.707],
            "1_part1": [0.51725545, 0.51737851, 0.48189126, 0.48223137],
            "2_part2": [0.51481971, 0.51462992, 0.48482265, 0.4848337],
            "3_part3": [0.50010382, -0.50023869, 0.49966084, -0.49999646],
            "4_part4": [0.00001242, -0.99988404, -0.0152285, 0.00000484],
        }
        return pos_init, quat_init

    def _compute_reward(self, ac) -> Tuple[float, bool, dict]:
        """
        Multistage reward.
        While moving the leg, we need to make sure the grip is stable by measuring
        angular movements.
        At any point, the robot should minimize pose displacement in non-relevant parts.

        Phases:
        move_eef_over_leg: move eef over table leg
        lower_eef_to_leg: lower eef onto the leg
        lift_leg: grip and lift the leg
        move_eef_over_conn: move the eef (holding leg) above the conn site
        align_leg: coarsely align the leg with the conn site
        lower_leg: move the leg 0.05 cm above the conn site
        align_leg_fine: fine grain alignment of the up and forward vectors
        lower_leg_fine: finely move the leg onto the conn site
        """
        phase_bonus = reward = 0
        done = False
        info = {}
        phase = self._phases[self._phase_i]

        opp_penalty, opp_info = self._other_parts_penalty()
        ctrl_penalty, ctrl_info = self._ctrl_penalty(ac)
        stable_grip_rew, sg_info = self._stable_grip_reward()
        grip_penalty, grip_info = self._gripper_penalty(ac)
        if phase == "move_eef_above_leg":
            phase_reward, phase_info = self._move_eef_above_leg_reward()
            if phase_info[f"{phase}_succ"] and sg_info["stable_grip_succ"]:
                self._phase_i += 1
                phase_bonus = self._phase_bonus
        elif phase == "lower_eef_to_leg":
            phase_reward, phase_info = self._lower_eef_to_leg_reward()
            if phase_info[f"{phase}_succ"] and sg_info["stable_grip_succ"]:
                print(f"DONE WITH PHASE {phase}")
                self._phase_i += 1
                phase_bonus = self._phase_bonus
        elif phase == "lift_leg":
            phase_reward, phase_info = self._lift_leg_reward()
            if phase_info[f"{phase}_succ"]:
                print(f"DONE WITH PHASE {phase}")
                self._phase_i += 1
                phase_bonus = self._phase_bonus
        elif phase == "move_eef_over_conn":
            phase_reward, phase_info = self._move_eef_over_conn_reward()
            if phase_info[f"{phase}_succ"]:
                print(f"DONE WITH PHASE {phase}")
                self._phase_i += 1
                phase_bonus = self._phase_bonus
        elif phase == "align_leg":
            phase_reward, phase_info = self._align_leg_reward()
            if phase_info[f"{phase}_succ"]:
                print(f"DONE WITH PHASE {phase}")
                self._phase_i += 1
                phase_bonus = self._phase_bonus
        elif phase == "lower_leg":
            phase_reward, phase_info = self._lower_leg_reward()
            if phase_info[f"{phase}_succ"]:
                print(f"DONE WITH PHASE {phase}")
                self._phase_i += 1
                phase_bonus = self._phase_bonus
        elif phase == "align_leg_fine":
            phase_reward, phase_info = self._align_leg_fine_reward()
            if phase_info[f"{phase}_succ"]:
                print(f"DONE WITH PHASE {phase}")
                self._phase_i += 1
                phase_bonus = self._phase_bonus
        elif phase == "lower_leg_fine":
            phase_reward, phase_info = self._lower_leg_fine_reward()
            if phase_info[f"{phase}_succ"]:
                print(f"DONE WITH PHASE {phase}")
                self._phase_i += 1
                phase_bonus = self._phase_bonus
        else:
            phase_reward, phase_info = 0, {}
            done = True

        reward += opp_penalty + ctrl_penalty + phase_reward + stable_grip_rew
        reward += grip_penalty + phase_bonus
        info["phase_bonus"] = phase_bonus
        info = {**info, **opp_info, **ctrl_info, **phase_info, **sg_info, **grip_info}
        # log phase if last frame
        if self._episode_length == self._env_config["max_episode_steps"] - 1:
            info["phase"] = self._phase_i
        return reward, done, info

    def _move_eef_above_leg_reward(self) -> Tuple[float, dict]:
        """
        Moves the eef above the leg and rotates the wrist.
        Negative euclidean distance between eef xy and leg xy.

        Return negative eucl distance
        """
        eef_pos = self._get_gripper_pos()
        leg_pos1 = self._get_pos(self._current_leg)
        leg_pos1[2] = self._above_leg_z
        leg_pos2 = leg_pos1 + [0, 0, 0.03]
        leg_pos = np.concatenate([leg_pos1, leg_pos2])
        eef_above_leg_distance = np.linalg.norm(eef_pos - leg_pos)
        rew = -eef_above_leg_distance * self._pos_dist_coef
        info = {"eef_above_leg_dist": eef_above_leg_distance, "eef_leg_rew": rew}
        info["move_eef_above_leg_succ"] = eef_above_leg_distance < 0.03
        assert rew <= 0
        return rew, info

    def _lower_eef_to_leg_reward(self) -> Tuple[float, dict]:
        """
        Moves the eef over the leg and rotates the wrist.
        Negative euclidean distance between eef xy and leg xy.
        Give additional reward for contacting the leg
        Return negative eucl distance
        """
        left, right = self._finger_contact(self._current_leg)
        leg_touched = int(left and right)
        touch_rew = (leg_touched - 1) * self._touch_coef
        info = {"touch": leg_touched, "touch_rew": touch_rew}

        eef_pos = self._get_gripper_pos()
        leg_pos1 = self._get_pos(self._current_leg) + [0, 0, -0.015]
        leg_pos2 = leg_pos1 + [0, 0, 0.03]
        leg_pos = np.concatenate([leg_pos1, leg_pos2])
        xy_distance = np.linalg.norm(eef_pos[:2] - leg_pos[:2])
        z_distance = np.abs(eef_pos[2] - leg_pos[2])
        eef_leg_distance = xy_distance + z_distance
        # eef_leg_distance = np.linalg.norm(eef_pos - leg_pos)
        rew = -eef_leg_distance * self._pos_dist_coef
        info.update({"eef_leg_dist": eef_leg_distance, "eef_leg_rew": rew})
        info["lower_eef_to_leg_succ"] = xy_distance < 0.015 and z_distance < 0.005
        assert rew <= 0
        return rew, info

    def _lift_leg_reward(self) -> Tuple[float, dict]:
        """
        Lift the leg to a target position
        Return negative eucl distance
        """
        left, right = self._finger_contact(self._current_leg)
        leg_touched = int(left and right)
        touch_rew = (leg_touched - 1) * self._touch_coef
        info = {"touch": leg_touched, "touch_rew": touch_rew}

        leg_pos = self._get_pos(self._current_leg)
        lift_leg_pos = self._get_pos(self._current_leg)
        lift_leg_pos[2] = 0.15
        lift_leg_distance = np.linalg.norm(lift_leg_pos - leg_pos)
        rew = -lift_leg_distance * self._pos_dist_coef
        info.update({"lift_leg_dist": lift_leg_distance, "lift_leg_rew": rew})
        info["lift_leg_succ"] = lift_leg_distance < self._pos_threshold
        assert rew <= 0
        # print(lift_leg_distance)
        return rew, info

    def _move_eef_over_conn_reward(self) -> Tuple[float, dict]:
        """
        Moves the eef holding leg over connection site
        Negative euclidean distance between eef xy and conn xy.
        Return negative eucl distance
        """
        above_conn_pos = self._get_pos(self._current_table_site)
        above_conn_pos[2] = 0.15
        eef_pos = self._get_cursor_pos()
        eef_conn_distance = np.linalg.norm(eef_pos - above_conn_pos)
        rew = -eef_conn_distance * self._pos_dist_coef
        info = {"eef_conn_dist": eef_conn_distance, "eef_conn_rew": rew}
        info["move_eef_over_conn_succ"] = eef_conn_distance < self._pos_threshold
        assert rew <= 0
        # print(eef_conn_distance)
        return rew, info

    def _align_leg_reward(self) -> Tuple[float, dict]:
        """
        Align the leg over the connection site
        Up vectors of leg conn and table conn should be parallel (cos dist 1)
        Return negative angular distance
        """
        leg_up = self._get_up_vector(self._current_leg_site)
        table_up = self._get_up_vector(self._current_table_site)
        align_leg_dist = T.cos_siml(leg_up, table_up)
        rew = (align_leg_dist - 1) * self._align_rot_dist_coef
        info = {"align_leg_dist": align_leg_dist, "align_leg_rew": rew}
        info["align_leg_succ"] = align_leg_dist > 0.85
        assert rew <= 0
        return rew, info

    def _lower_leg_reward(self) -> Tuple[float, dict]:
        """
        Move the leg conn site to a point above the table conn sit
        Negative euclidean distance between eef xy and conn xy.
        Return negative eucl distance
        """
        above_conn_pos = self._get_pos(self._current_table_site)
        above_conn_pos[2] += 0.05
        leg_conn_pos = self._get_pos(self._current_leg_site)
        dist = np.linalg.norm(above_conn_pos - leg_conn_pos)
        rew = -dist * self._pos_dist_coef
        info = {"lower_leg_dist": dist, "lower_leg_rew": rew}
        info["lower_leg_succ"] = dist < 0.03
        assert rew <= 0
        return rew, info

    def _align_leg_fine_reward(self) -> Tuple[float, dict]:
        """
        Finely align the leg over the connection site
        Up vectors of leg conn and table conn should be parallel (cos dist 1)
        Projection vectors should be parallel as well
        Return negative angular distance
        """
        leg_up = self._get_up_vector(self._current_leg_site)
        table_up = self._get_up_vector(self._current_table_site)
        align_leg_dist = T.cos_siml(leg_up, table_up)
        rew = (align_leg_dist - 1) * self._fine_align_rot_dist_coef
        info = {"align_leg_fine_dist": align_leg_dist, "align_leg_fine_rew": rew}

        leg_site_pos = self._get_pos(self._current_leg_site)
        table_site_pos = self._get_pos(self._current_table_site)
        proj_t = T.cos_siml(table_up, leg_site_pos - table_site_pos)
        proj_l = T.cos_siml(-leg_up, table_site_pos - leg_site_pos)
        proj_t_rew = (proj_t - 1) * self._fine_align_rot_dist_coef
        proj_l_rew = (proj_l - 1) * self._fine_align_rot_dist_coef
        info.update({"proj_t_rew": proj_t_rew, "proj_t": proj_t})
        info.update({"proj_l_rew": proj_l_rew, "proj_tl": proj_l})
        rew += proj_t_rew + proj_l_rew
        info["align_leg_fine_succ"] = (
            align_leg_dist > 0.95 and proj_t > 0.9 and proj_l > 0.9
        )
        assert rew <= 0
        return rew, info

    def _stable_grip_reward(self) -> Tuple[float, dict]:
        """
        Makes sure the eef and object axes are aligned
        Prioritize wrist alignment more than vertical alignment
        Returns negative angular distance
        """
        # up vector of leg and up vector of grip site should be perpendicular
        eef_up = self._get_up_vector("grip_site")
        leg_up = self._get_up_vector(self._current_leg_site)
        eef_leg_up_dist = T.cos_siml(eef_up, leg_up)
        eef_leg_up_rew = self._rot_dist_coef / 3 * -np.abs(eef_leg_up_dist)

        # up vector of leg and left vector of grip site should be parallel (close to -1 or 1)
        eef_left = self._get_left_vector("grip_site")
        eef_leg_left_dist = T.cos_siml(eef_left, leg_up)
        eef_leg_left_rew = (np.abs(eef_leg_left_dist) - 1) * self._rot_dist_coef
        info = {
            "eef_leg_up_dist": eef_leg_up_dist,
            "eef_leg_up_rew": eef_leg_up_rew,
            "eef_leg_left_dist": eef_leg_left_dist,
            "eef_leg_left_rew": eef_leg_left_rew,
        }
        assert eef_leg_up_rew <= 0 and eef_leg_left_rew <= 0
        # print(f"Close to 0; eef_leg_up_siml: {eef_leg_up_dist}")
        # print(f"Close to 1 or -1; eef_leg_left_dist: {eef_leg_left_dist}")
        rew = eef_leg_up_rew + eef_leg_left_rew
        info["stable_grip_succ"] = (
            np.abs(eef_leg_up_dist) < self._rot_threshold
            and np.abs(eef_leg_left_dist) > 1 - self._rot_threshold
        )
        return rew, info

    def _lower_leg_fine_reward(self) -> Tuple[float, dict]:
        """
        Move the leg conn site to table conn site
        Negative euclidean distance between eef xy and conn xy.
        Return negative eucl distance
        """
        above_conn_pos = self._get_pos(self._current_table_site)
        leg_conn_pos = self._get_pos(self._current_leg_site)
        dist = np.linalg.norm(above_conn_pos - leg_conn_pos)
        rew = -dist * self._pos_dist_coef
        info = {"lower_leg_fine_dist": dist, "lower_leg_fine_rew": rew}
        info["lower_leg_fine_succ"] = dist < 0.01
        assert rew <= 0
        return rew, info

    def _gripper_penalty(self, ac) -> Tuple[float, dict]:
        """
        Give penalty on status of gripper.

        Returns 0 if gripper is in desired position, range is [-2, 0]
        """
        grip_open = self._phases[self._phase_i] in self._grip_open_phases
        # ac[-2] is -1 for open, 1 for closed
        rew = (-1 - ac[-2] if grip_open else ac[-2] - 1) * self._gripper_penalty_coef
        assert rew <= 0
        info = {"gripper_penalty": rew}
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

    def get_bodyiterator(self, bodyname):
        for body in self.mujoco_objects[bodyname].root.find("worldbody"):
            if "name" in body.attrib and bodyname == body.attrib["name"]:
                return body.getiterator()
        return None

    def _get_groupname(self, bodyname):
        bodyiterator = self.get_bodyiterator(bodyname)
        for child in bodyiterator:
            if child.tag == "site":
                if "name" in child.attrib and "conn_site" in child.attrib["name"]:
                    return child.attrib["name"].split("-")[0]
        return None

    def get_connsites(self, gbody_name, tbody_name):
        gripbody_connsite, tbody_connsite = [], []
        group1 = self._get_groupname(gbody_name)
        group2 = self._get_groupname(tbody_name)
        iter1 = self.get_bodyiterator(gbody_name)
        iter2 = self.get_bodyiterator(tbody_name)
        griptag = group1 + "-" + group2
        tgttag = group2 + "-" + group1
        for child in iter1:
            if child.tag == "site":
                if (
                    "name" in child.attrib
                    and "conn_site" in child.attrib["name"]
                    and griptag in child.attrib["name"]
                    and child.attrib["name"] not in self._used_connsites
                ):
                    gripbody_connsite.append(child.attrib["name"])
        for child in iter2:
            if child.tag == "site":
                if (
                    "name" in child.attrib
                    and "conn_site" in child.attrib["name"]
                    and tgttag in child.attrib["name"]
                    and child.attrib["name"] not in self._used_connsites
                ):
                    tbody_connsite.append(child.attrib["name"])
        return gripbody_connsite, tbody_connsite

    def get_furthest_connsite(self, conn_sites, gripper_pos):
        furthest = None
        max_dist = None
        for name in conn_sites:
            pos = self.sim.data.get_site_xpos(name)
            dist = T.l2_dist(gripper_pos, pos)
            if furthest is None:
                furthest = name
                max_dist = dist
            else:
                if dist > max_dist:
                    furthest = name
                    max_dist = dist
        return furthest

    def get_closest_connsite(self, conn_sites, gripper_pos):
        closest = None
        min_dist = None
        for name in conn_sites:
            pos = self.sim.data.get_site_xpos(name)
            dist = T.l2_dist(gripper_pos, pos)
            if closest is None:
                closest = name
                min_dist = dist
            else:
                if dist < min_dist:
                    closest = name
                    min_dist = dist
        return closest

    def _get_gripper_pos(self) -> list:
        """return 6d pos [griptip, grip] """
        return np.concatenate(
            [self._get_pos("griptip_site"), self._get_pos("grip_site")]
        )

    def _get_fingertip_pos(self) -> list:
        """return 6d pos [left grip, right grip]"""
        return np.concatenate(
            [self._get_pos("lgriptip_site"), self._get_pos("rgriptip_site")]
        )


def main():
    from config import create_parser

    parser = create_parser(env="furniture-sawyer-tablelack-v0")
    config, unparsed = parser.parse_known_args()
    if len(unparsed):
        logger.error("Unparsed argument is detected:\n%s", unparsed)
        return

    # create an environment and run manual control of Sawyer environment
    env = FurnitureSawyerTableLackEnv(config)
    env.run_manual(config)


if __name__ == "__main__":
    main()
