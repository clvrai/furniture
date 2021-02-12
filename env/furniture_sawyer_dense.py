from typing import Tuple

import numpy as np
import yaml
import os

import env.transform_utils as T
from env.furniture_sawyer import FurnitureSawyerEnv
from env.models import furniture_name2id
from util import PrettySafeLoader
from util.logger import logger


class FurnitureSawyerDenseRewardEnv(FurnitureSawyerEnv):
    """
    Sawyer environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.furniture_id = furniture_name2id[config.furniture_name]
        super().__init__(config)

        self._diff_rew = config.diff_rew
        self._phase_bonus = config.phase_bonus
        self._ctrl_penalty_coef = config.ctrl_penalty_coef
        self._pos_threshold = config.pos_threshold
        self._rot_threshold = config.rot_threshold
        self._eef_rot_dist_coef = config.eef_rot_dist_coef
        self._eef_up_rot_dist_coef = config.eef_up_rot_dist_coef
        self._eef_pos_dist_coef = config.eef_pos_dist_coef
        self._rot_dist_coef = config.rot_dist_coef
        self._pos_dist_coef = config.pos_dist_coef
        self._lift_dist_coef = config.lift_dist_coef
        self._grasp_dist_coef = config.grasp_dist_coef
        self._gripper_penalty_coef = config.gripper_penalty_coef
        self._align_pos_dist_coef = config.align_pos_dist_coef
        self._align_rot_dist_coef = config.align_rot_dist_coef
        self._fine_align_rot_dist_coef = config.fine_align_rot_dist_coef
        self._fine_align_pos_dist_coef = config.fine_align_pos_dist_coef
        self._touch_coef = config.touch_coef

        self._num_connect_steps = 0
        self._discrete_grip = config.discrete_grip
        self._grip_open_phases = set(["move_eef_above_leg", "lower_eef_to_leg"])
        self._phases = [
            "move_eef_above_leg",
            "lower_eef_to_leg",
            "grasp_leg",
            "lift_leg",
            "move_leg",
            "move_leg_fine",
        ]

    def _reset_reward_variables(self):
        self._subtask_step = len(self._preassembled)
        self._update_reward_variables(self._subtask_step)

    def _set_next_subtask(self) -> bool:
        """Returns True if we are done with all attaching steps"""
        self._subtask_step += 1
        if self._subtask_step == self._success_num_conn:
            return True
        self._update_reward_variables(self._subtask_step)
        return False

    def _update_reward_variables(self, subtask_step):
        """Updates the reward variables wrt subtask step"""
        self._phase_i = 0
        self._leg, self._table = self._recipe["recipe"][subtask_step]
        self._leg_site, self._table_site = self._site_recipe[subtask_step][:2]
        if len(self._site_recipe[subtask_step]) == 3:
            self._table_leg_angle = self._site_recipe[subtask_step][2]
        else:
            self._table_leg_angle = None
        # update the observation to the current objects of interest
        self._subtask_part1 = self._object_name2id[self._leg]
        self._subtask_part2 = self._object_name2id[self._table]
        self._leg_touched = False
        self._leg_lift = False
        self._init_leg_pos = self._get_pos(self._leg)
        self._leg_fine_aligned = False

        g1, g2 = f"{self._leg}_ltgt_site0", f"{self._leg}_rtgt_site0"
        if "grip_site_recipe" in self._recipe and self._subtask_step < len(
            self._recipe["grip_site_recipe"]
        ):
            g1, g2 = self._recipe["grip_site_recipe"][self._subtask_step]
        self._get_leg_grasp_pos = lambda x: (self._get_pos(g1) + self._get_pos(g2)) / 2
        self._get_leg_grasp_vector = lambda x: self._get_pos(g2) - self._get_pos(g1)

        if self._diff_rew:
            eef_pos = self._get_pos("griptip_site")
            leg_pos = self._get_leg_grasp_pos(self._leg)  # + [0, 0, 0.07]
            leg_pos[2] = 0.08
            xy_distance = np.linalg.norm(eef_pos[:2] - leg_pos[:2])
            z_distance = np.abs(eef_pos[2] - leg_pos[2])
            self._prev_eef_above_leg_distance = xy_distance + z_distance

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

        ob, reward, done, info = super()._step(a)
        return ob, reward, done, info

    def _compute_reward(self, ac) -> Tuple[float, bool, dict]:
        """
        Multistage reward.
        While moving the leg, we need to make sure the grip is stable by measuring
        angular movements.
        At any point, the robot should minimize pose displacement in non-relevant parts.

        Phases:
        - move_eef_over_leg: move eef over table leg
        - lower_eef_to_leg: lower eef onto the leg
        - lift_leg: grip and lift the leg
        - move_eef_over_conn: move the eef (holding leg) above the conn site
        - align_leg: coarsely align the leg with the conn site
        - lower_leg: move the leg 0.05 cm above the conn site
        - align_leg_fine: fine grain alignment of the up and forward vectors
        - lower_leg_fine: finely move the leg onto the conn site
        """
        phase_bonus = reward = 0
        _, _, info = super()._compute_reward(ac)

        # clear the original success and done
        done = False
        self._success = False

        ctrl_penalty, ctrl_info = self._ctrl_penalty(ac)
        stable_grip_rew, sg_info = self._stable_grip_reward()

        left, right = self._finger_contact(self._leg)
        leg_touched = int(left and right)

        # detect early picking
        if leg_touched and sg_info["stable_grip_succ"] and self._phase_i < 2:
            phase_bonus += self._phase_bonus * (2 - self._phase_i)
            self._phase_i = 2
            eef_pos = self._get_gripper_pos()
            leg_pos = self._get_leg_grasp_pos(self._leg) + [0, 0, -0.015]
            xy_distance = np.linalg.norm(eef_pos[:2] - leg_pos[:2])
            z_distance = np.abs(eef_pos[2] - leg_pos[2])
            self._prev_eef_leg_distance = xy_distance + z_distance
            self._prev_grasp_leg_rew = ac[-2]

        phase = self._phases[self._phase_i]
        info["phase_i"] = self._phase_i + len(self._phases) * self._subtask_step
        info["subtask"] = self._subtask_step

        grip_penalty, grip_info = self._gripper_penalty(ac)

        # impose stable_grip_rew until lifting
        if self._phase_i > 3:
            stable_grip_rew = 0

        if phase == "move_eef_above_leg":
            phase_reward, phase_info = self._move_eef_above_leg_reward()
            if phase_info[f"{phase}_succ"] and sg_info["stable_grip_succ"]:
                self._phase_i += 1
                phase_bonus += self._phase_bonus
                eef_pos = self._get_gripper_pos()
                leg_pos = self._get_leg_grasp_pos(self._leg) + [0, 0, -0.015]
                xy_distance = np.linalg.norm(eef_pos[:2] - leg_pos[:2])
                z_distance = np.abs(eef_pos[2] - leg_pos[2])
                self._prev_eef_leg_distance = xy_distance + z_distance

        elif phase == "lower_eef_to_leg":
            phase_reward, phase_info = self._lower_eef_to_leg_reward()
            if phase_info[f"{phase}_succ"] and sg_info["stable_grip_succ"]:
                phase_bonus += self._phase_bonus
                self._phase_i += 1
                self._prev_grasp_leg_rew = ac[-2]

        elif phase == "grasp_leg":
            phase_reward, phase_info = self._grasp_leg_reward(ac)
            if phase_info["grasp_leg_succ"] and sg_info["stable_grip_succ"]:
                self._phase_i += 1
                phase_bonus += self._phase_bonus
                leg_pos = self._get_pos(self._leg_site)
                self._init_lift_leg_pos = leg_pos
                self._lift_leg_pos = leg_pos + [0, 0, 0.2]
                xy_distance = np.linalg.norm(self._lift_leg_pos[:2] - leg_pos[:2])
                z_distance = np.abs(self._lift_leg_pos[2] - leg_pos[2])
                self._prev_lift_leg_z_distance = z_distance
                self._prev_lift_leg_xy_distance = xy_distance

        elif phase == "lift_leg":
            phase_reward, phase_info = self._lift_leg_reward()

            if not phase_info["touch"]:
                print("Dropped leg")
                done = True
                phase_bonus += -self._phase_bonus / 2

            elif phase_info[f"lift_leg_succ"]:
                print(f"DONE WITH PHASE {phase}")
                self._phase_i += 1
                phase_bonus += self._phase_bonus
                above_table_site = self._get_pos(self._table_site) + [
                    0,
                    0,
                    self._recipe["z_finedist"],
                ]
                leg_site = self._get_pos(self._leg_site)
                self._prev_move_pos_distance = np.linalg.norm(
                    above_table_site - leg_site
                )
                leg_up = self._get_up_vector(self._leg_site)
                table_up = self._get_up_vector(self._table_site)
                self._prev_move_ang_dist = T.cos_siml(leg_up, table_up)

                leg_forward = self._get_forward_vector(self._leg_site)
                target_forward = self._project_connector_forward(
                    self._table_site, self._leg_site, self._table_leg_angle
                )
                self._prev_move_forward_ang_dist = T.cos_siml(
                    leg_forward, target_forward
                )

        elif phase == "move_leg":
            phase_reward, phase_info = self._move_leg_reward()
            if not phase_info["touch"] or not phase_info["lift"]:
                print("Dropped leg")
                done = True
                phase_bonus += -self._phase_bonus / 2

            if phase_info[f"{phase}_succ"]:
                print(f"DONE WITH PHASE {phase}")
                self._phase_i += 1
                phase_bonus += self._phase_bonus
                table_site = self._get_pos(self._table_site)
                leg_site = self._get_pos(self._leg_site)
                self._prev_move_pos_distance = np.linalg.norm(table_site - leg_site)

                leg_up = self._get_up_vector(self._leg_site)
                table_up = self._get_up_vector(self._table_site)
                self._prev_move_ang_dist = T.cos_siml(leg_up, table_up)

                leg_forward = self._get_forward_vector(self._leg_site)
                target_forward = self._project_connector_forward(
                    self._table_site, self._leg_site, self._table_leg_angle
                )
                self._prev_move_forward_ang_dist = T.cos_siml(
                    leg_forward, target_forward
                )

                self._prev_proj_t = T.cos_siml(-table_up, leg_site - table_site)
                self._prev_proj_l = T.cos_siml(leg_up, table_site - leg_site)

        elif phase == "move_leg_fine":
            phase_reward, phase_info = self._move_leg_fine_reward(ac)

            if phase_info["connect_succ"]:
                phase_bonus += self._phase_bonus * 2
                self._phase_i = 0
                print(f"CONNECTED!!!!!!!!!!!!!!!!!!!!!!")
                # update reward variables for next attachment
                self._success = self._set_next_subtask()
                if self._success:
                    done = True

            elif not phase_info["touch"]:
                print("Dropped leg")
                done = True
                phase_bonus += -self._phase_bonus / 2

        else:
            phase_reward, phase_info = 0, {}
            done = True

        reward += ctrl_penalty + phase_reward + stable_grip_rew
        reward += grip_penalty + phase_bonus
        info["phase_bonus"] = phase_bonus
        info = {**info, **ctrl_info, **phase_info, **sg_info, **grip_info}
        # log phase if last frame
        if self._episode_length == self._env_config["max_episode_steps"] - 1 or done:
            info["phase"] = self._phase_i + len(self._phases) * self._subtask_step
        return reward, done, info

    def _move_eef_above_leg_reward(self) -> Tuple[float, dict]:
        """
        Moves the eef above the leg and rotates the wrist.
        Negative euclidean distance between eef xy and leg xy.

        Return negative eucl distance
        """
        eef_pos = self._get_pos("griptip_site")
        leg_pos = self._get_leg_grasp_pos(self._leg)  # + [0, 0, 0.07]
        leg_pos[2] = 0.08
        xy_distance = np.linalg.norm(eef_pos[:2] - leg_pos[:2])
        z_distance = np.abs(eef_pos[2] - leg_pos[2])
        eef_above_leg_distance = xy_distance + z_distance
        if self._diff_rew:
            offset = self._prev_eef_above_leg_distance - eef_above_leg_distance
            rew = offset * self._eef_pos_dist_coef * 10
            self._prev_eef_above_leg_distance = eef_above_leg_distance
        else:
            rew = -eef_above_leg_distance * self._eef_pos_dist_coef
        info = {
            "eef_above_leg_dist": eef_above_leg_distance,
            "eef_above_leg_rew": rew,
            "move_eef_above_leg_succ": int(xy_distance < 0.02 and z_distance < 0.02),
        }
        return rew, info

    def _lower_eef_to_leg_reward(self) -> Tuple[float, dict]:
        """
        Moves the eef over the leg and rotates the wrist.
        Negative euclidean distance between eef xy and leg xy.
        Give additional reward for contacting the leg
        Return negative eucl distance
        """
        info = {}
        eef_pos = self._get_gripper_pos()
        leg_pos = self._get_leg_grasp_pos(self._leg) + [0, 0, -0.015]
        xy_distance = np.linalg.norm(eef_pos[:2] - leg_pos[:2])
        z_distance = np.abs(eef_pos[2] - leg_pos[2])
        eef_leg_distance = xy_distance + z_distance
        if self._diff_rew:
            offset = self._prev_eef_leg_distance - eef_leg_distance
            rew = offset * self._eef_pos_dist_coef * 10
            self._prev_eef_leg_distance = eef_leg_distance
        else:
            rew = -eef_leg_distance * self._pos_dist_coef

        rew *= 5
        info.update({"eef_leg_dist": eef_leg_distance, "eef_leg_rew": rew})
        info["lower_eef_to_leg_succ"] = int(xy_distance < 0.02 and z_distance < 0.01)
        return rew, info

    def _grasp_leg_reward(self, ac) -> Tuple[float, dict]:
        """
        Grasp the leg, making sure it is in position
        """
        rew, info = self._lower_eef_to_leg_reward()
        # if eef in correct position, add additional grasping success

        left, right = self._finger_contact(self._leg)
        leg_touched = int(left and right)
        info["grasp_leg_succ"] = leg_touched

        # closed gripper is 1, want to maximize gripper
        offset = ac[-2] - self._prev_grasp_leg_rew
        grasp_leg_rew = offset * self._grasp_dist_coef
        self._prev_grasp_leg_rew = ac[-2]
        info["grasp_leg_rew"] = grasp_leg_rew

        touch_rew = 0
        # touch_rew = leg_touched * self._touch_coef * 10

        # gripper rew, 1 if closed
        # further bonus for touch
        if leg_touched and not self._leg_touched:
            touch_rew += self._touch_coef * 10
            self._leg_touched = True
        info.update({"touch": leg_touched, "touch_rew": touch_rew})
        rew += grasp_leg_rew + touch_rew

        return rew, info

    def _lift_leg_reward(self) -> Tuple[float, dict]:
        """
        Lift the leg
        """
        left, right = self._finger_contact(self._leg)
        leg_touched = int(left and right)

        # reward for grasping
        # touch_rew = leg_touched * self._touch_coef
        touch_rew = 0

        # reward for lifting
        leg_pos = self._get_pos(self._leg_site)
        xy_distance = np.linalg.norm(self._lift_leg_pos[:2] - leg_pos[:2])
        z_distance = np.abs(self._lift_leg_pos[2] - leg_pos[2])
        if self._diff_rew:
            z_offset = self._prev_lift_leg_z_distance - z_distance
            lift_leg_rew = z_offset * self._lift_dist_coef * 10
            self._prev_lift_leg_z_distance = z_distance
            xy_offset = self._prev_lift_leg_xy_distance - xy_distance
            lift_leg_rew += xy_offset * self._lift_dist_coef * 10
            self._prev_lift_leg_xy_distance = xy_distance
        else:
            lift_leg_rew = -(z_distance + xy_distance) * self._lift_dist_coef

        # give one time reward for lifting the leg
        leg_lift = leg_pos[2] > (self._init_lift_leg_pos[2] + 0.01)
        if (leg_touched and leg_lift) and not self._leg_lift:
            print("Lift leg")
            self._leg_lift = True
            lift_leg_rew += self._phase_bonus / 10

        rew = lift_leg_rew + touch_rew
        info = {"touch": leg_touched, "touch_rew": touch_rew, "lift": int(leg_lift)}
        info["lift_leg_rew"] = lift_leg_rew
        info["lift_leg_xy_dist"] = xy_distance
        info["lift_leg_z_dist"] = z_distance
        info["lift_leg_succ"] = int(xy_distance < 0.03 and z_distance < 0.01)
        info["lift_leg_pos"] = leg_pos
        info["lift_leg_target"] = self._lift_leg_pos

        return rew, info

    def _move_leg_reward(self) -> Tuple[float, dict]:
        """
        Coarsely move the leg site over the conn_site
        Also give reward for angular alignment
        """
        left, right = self._finger_contact(self._leg)
        leg_touched = int(left and right)
        # touch_rew = leg_touched * self._touch_coef
        touch_rew = 0
        info = {"touch": leg_touched, "touch_rew": touch_rew}

        # calculate position rew
        above_table_site = self._get_pos(self._table_site) + [
            0,
            0,
            self._recipe["z_finedist"],
        ]
        leg_site = self._get_pos(self._leg_site)
        move_pos_distance = np.linalg.norm(above_table_site - leg_site)
        if self._diff_rew:
            offset = self._prev_move_pos_distance - move_pos_distance
            pos_rew = offset * self._align_pos_dist_coef * 10
            self._prev_move_pos_distance = move_pos_distance
        else:
            pos_rew = -move_pos_distance * self._align_pos_dist_coef
        info.update({"move_pos_dist": move_pos_distance, "move_pos_rew": pos_rew})

        # calculate angular rew
        leg_up = self._get_up_vector(self._leg_site)
        table_up = self._get_up_vector(self._table_site)
        move_ang_dist = T.cos_siml(leg_up, table_up)
        if self._diff_rew:
            offset = move_ang_dist - self._prev_move_ang_dist
            ang_rew = offset * self._align_rot_dist_coef * 10
            self._prev_move_ang_dist = move_ang_dist
        else:
            ang_rew = (move_ang_dist - 1) * self._align_rot_dist_coef
        info.update({"move_ang_dist": move_ang_dist, "move_ang_rew": ang_rew})

        leg_forward = self._get_forward_vector(self._leg_site)
        target_forward = self._project_connector_forward(
            self._table_site, self._leg_site, self._table_leg_angle
        )
        move_forward_ang_dist = T.cos_siml(leg_forward, target_forward)
        if self._diff_rew:
            offset = move_forward_ang_dist - self._prev_move_forward_ang_dist
            forward_ang_rew = offset * self._align_rot_dist_coef * 10
            self._prev_move_forward_ang_dist = move_forward_ang_dist
        else:
            forward_ang_rew = (move_forward_ang_dist - 1) * self._align_rot_dist_coef
        info["move_forward_ang_dist"] = move_forward_ang_dist
        info["move_forward_ang_rew"] = forward_ang_rew

        info["move_leg_succ"] = int(
            move_pos_distance < 0.06
            and move_ang_dist > 0.85
            and move_forward_ang_dist > 0.85
            and leg_touched
        )

        rew = pos_rew + ang_rew + forward_ang_rew + touch_rew

        leg_lift = leg_site[2] > (self._init_lift_leg_pos[2] + 0.002)
        info["lift"] = leg_lift

        return rew, info

    def _move_leg_fine_reward(self, ac) -> Tuple[float, dict]:
        """
        Finely move the leg site over the conn_site
        Also give reward for angular alignment
        Also check for connected pieces
        """
        # no dense reward when completed
        info = {"connect_succ": self._connected}
        if self._connected:
            return 0, info

        left, right = self._finger_contact(self._leg)
        leg_touched = int(left and right)
        # touch_rew = leg_touched * self._touch_coef
        touch_rew = 0
        info.update({"touch": leg_touched, "touch_rew": touch_rew})

        # calculate position rew
        table_site = self._get_pos(self._table_site)
        leg_site = self._get_pos(self._leg_site)
        move_pos_distance = np.linalg.norm(table_site - leg_site)

        if self._diff_rew:
            f = lambda x: np.exp(-25 * x)
            offset = f(move_pos_distance) - f(self._prev_move_pos_distance)
            # offset = self._prev_move_pos_distance - move_pos_distance
            pos_rew = offset * self._fine_align_pos_dist_coef * 10
            self._prev_move_pos_distance = move_pos_distance
        else:
            pos_rew = -move_pos_distance * self._fine_align_pos_dist_coef
        info.update(
            {"move_fine_pos_dist": move_pos_distance, "move_fine_pos_rew": pos_rew}
        )

        # calculate angular rew
        leg_up = self._get_up_vector(self._leg_site)
        table_up = self._get_up_vector(self._table_site)
        move_ang_dist = T.cos_siml(leg_up, table_up)
        if self._diff_rew:
            f = lambda x: np.exp(-3 * (1 - x))
            offset = f(move_ang_dist) - f(self._prev_move_ang_dist)
            # offset = move_ang_dist - self._prev_move_ang_dist
            ang_rew = offset * self._fine_align_rot_dist_coef * 10
            self._prev_move_ang_dist = move_ang_dist
        else:
            ang_rew = (move_ang_dist - 1) * self._fine_align_rot_dist_coef
        info["move_fine_ang_dist"] = move_ang_dist
        info["move_fine_ang_rew"] = ang_rew

        leg_forward = self._get_forward_vector(self._leg_site)
        target_forward = self._project_connector_forward(
            self._table_site, self._leg_site, self._table_leg_angle
        )
        move_forward_ang_dist = T.cos_siml(leg_forward, target_forward)
        if self._diff_rew:
            f = lambda x: np.exp(-3 * (1 - x))
            offset = f(move_forward_ang_dist) - f(self._prev_move_forward_ang_dist)
            # offset = move_forward_ang_dist - self._prev_move_forward_ang_dist
            forward_ang_rew = offset * self._fine_align_rot_dist_coef * 10
            self._prev_move_forward_ang_dist = move_forward_ang_dist
        else:
            forward_ang_rew = (
                move_forward_ang_dist - 1
            ) * self._fine_align_rot_dist_coef
        info["move_fine_forward_ang_dist"] = move_forward_ang_dist
        info["move_fine_forward_ang_rew"] = forward_ang_rew

        # proj will approach 1 if aligned correctly
        proj_t = T.cos_siml(-table_up, leg_site - table_site)
        proj_l = T.cos_siml(leg_up, table_site - leg_site)
        if self._diff_rew:
            f = lambda x: np.exp(-3 * (1 - x))
            offset = f(proj_t) - f(self._prev_proj_t)
            proj_t_rew = offset * self._fine_align_rot_dist_coef * 10
            self._prev_proj_t = proj_t
            offset = f(proj_l) - f(self._prev_proj_l)
            proj_l_rew = offset * self._fine_align_rot_dist_coef * 10
            self._prev_proj_l = proj_l
        else:
            proj_t_rew = (proj_t - 1) * self._fine_align_rot_dist_coef / 10
            proj_l_rew = (proj_l - 1) * self._fine_align_rot_dist_coef / 10
        info.update({"proj_t_rew": proj_t_rew, "proj_t": proj_t})
        info.update({"proj_l_rew": proj_l_rew, "proj_l": proj_l})
        info["move_leg_fine_succ"] = int(
            self._is_aligned(self._leg_site, self._table_site)
        )
        rew = pos_rew + ang_rew + forward_ang_rew + touch_rew + proj_t_rew + proj_l_rew

        if info["move_leg_fine_succ"]:  # and not self._leg_fine_aligned:
            self._leg_fine_aligned = True
            info["connect_rew"] = ac[-1] * 100
            rew += info["connect_rew"]
        return rew, info

    def _stable_grip_reward(self) -> Tuple[float, dict]:
        """
        Makes sure the eef and object axes are aligned
        Prioritize wrist alignment more than vertical alignment
        Returns negative angular distance
        """
        # up vector of leg and world up vector should be aligned
        eef_up = self._get_up_vector("grip_site")
        eef_up_grasp_dist = T.cos_siml(eef_up, [0, 0, -1])
        eef_up_grasp_rew = self._eef_up_rot_dist_coef * (eef_up_grasp_dist - 1)

        grasp_vec = self._get_leg_grasp_vector(self._leg_site)
        # up vector of leg and forward vector of grip site should be parallel (close to -1 or 1)
        eef_forward = self._get_forward_vector("grip_site")
        eef_forward_grasp_dist = T.cos_siml(eef_forward, grasp_vec)
        eef_forward_grasp_rew = (
            np.abs(eef_forward_grasp_dist) - 1
        ) * self._eef_rot_dist_coef
        info = {
            "eef_up_grasp_dist": eef_up_grasp_dist,
            "eef_up_grasp_rew": eef_up_grasp_rew,
            "eef_forward_grasp_dist": eef_forward_grasp_dist,
            "eef_forward_grasp_rew": eef_forward_grasp_rew,
        }
        # print(f"Close to 1; eef_up_grasp_siml: {eef_up_grasp_dist}")
        # print(f"Close to 1/-1; eef_forward_grasp_dist: {eef_forward_grasp_dist}")
        rew = eef_up_grasp_rew + eef_forward_grasp_rew
        info["stable_grip_succ"] = int(
            eef_up_grasp_dist > 1 - self._rot_threshold
            and np.abs(eef_forward_grasp_dist) > 1 - self._rot_threshold
        )
        return rew, info

    def _gripper_penalty(self, ac) -> Tuple[float, dict]:
        """
        Give penalty on status of gripper. Only give it on phases where
        gripper should close
        Returns 0 if gripper is in desired position, range is [-2, 0]
        """
        if self._discrete_grip:
            ac = ac.copy()
            ac[-2] = -1 if ac[-2] < 0 else 1
        grip_open = self._phases[self._phase_i] in self._grip_open_phases
        # ac[-2] is -1 for open, 1 for closed
        rew = 0
        if not grip_open:
            rew = (
                -1 - ac[-2] if grip_open else ac[-2] - 1
            ) * self._gripper_penalty_coef
        assert rew <= 0
        info = {"gripper_penalty": rew, "gripper_action": ac[-2]}
        return rew, info

    def _ctrl_penalty(self, action) -> Tuple[float, dict]:
        rew = np.linalg.norm(action[:-2]) * -self._ctrl_penalty_coef
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

    parser = create_parser(env="furniture-sawyer-densereward-v0")
    parser.add_argument(
        "--run_mode", type=str, default="manual", choices=["manual", "vr", "demo"]
    )
    config, unparsed = parser.parse_known_args()
    if len(unparsed):
        logger.error("Unparsed argument is detected:\n%s", unparsed)
        return

    # create an environment and run manual control of Sawyer environment
    env = FurnitureSawyerDenseRewardEnv(config)
    if config.run_mode == "manual":
        env.run_manual(config)
    elif config.run_mode == "vr":
        env.run_vr(config)
    elif config.run_mode == "demo":
        env.run_demo_actions(config)


if __name__ == "__main__":
    main()
