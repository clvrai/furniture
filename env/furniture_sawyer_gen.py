import math
import time
import yaml

import numpy as np
from tqdm import tqdm

import env.transform_utils as T
from env.furniture_sawyer import FurnitureSawyerEnv
from env.models import background_names, furniture_name2id, furniture_xmls
from util import PrettySafeLoader
from util.logger import logger


class FurnitureSawyerGenEnv(FurnitureSawyerEnv):
    """
    Sawyer environment for assemblying furniture programmatically
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.

        Abbreviations:
            grip ~ gripper
            g ~ gripped body
            t ~ target body
            conn ~ connection

        Phases Descrption:
            1. xy_move_g:
                    move to xy-pos of gripper with gbody
            2. align_g:
                    rotate gripper fingers vector to gbody_gripsites vector (xy plane only),
                    rotate up-vector of gripper to down vector (0,0,-1)
            3. z_move_g:
                    move gripper down to z-pos of gbody
                    *has special check to ensure gripper doesn't repeatedly hit ground/table
            4. move_grip_safepos:
                    grip gbody then move up to the grip_safepos
            5. xy_move_t:
                    move gripper to tbody xy-pos
            6. align_conn:
                    rotate gbody conn_site up-vector to tbody conn_site up-vector
            7. xy_move_conn:
                    move to xy-pos of gbody conn_site w.r.t. tbody conn_site
            8. z_move_conn:
                    move to xyz-pos of gbody conn_site w.r.t. tbody conn_site
            9. align_conn_fine:
                    finely rotate up-vector of gbody conn_site to up-vector of tbody conn_site
            10. z_move_conn_fine:
                    finely move to xyz position of gbody conn_site w.r.t. tbody conn_site,
                    then try connecting
            11. move_nogrip_safepos:
                    release gripper and move up to nogrip_safepos
            12  part_done:
                    set part_done = True, and part is connected
        """
        config.record_demo = True
        super().__init__(config)

        self._phase = None
        self._num_connected_prev = 0
        self._part_success = False
        self._phases = [
            "xy_move_g",
            "align_g",
            "z_move_g",
            "move_grip_safepos",
            "xy_move_t",
            "align_conn",
            "xy_move_conn",
            "z_move_conn",
            "align_conn_fine",
            "z_move_conn_fine",
            "move_nogrip_safepos",
            "part_done",
        ]

        self._phase_noise = {
            #   phase      : (min_val, max_val, dimensions)
            "xy_move_g": (0, 0, 2),
            "xy_move_t": (-self._config.furn_xyz_rand, self._config.furn_xyz_rand, 2),
            "move_grip_safepos": (0, 2 * self._config.furn_xyz_rand, 3),
            "move_nogrip_safepos": (0, 2 * self._config.furn_xyz_rand, 3),
        }
        self.reset()

    def get_random_noise(self, noise):
        for phase, val in self._phase_noise.items():
            minimum, maximum, size = val
            noise[phase] = self._rng.uniform(low=minimum, high=maximum, size=size)

    def _norm_rot_action(self, action, cap=1):
        if "fine" in self._phase:
            for a in range(3, 7):
                if 0 < abs(action[a]) < self.min_rot_act_fine:
                    action[a] *= self.min_rot_act_fine / abs(action[a])
        else:
            for a in range(3, 7):
                if 0 < abs(action[a]) < self.min_rot_act:
                    action[a] *= self.min_rot_act / abs(action[a])
        return action

    def _cap_action(self, action, cap=1):
        return np.clip(action, -cap, cap)

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        ob, reward, done, info = super(FurnitureSawyerEnv, self)._step(a)
        if self._num_connected > self._num_connected_prev:
            self._part_success = True
            self._num_connected_prev = self._num_connected

        if (
            self._num_connected == self._success_num_conn
            and len(self._object_names) > 1
        ):
            self._success = True
            done = True
        return ob, reward, done, info

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

    def get_conn_sites(self, gbody_name, tbody_name):
        gripbody_conn_site, tbody_conn_site = [], []
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
                    and child.attrib["name"] not in self._used_sites
                ):
                    gripbody_conn_site.append(child.attrib["name"])
        for child in iter2:
            if child.tag == "site":
                if (
                    "name" in child.attrib
                    and "conn_site" in child.attrib["name"]
                    and tgttag in child.attrib["name"]
                    and child.attrib["name"] not in self._used_sites
                ):
                    tbody_conn_site.append(child.attrib["name"])
        return gripbody_conn_site, tbody_conn_site

    def get_furthest_conn_site(self, conn_sites, gripper_pos):
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

    def get_closest_conn_site(self, conn_sites, gripper_pos):
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

    def align_gripsites(self, gripvec, gbodyvec, epsilon):
        if T.angle_between(-gripvec, gbodyvec) < T.angle_between(gripvec, gbodyvec):
            gripvec = -gripvec
        xyaction = T.angle_between2D(gripvec, gbodyvec)
        if abs(xyaction) < epsilon:
            xyaction = 0
        return xyaction

    def get_closest_xy_fwd(self, allowed_angles, gconn, tconn):
        # return tconn forward vector with most similar xy-plane angle to gconn vector
        if len(allowed_angles) == 0:
            # no need for xy-alignment, all angles are acceptable
            return self._get_forward_vector(gconn)[0:2]
        # get closest forward vector
        gfwd = self._get_forward_vector(gconn)[0:2]
        tfwd = self._get_forward_vector(tconn)[0:2]
        min_angle = min(
            abs(T.angle_between2D(gfwd, tfwd)),
            abs((2 * np.pi) + T.angle_between2D(gfwd, tfwd)),
        )
        min_all_angle = 0
        min_tfwd = tfwd
        for angle in allowed_angles:
            tfwd_rotated = T.rotate_vector2D(tfwd, angle * (np.pi / 180))
            xy_angle = T.angle_between2D(gfwd, tfwd_rotated)
            if np.pi <= xy_angle < 2 * np.pi:
                xy_angle = 2 * np.pi - xy_angle
            elif -(2 * np.pi) <= xy_angle < -np.pi:
                xy_angle = 2 * np.pi + xy_angle
            if abs(xy_angle) < min_angle:
                min_angle = abs(xy_angle)
                min_tfwd = tfwd_rotated
                min_all_angle = angle
        return min_tfwd

    def align2D(self, vec, targetvec, epsilon):
        """
        Returns a scalar corresponding to a rotation action in a single 2D-dimension
        Gives a rotation action to align vec with targetvec
        epsilon ~ threshold at which to set action=0
        """
        if abs(vec[0]) + abs(vec[1]) < 0.5:
            # unlikely current orientation allows for helpful rotation action due to gimbal lock
            return 0
        angle = T.angle_between2D(vec, targetvec)
        # move in direction that gets closer to closest of (-2pi, 0, or 2pi)
        if -(2 * np.pi) < angle <= -np.pi:
            action = -(2 * np.pi + angle)
        if -np.pi < angle <= 0:
            action = -angle
        if 0 < angle <= np.pi:
            action = -angle
        if np.pi < angle <= 2 * np.pi:
            action = 2 * np.pi - angle
        if abs(action) < epsilon:
            action = 0
        return action

    def move_xy(self, cur_pos, target_pos, epsilon, noise=None):
        """
        Returns a vector corresponding to action[0:2] (xy action)
        Move from current position to target position in xy dimensions
        epsilon ~ threshold at which to set action=0
        """
        d = target_pos - cur_pos
        if noise is not None:
            d += noise
        if abs(d[0]) > epsilon or abs(d[1]) > epsilon:
            if abs(d[0]) < epsilon:
                d[0] = 0
            if abs(d[1]) < epsilon:
                d[1] = 0
        else:
            self._phase_num += 1
            d[0:2] = 0
        return d

    def move_xyz(self, cur_pos, target_pos, epsilon, noise=None):
        """
        Returns a vector corresponding to action[0:3] (xyz action)
        Move from current position to target position in xyz dimensions
        epsilon ~ threshold at which to set action=0
        """
        d = target_pos - cur_pos
        if noise is not None:
            d += noise
        if abs(d[0]) > epsilon or abs(d[1]) > epsilon or abs(d[2]) > epsilon:
            if abs(d[0]) < epsilon:
                d[0] = 0
            if abs(d[1]) < epsilon:
                d[1] = 0
            if abs(d[2]) < epsilon:
                d[2] = 0
        else:
            d[0:3] = 0
        return d

    def move_z(self, cur_pos, target_pos, epsilon, conn_dist, noise=None, fine=None):
        """
        Returns a vector corresponding to action[0:3] (xyz action)
        Move from current position to target position in xyz dimensions
        epsilon ~ threshold at which to set action=0
        conn_dist ~ a scalar YAML configurable variable to make connection easier/harder
            closer to 0 -> harder, more than 0 -> easier
        fine ~ a YAML configurable variable to reduce scale of movement,
            useful for phase 'z_move_conn_fine' to get consistent connection behavior

        """
        d = target_pos - cur_pos
        if noise is not None:
            d += noise
        if abs(d[0]) < epsilon:
            d[0] = 0
        if abs(d[1]) < epsilon:
            d[1] = 0
        if abs(d[2]) < epsilon:
            self._phase_num += 1
            d[0:3] = 0
        elif fine:
            d /= fine
        return d

    def generate_demos(self, n_demos):
        """
        Issues:
            1. Only downward gripping works
            2. Once any collision occurs, unlikely to recover
            3. fine adjustment phase sometimes very challenging
        """
        p = self._recipe

        n_successful_demos = 0
        n_failed_demos = 0
        safepos_idx = 0
        noise = dict()
        safepos = []
        pbar = tqdm(total=n_demos)
        # two_finger gripper sites, as defined in gripper xml
        griptip_site = "griptip_site"
        gripbase_site = "right_gripper_base_collision"
        grip_site = "grip_site"
        # define assembly order and furniture specific variables
        grip_angles = None
        if "grip_angles" in p.keys():
            grip_angles = p["grip_angles"]
        self._config.max_episode_steps = p["max_success_steps"] + 1
        # align_g target vector
        align_g_tgt = np.array([0, -1])
        # background specific, only tested on --background Industrial
        ground_offset = 0.0001
        self.min_rot_act = p["min_rot_act"]
        self.min_rot_act_fine = p["min_rot_act_fine"]

        while n_successful_demos < n_demos:
            ob = self.reset()
            self._used_sites = set()
            self.get_random_noise(noise)

            for j in range(len(self._config.preassembled), len(p["recipe"])):
                self._phase_num = 0
                t_fwd = None
                z_move_g_prev = None
                safepos_idx = 0
                safepos.clear()
                self._phase = self._phases[self._phase_num]
                gbody_name, tbody_name = p["recipe"][j]
                # use conn_sites in site_recipe, other dynamically get closest/furthest conn_site from gripper
                if "site_recipe" in p:
                    gconn, tconn = p["site_recipe"][j][:2]
                else:
                    gconn_names, tconn_names = self.get_conn_sites(
                        gbody_name, tbody_name
                    )
                    grip_pos = self._get_pos(grip_site)
                    if p["use_closest"]:
                        gconn = self.get_closest_conn_site(gconn_names, grip_pos)
                    else:
                        gconn = self.get_furthest_conn_site(gconn_names, grip_pos)
                g_pos = self._get_pos(gbody_name)
                allowed_angles = [float(x) for x in gconn.split(",")[1:-1] if x]
                # get unused target sites
                for i in range(len(p["recipe"])):
                    g_l = (
                        gbody_name + "_ltgt_site" + str(i)
                    )  # gripped body left gripsite
                    g_r = (
                        gbody_name + "_rtgt_site" + str(i)
                    )  # gripped body right gripsite
                    if g_l in self._used_sites or g_r in self._used_sites:
                        pass
                    else:
                        self._used_sites.add(g_l)
                        self._used_sites.add(g_r)
                        break

                if self._config.render:
                    self.render()
                if self._config.record_vid:
                    self.vid_rec.capture_frame(self.render("rgb_array")[0])

                # initiate phases for single-part assembly
                while self._phase != "part_done":
                    action = np.zeros((8,))
                    # logger.info(self._phase)
                    if self._phase == "xy_move_g":
                        grip_xy_pos = self._get_pos(grip_site)[0:2]
                        g_xy_pos = (self._get_pos(g_l) + self._get_pos(g_r))[0:2] / 2
                        action[0:2] = self.move_xy(
                            grip_xy_pos, g_xy_pos, p["eps"], noise=noise[self._phase]
                        )

                    elif self._phase == "align_g":
                        action[6] = -1
                        if grip_angles is None or grip_angles[j] is not None:
                            # align gripper fingers with grip sites
                            gripfwd_xy = self._get_forward_vector(grip_site)[0:2]
                            gvec_xy = (self._get_pos(g_r) - self._get_pos(g_l))[0:2]
                            xy_ac = self.align_gripsites(
                                gripfwd_xy, gvec_xy, p["rot_eps"]
                            )
                            # xy_ac = self.align2D(gripfwd_xy, gvec_xy, p['rot_eps'])
                            # point gripper z downwards
                            gripvec = self._get_up_vector(grip_site)
                            yz_ac = self.align2D(
                                gripvec[1:3], align_g_tgt, p["rot_eps"]
                            )
                            xz_ac = self.align2D(
                                gripvec[0::2], align_g_tgt, p["rot_eps"]
                            )
                            rot_action = [xy_ac, yz_ac, xz_ac]
                            if rot_action == [0, 0, 0]:
                                grip_pos = self._get_pos(grip_site)[0:2]
                                g_pos = (self._get_pos(g_l) + self._get_pos(g_r)) / 2
                                action[0:2] = self.move_xy(
                                    grip_pos, g_pos[0:2], p["eps"]
                                )
                            else:
                                action[3:6] = rot_action
                        else:
                            self._phase_num += 1

                    elif self._phase == "z_move_g":
                        action[6] = -1
                        grip_pos = self._get_pos(grip_site)
                        grip_tip = self._get_pos(griptip_site)
                        g_pos = (self._get_pos(g_l) + self._get_pos(g_r)) / 2
                        d = (g_pos) - grip_pos
                        if z_move_g_prev is None:
                            z_move_g_prev = grip_tip[2] + ground_offset

                        if abs(d[2]) > p["eps"] and grip_tip[2] < z_move_g_prev:
                            action[0:3] = d
                            z_move_g_prev = grip_tip[2] - ground_offset
                        else:
                            self._phase_num += 1
                            if p["grip_safepos"][j] is not None:
                                gripbase_pos = self._get_pos(gripbase_site)
                                for pos in p["grip_safepos"][j]:
                                    safepos.append(gripbase_pos + pos)

                    elif self._phase == "move_grip_safepos":
                        action[6] = 1
                        if p["grip_safepos"][j] is None or (
                            p["grip_safepos"][j]
                            and safepos_idx >= len(p["grip_safepos"][j])
                        ):
                            safepos_idx = 0
                            safepos.clear()
                            self._phase_num += 1
                            gconn_pos = self.sim.data.get_site_xpos(gconn)
                            if "site_recipe" not in p:
                                if p["use_closest"]:
                                    tconn = self.get_closest_conn_site(
                                        tconn_names, gconn_pos
                                    )
                                else:
                                    tconn = self.get_furthest_conn_site(
                                        tconn_names, gconn_pos
                                    )
                            tconn_pos = self.sim.data.get_site_xpos(tconn)
                        else:
                            gripbase_pos = self._get_pos(gripbase_site)
                            action[0:3] = self.move_xyz(
                                gripbase_pos,
                                safepos[safepos_idx],
                                p["eps"],
                                noise=noise[self._phase],
                            )
                            if not np.any(action[0:3]):
                                safepos_idx += 1

                    elif self._phase == "xy_move_t":
                        action[6] = 1
                        grip_pos = self._get_pos(grip_site)
                        action[0:2] = self.move_xy(
                            grip_pos[0:2],
                            tconn_pos[0:2],
                            p["eps"],
                            noise=noise[self._phase],
                        )

                    elif self._phase == "align_conn":
                        action[6] = 1
                        g_up = self._get_up_vector(gconn)
                        t_up = self._get_up_vector(tconn)
                        yz_ac = self.align2D(g_up[1:3], t_up[1:3], p["rot_eps"])
                        xz_ac = self.align2D(g_up[0::2], t_up[0::2], p["rot_eps"])
                        rot_action = [0, yz_ac, xz_ac]
                        if rot_action == [0, 0, 0]:
                            g_xy_fwd = self._get_forward_vector(gconn)
                            if t_fwd is None:
                                t_fwd = self.get_closest_xy_fwd(
                                    allowed_angles, gconn, tconn
                                )
                                t_xy_fwd = t_fwd[0:2]
                            xy_ac = self.align2D(g_xy_fwd, t_xy_fwd, p["rot_eps"])
                            if xy_ac == 0:
                                t_fwd = None
                                self._phase_num += 1
                            else:
                                action[3] = -xy_ac
                        else:
                            action[3:6] = rot_action

                    elif self._phase == "xy_move_conn":
                        action[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        action[0:2] = self.move_xy(
                            gconn_pos[0:2], tconn_pos[0:2], p["eps"]
                        )

                    elif self._phase == "z_move_conn":
                        action[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        action[0:3] = self.move_z(
                            gconn_pos, tconn_pos, p["z_finedist"], p["z_conn_dist"]
                        )

                    elif self._phase == "align_conn_fine":
                        action[6] = 1
                        g_up = self._get_up_vector(gconn)
                        t_up = self._get_up_vector(tconn)
                        yz_ac = self.align2D(g_up[1:], t_up[1:], p["rot_eps_fine"])
                        xz_ac = self.align2D(g_up[0::2], t_up[0::2], p["rot_eps_fine"])
                        rot_action = [0, yz_ac, xz_ac]
                        if rot_action == [0, 0, 0]:
                            g_xy_fwd = self._get_forward_vector(gconn)[0:2]
                            if t_fwd is None:
                                t_fwd = self.get_closest_xy_fwd(
                                    allowed_angles, gconn, tconn
                                )
                                t_xy_fwd = t_fwd[0:2]
                            xy_ac = self.align2D(g_xy_fwd, t_xy_fwd, p["rot_eps_fine"])
                            if xy_ac == 0:
                                # must be finely aligned rotationally and translationally to go to next phase
                                action[0:2] = self.move_xy(
                                    gconn_pos[0:2], tconn_pos[0:2], p["eps_fine"]
                                )
                            else:
                                action[3] = -xy_ac
                        else:
                            action[3:6] = rot_action

                    elif self._phase == "z_move_conn_fine":
                        action[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        action[0:3] = self.move_z(
                            gconn_pos,
                            tconn_pos,
                            p["eps_fine"],
                            p["z_conn_dist"],
                            fine=p["fine_magnitude"],
                        )
                        if not np.any(action[0:3]):
                            action[7] = 1
                        if p["nogrip_safepos"][j] is not None:
                            gripbase_pos = self._get_pos(gripbase_site)
                            for pos in p["nogrip_safepos"][j]:
                                safepos.append(gripbase_pos + pos)

                    elif self._phase == "move_nogrip_safepos":
                        action[6] = -1
                        if p["nogrip_safepos"][j] is None or (
                            p["nogrip_safepos"][j]
                            and safepos_idx >= len(p["nogrip_safepos"][j])
                        ):
                            safepos_idx = 0
                            safepos.clear()
                            self._phase_num += 1
                        else:
                            gripbase_pos = self._get_pos(gripbase_site)
                            action[0:3] = self.move_xyz(
                                gripbase_pos,
                                safepos[safepos_idx],
                                p["eps"],
                                noise=noise[self._phase],
                            )
                            if not np.any(action[0:3]):
                                safepos_idx += 1

                    self._phase = self._phases[self._phase_num]
                    action[0:3] = p["lat_magnitude"] * action[0:3]
                    action[3:6] = p["rot_magnitude"] * action[3:6]
                    action = self._norm_rot_action(action)
                    action = self._cap_action(action)
                    ob, reward, _, info = self.step(action)

                    if self._config.render:
                        self.render()
                    if self._config.record_vid:
                        self.vid_rec.capture_frame(self.render("rgb_array")[0])

                    if self._episode_length > p["max_success_steps"]:
                        logger.info(
                            "Time-limit exceeds %d/%d",
                            self._episode_length,
                            p["max_success_steps"],
                        )
                        break
                    if self._success:
                        break

                if self._part_success:
                    self._used_sites.add(gconn)
                    self._used_sites.add(tconn)
                    self._part_success = False

                if self._success:
                    logger.warn(
                        "assembled (%s) in %d steps!",
                        self._config.furniture_name,
                        self._episode_length,
                    )
                    if self._config.record_vid:
                        self.vid_rec.close()
                    if self._config.start_count is not None:
                        demo_count = self._config.start_count + n_successful_demos
                        self._demo.save(self.file_prefix, count=demo_count)
                    else:
                        self._demo.save(self.file_prefix)
                    pbar.update(1)
                    n_successful_demos += 1
                    break
                elif self._episode_length > p["max_success_steps"]:
                    # failed
                    logger.warn("Failed to assemble!")
                    n_failed_demos += 1
                    if self._config.record_vid:
                        self.vid_rec.close(success=True)
                    break

        logger.info("n_failed_demos: %d", n_failed_demos)


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureSawyerGenEnv")
    config, unparsed = parser.parse_known_args()
    if len(unparsed):
        logger.error("Unparsed argument is detected:\n%s", unparsed)
        return

    env = FurnitureSawyerGenEnv(config)
    env.generate_demos(config.n_demos)


if __name__ == "__main__":
    main()
