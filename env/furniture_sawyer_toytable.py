""" Define sawyer environment class FurnitureSawyerToyTableEnv. """

from collections import OrderedDict

import numpy as np

from env.models import furniture_name2id
from env.furniture_sawyer import FurnitureSawyerEnv
import env.transform_utils as T
from util.logger import logger

class FurnitureSawyerToyTableEnv(FurnitureSawyerEnv):
    """
    Sawyer environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.furniture_id = furniture_name2id["toy_table"]

        super().__init__(config)
        # default values
        self._env_config.update({
            "pos_dist": 0.06,
            "rot_dist_up": 0.97,
            "rot_dist_forward": 0.9,
            "project_dist": -1,
            "site_dist_rew": config.site_dist_rew,
            "site_up_rew": config.site_up_rew,
            "grip_up_rew": config.grip_up_rew,
            "grip_dist_rew": config.grip_dist_rew,
            "aligned_rew": config.aligned_rew,
            "connect_rew": config.connect_rew,
            "success_rew": config.success_rew,
            "pick_rew": config.pick_rew,
            "ctrl_penalty": config.ctrl_penalty,
            "grip_z_offset": config.grip_z_offset,
        })
        self._gravity_compensation = 1
        # requires multiple connection actions to make connection between two
        # parts.
        self._num_connect_steps = 0

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        ob, _, done, _ = super(FurnitureSawyerEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)

        for i, body in enumerate(self._object_names):
            pose  = self._get_qpos(body)
            logger.debug(f'{body} {pose[:3]} {pose[3:]}')

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

        self._leg_picked = False

        top_site_name = "top-leg,,conn_site4"
        leg_site_name = "leg-top,,conn_site4"
        top_site_xpos = self._site_xpos_xquat(top_site_name)
        leg_site_xpos = self._site_xpos_xquat(leg_site_name)

        top_site_pos = top_site_xpos[:3]
        self._prev_top_site_pos = top_site_pos

        up1 = self._get_up_vector(top_site_name)
        up2 = self._get_up_vector(leg_site_name)
        # calculate distance between site + z-offset and other site
        point_above_topsite = top_site_xpos[:3] + np.array([0,0,0.15])
        offset_dist = T.l2_dist(point_above_topsite, leg_site_xpos[:3])
        site_dist = T.l2_dist(top_site_xpos[:3], leg_site_xpos[:3])
        rot_dist_up = T.cos_dist(up1, up2)
        rot_dist_project1_2 = T.cos_dist(up1, leg_site_xpos[:3] - top_site_xpos[:3])
        rot_dist_project2_1 = T.cos_dist(-up2, top_site_xpos[:3] - leg_site_xpos[:3])

        leg_top_site_xpos = self._site_xpos_xquat("2_part2_top_site")
        leg_bot_site_xpos = self._site_xpos_xquat("2_part2_bottom_site")
        # 1st phase is to pick the object
        hand_pos = self.sim.data.site_xpos[self.eef_site_id]
        # midpoint of the cylinder
        grasp_pos = leg_bot_site_xpos[:3] + 0.5 * (leg_top_site_xpos - leg_bot_site_xpos)[:3]
        grasp_pos_offset = grasp_pos + np.array([0,0,self._env_config['grip_z_offset']])
        grip_dist = np.linalg.norm(hand_pos - grasp_pos_offset)

        # grip up dist
        hand_up = self._get_up_vector('grip_site')
        grip_site_up = self._get_up_vector('2_part2_top_site')
        grip_up_dist = T.cos_dist(hand_up, grip_site_up)

        # offset site dist
        self._prev_offset_dist = offset_dist
        # actual site dist
        self._prev_site_dist = site_dist
        self._prev_rot_dist_up = rot_dist_up
        self._prev_rot_dist_project1_2 = rot_dist_project1_2
        self._prev_rot_dist_project2_1 = rot_dist_project2_1
        self._prev_grip_up_dist = grip_up_dist

        self._prev_grip_dist = grip_dist
        self._phase = 'grasp_offset'
        self._num_connect_successes = 0


    def _finger_contact(self, obj):
        """
        Returns if left, right fingers contact with obj
        """
        touch_left_finger = False
        touch_right_finger = False
        for j in range(self.sim.data.ncon):
            c = self.sim.data.contact[j]
            body1 = self.sim.model.geom_bodyid[c.geom1]
            body2 = self.sim.model.geom_bodyid[c.geom2]
            body1_name = self.sim.model.body_id2name(body1)
            body2_name = self.sim.model.body_id2name(body2)

            if c.geom1 in self.l_finger_geom_ids[0] and body2_name == obj:
                touch_left_finger = True
            if c.geom2 in self.l_finger_geom_ids[0] and body1_name == obj:
                touch_left_finger = True

            if c.geom1 in self.r_finger_geom_ids[0] and body2_name == obj:
                touch_right_finger = True
            if c.geom2 in self.r_finger_geom_ids[0] and body1_name == obj:
                touch_right_finger = True

        return touch_left_finger, touch_right_finger




    def _place_objects(self):
        """
        Returns fixed initial position and rotations of the toy table.
        The first case has the table top on the left and legs on the right.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        # pos_init = [[ 0.21250838, -0.1163671 ,  0.02096991], [-0.30491682, -0.09045364,  0.03429339],[ 0.38134436, -0.11249256,  0.02096991],[ 0.12432612, -0.13662719,  0.02096991],[ 0.29537311, -0.12992911,  0.02096991]]
        # quat_init = [[0.706332  , 0.70633192, 0.03309327, 0.03309326], [ 0.00000009, -0.99874362, -0.05011164,  0.00000002], [ 0.70658149,  0.70706735, -0.00748174,  0.0272467 ], [0.70610751, 0.7061078 , 0.03757641, 0.03757635], [0.70668613, 0.70668642, 0.02438253, 0.02438249]]

        pos_init = [[-0.34684698 + 0.15, -0.12887974,  0.03418991],[0.03472849 - 0.0285, 0.11868485 - 0.05, 0.02096991]]
        quat_init = [[ 0.00000009, -0.99874362, -0.05011164,  0.00000002],  [-0.70610751, 0.7061078 , -0.03757641, 0.03757635]]

        return pos_init, quat_init


    def _compute_reward(self, action):
        """
        Two stage reward.
        The first stage gives reward for picking up a table top and a leg.

        The next stage gives reward for bringing the leg connection site close to the table
        connection site.
        """
        rew = pick_rew = grip_up_rew = grip_dist_rew = site_dist_rew = site_up_rew = connect_rew = success_rew = \
            aligned_rew = ctrl_penalty = 0
        done = self._num_connected > 0
        info = {}

        top_site_name = "top-leg,,conn_site3"
        leg_site_name = "leg-top,,conn_site4"
        top_site_xpos = self._site_xpos_xquat(top_site_name)
        leg_site_xpos = self._site_xpos_xquat(leg_site_name)
        leg_top_site_xpos = self._site_xpos_xquat("2_part2_top_site")
        leg_bot_site_xpos = self._site_xpos_xquat("2_part2_bottom_site")
        ctrl_penalty = -self._env_config['ctrl_penalty'] * np.linalg.norm(action[:6])

        hand_pos = self.sim.data.site_xpos[self.eef_site_id]
        hand_up = self._get_up_vector('grip_site')
        grasp_pos = leg_bot_site_xpos[:3] + 0.5 * (leg_top_site_xpos - leg_bot_site_xpos)[:3]

        touch_left, touch_right = self._finger_contact('2_part2')
        gripped = touch_left and touch_right

        up1 = self._get_up_vector(top_site_name)
        up2 = self._get_up_vector(leg_site_name)
        rot_dist_up = T.cos_dist(up1, up2)
        rot_dist_project1_2 = T.cos_dist(up1, leg_site_xpos[:3] - top_site_xpos[:3])
        rot_dist_project2_1 = T.cos_dist(-up2, top_site_xpos[:3] - leg_site_xpos[:3])

        site_dist = T.l2_dist(top_site_xpos[:3], leg_site_xpos[:3])


        point_above_topsite = top_site_xpos[:3] + np.array([0,0,0.15])
        offset_dist = T.l2_dist(point_above_topsite, leg_site_xpos[:3])

        if self._phase == 'grasp_offset': # make hand hover over object
            # midpoint of the cylinder
            grasp_pos_offset = grasp_pos + np.array([0,0,self._env_config['grip_z_offset']])
            grip_dist = np.linalg.norm(hand_pos - grasp_pos_offset)
            logger.debug(f'grip_dist {grip_dist}')
            grip_dist_rew = self._env_config['grip_dist_rew'] * (self._prev_grip_dist - grip_dist)
            self._prev_grip_dist = grip_dist

            # up vector of leg and up vector of grip site should be perpendicular
            grip_site_up = self._get_up_vector('2_part2_top_site')
            grip_up_dist = np.abs(T.cos_dist(hand_up, grip_site_up))
            logger.debug(f'grip_up_dist {grip_up_dist}')
            grip_up_rew = self._env_config['grip_up_rew'] * (self._prev_grip_up_dist -grip_up_dist)
            self._prev_grip_up_dist = grip_up_dist


            if grip_dist < 0.03 and grip_up_dist < 0.15:
                logger.debug('Done with grasp offset alignment')
                self._phase = 'grasp_leg'
                self._prev_grip_dist = np.linalg.norm(hand_pos - grasp_pos)

        elif self._phase == 'grasp_leg': # move hand down to grasp object
             # midpoint of the cylinder
            grip_dist = np.linalg.norm(hand_pos - grasp_pos)
            logger.debug(f'grip_dist {grip_dist}')
            grip_dist_rew = self._env_config['grip_dist_rew'] * (self._prev_grip_dist - grip_dist)
            self._prev_grip_dist = grip_dist

            # up vector of leg and up vector of grip site should be perpendicular
            grip_site_up = self._get_up_vector('2_part2_top_site')
            grip_up_dist = np.abs(T.cos_dist(hand_up, grip_site_up))
            logger.debug(f'grip_up_dist {grip_up_dist}')
            logger.debug(f'grip z dist {hand_pos[-1] - grasp_pos[-1]}')
            grip_up_rew = self._env_config['grip_up_rew'] * (self._prev_grip_up_dist - grip_up_dist)
            self._prev_grip_up_dist = grip_up_dist

            if grip_dist < 0.03 and (hand_pos[-1] - grasp_pos[-1]) < 0.001 and grip_up_dist < 0.12:
                logger.debug('Done with grasp leg alignment')
                self._phase = 'grip_leg'

        elif self._phase == 'grip_leg': # close the gripper
            if gripped:
                logger.debug('Gripped leg')
                pick_rew = self._env_config['pick_rew']
                self._phase = 'move_leg_up'
                self._leg_picked = True
                # set the grip pos offset to above this grip point
                self._grip_pos_offset = hand_pos + np.array([0, 0, 0.15])
                self._prev_grip_pos_offset_dist = np.linalg.norm(hand_pos - self._grip_pos_offset)

        elif self._leg_picked and not gripped: # dropped the leg
            done = True
        elif self._phase == 'move_leg_up': # move the leg up

            grip_dist = np.linalg.norm(hand_pos - self._grip_pos_offset)
            grip_dist_rew = self._env_config['grip_dist_rew'] * (self._prev_grip_pos_offset_dist - grip_dist)
            self._prev_grip_pos_offset_dist = grip_dist
            if grip_dist < 0.03:
                logger.warning('Done with move leg up')
                self._phase = 'move_leg'

        elif self._phase == 'move_leg': # move leg to offset

            site_dist_diff = self._prev_offset_dist - offset_dist
            site_dist_rew = self._env_config['site_dist_rew'] * site_dist_diff
            self._prev_offset_dist = offset_dist
            logger.debug(f'offset_dist: {offset_dist}')

            # give rew for making angular dist between sites
            site_up_diff = rot_dist_up - self._prev_rot_dist_up
            site_up_diff += (rot_dist_project1_2 - self._prev_rot_dist_project1_2) * 0.5
            site_up_diff += (rot_dist_project2_1 - self._prev_rot_dist_project2_1) * 0.5
            site_up_rew = self._env_config['site_up_rew'] * site_up_diff
            logger.debug(f'offset rotation dist: {site_up_diff}')
            self._prev_rot_dist_up = rot_dist_up
            self._prev_rot_dist_project1_2 = rot_dist_project1_2
            self._prev_rot_dist_project2_1 = rot_dist_project2_1

            if offset_dist < self._env_config['pos_dist'] and rot_dist_up > self._env_config['rot_dist_up'] \
                and rot_dist_project1_2 > 0.98 and rot_dist_project2_1 > 0.98:
                self._phase = 'move_leg_2'
                aligned_rew = self._env_config['aligned_rew']
                logger.debug('leg aligned with offset')

        #elif self._phase == 'rot_up':

        #    site_up_diff = rot_dist_up - self._prev_rot_dist_up
        #    site_up_rew = self._env_config['site_up_rew'] * site_up_diff
        #    self._prev_rot_dist_up = rot_dist_up
        #    logger.debug(f'rot_dist_up: {rot_dist_up}')
        #    if rot_dist_up > self._env_config['rot_dist_up']:
        #        self._phase = 'move_leg_2'
        #        aligned_rew = self._env_config['aligned_rew']
        #        logger.debug('leg rot is aligned')

        elif self._phase == 'move_leg_2':
            site_dist_diff = self._prev_site_dist - site_dist
            site_dist_rew = self._env_config['site_dist_rew'] * site_dist_diff
            self._prev_site_dist = site_dist
            logger.debug(f'site_dist: {site_dist}')

            # give rew for making angular dist between sites
            site_up_diff = rot_dist_up - self._prev_rot_dist_up
            site_up_diff += (rot_dist_project1_2 - self._prev_rot_dist_project1_2) * 0.5
            site_up_diff += (rot_dist_project2_1 - self._prev_rot_dist_project2_1) * 0.5
            site_up_rew = self._env_config['site_up_rew'] * site_up_diff
            self._prev_rot_dist_up = rot_dist_up
            self._prev_rot_dist_project1_2 = rot_dist_project1_2
            self._prev_rot_dist_project2_1 = rot_dist_project2_1

            if site_dist < self._env_config['pos_dist'] and rot_dist_up > self._env_config['rot_dist_up'] \
                and rot_dist_project1_2 > 0.98 and rot_dist_project2_1 > 0.98:
                self._phase = 'connect'
                aligned_rew = self._env_config['aligned_rew']
                logger.debug('leg aligned with site')

        elif self._phase == 'connect':
            connect = action[-1]
            if connect > 0:
                connect_rew +=  self._env_config['connect_rew']
            if self._num_connected > 0:
                success_rew = self._env_config['success_rew']
                # done = True


        info['phase'] = self._phase
        info['leg_picked'] = self._leg_picked
        info['grip_dist_rew'] = grip_dist_rew
        info['grip_up_rew'] = grip_up_rew
        info['pick_rew'] = pick_rew
        info['site_dist_rew'] = site_dist_rew
        info['site_up_rew'] = site_up_rew
        info['aligned_rew'] = aligned_rew
        info['connect_rew'] = connect_rew
        info['success_rew'] = success_rew
        info['ctrl_penalty'] = ctrl_penalty
        info['rot_dist_up'] = rot_dist_up
        info['site_dist'] = site_dist
        info['offset_dist'] = offset_dist

        rew = pick_rew + grip_dist_rew + grip_up_rew + site_dist_rew + site_up_rew + connect_rew + \
               + aligned_rew + success_rew + ctrl_penalty
        return rew, done, info
        #holding_top = self._cursor_selected[0] == '4_part4'
        #holding_leg = self._cursor_selected[1] == '2_part2'
        #c0_action, c1_action = action[:7], action[7:14]
        #c0_moverotate, c1_moverotate = c0_action[:-1], c1_action[:-1]
        #c0_ctrl_penalty, c1_ctrl_penalty = 2 * np.linalg.norm(c0_moverotate, 2), np.linalg.norm(c1_moverotate, 2)
        #ctrl_penalty = -self._env_config['ctrl_penalty'] * (c0_ctrl_penalty + c1_ctrl_penalty)

        #top_site_name = "top-leg,,conn_site4"
        #leg_site_name = "leg-top,,conn_site4"
        #top_site_xpos = self._site_xpos_xquat(top_site_name)
        #leg_site_xpos = self._site_xpos_xquat(leg_site_name)

        #up1 = self._get_up_vector(top_site_name)
        #up2 = self._get_up_vector(leg_site_name)
        ## calculate distance between site + z-offset and other site
        #point_above_topsite = top_site_xpos[:3] + np.array([0,0,0.15])
        #pos_dist = T.l2_dist(point_above_topsite, leg_site_xpos[:3])
        #site_dist = T.l2_dist(top_site_xpos[:3], leg_site_xpos[:3])
        #rot_dist_up = T.cos_dist(up1, up2)

        ## cursor 0 select table top
        #if holding_top and not self._top_picked:
        #    pick_rew += self._env_config['pick_rew']
        #    self._top_picked = True
        ## cursor 1 select table leg
        #if holding_leg and not self._leg_picked:
        #    pick_rew += self._env_config['pick_rew']
        #    self._leg_picked = True

        #if self._num_connected > 0:
        #    success_rew = self._env_config['success_rew']

        ## if parts are in hand, then give reward for moving parts closer
        #elif holding_top and holding_leg:

        #    project1_2 = np.dot(up1, T.unit_vector(leg_site_xpos[:3] - top_site_xpos[:3]))
        #    project2_1 = np.dot(up2, T.unit_vector(top_site_xpos[:3] - leg_site_xpos[:3]))

        #    angles_aligned = rot_dist_up > self._env_config['rot_dist_up']
        #    dist_aligned = pos_dist < 0.1
        #    sites_aligned = site_dist < self._env_config['pos_dist']

        #    if angles_aligned and dist_aligned and sites_aligned:
        #        self._phase = 'connect'
        #    elif angles_aligned and dist_aligned:
        #        self._phase = 'align_eucl_2'
        #    elif dist_aligned:
        #        self._phase = 'align_rot'
        #    elif angles_aligned:
        #        self._phase = 'align_eucl'

        #    if self._phase == 'connect':
        #        # give reward for getting alignment right
        #        aligned_rew = self._env_config['aligned_rew']
        #        connect = action[14]
        #        if connect > 0 and self._num_connect_successes < 10:
        #            connect_rew +=  self._env_config['connect_rew']
        #            self._num_connect_successes += 1

        #    elif self._phase == 'align_rot':
        #        # Second phase: bring angle distance close together
        #        # give bonus for being done with align_eucl
        #        aligned_rew = self._env_config['aligned_rew']/10
        #        # give rew for making angular dist between sites 1
        #        site_up_diff = rot_dist_up - self._prev_rot_dist_up
        #        if not abs(site_up_diff) < 0.01:
        #            site_up_rew = self._env_config['site_up_rew'] * site_up_diff

        #    elif self._phase == 'align_eucl':
        #        # First phase: bring eucl distance close
        #        aligned_rew = self._env_config['aligned_rew']/10
        #        # give rew for minimizing eucl distance between sites
        #        site_dist_diff = self._prev_pos_dist - pos_dist
        #        if not abs(site_dist_diff) < 0.01:
        #            site_dist_rew = self._env_config['site_dist_rew'] * site_dist_diff

        #    elif self._phase == 'align_eucl_2':
        #        # Third phase: bring sites close together
        #        # give reward for getting alignment right
        #        aligned_rew = self._env_config['aligned_rew']/5
        #        # give reward for minimizing eucl distance between sites
        #        site_dist_diff = self._prev_site_dist - site_dist
        #        if not abs(site_dist_diff) < 0.01:
        #            site_dist_rew = self._env_config['site_dist_rew'] * site_dist_diff


        #    self._prev_site_dist = site_dist
        #    self._prev_pos_dist = pos_dist
        #    self._prev_rot_dist_up = rot_dist_up
        #elif (not holding_top and self._top_picked) or (not holding_leg and self._leg_picked):
        #    # give penalty for dropping top or leg
        #    #pick_rew = -2
        #    done = True

        #info['phase'] = self._phase
        #info['leg_picked'] = self._leg_picked
        #info['top_picked'] = self._top_picked
        #info['pick_rew'] = pick_rew
        #info['site_dist_rew'] = site_dist_rew
        #info['site_up_rew'] = site_up_rew
        #info['aligned_rew'] = aligned_rew
        #info['connect_rew'] = connect_rew
        #info['success_rew'] = success_rew
        #info['ctrl_penalty'] = ctrl_penalty
        #info['c0_ctrl_penalty'] = c0_ctrl_penalty
        #info['c1_ctrl_penalty'] = c1_ctrl_penalty
        #info['rot_dist_up'] = rot_dist_up
        #info['site_dist'] = site_dist
        #info['pos_dist'] = pos_dist

        #rew = pick_rew + site_dist_rew + site_up_rew + connect_rew + \
        #        + aligned_rew + success_rew + ctrl_penalty
        #return rew, done, info


def main():
    from config import create_parser
    parser = create_parser(env="FurnitureSawyerToyTableEnv")
    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Sawyer environment
    env = FurnitureSawyerToyTableEnv(config)
    env.run_manual(config)


if __name__ == "__main__":
    main()
