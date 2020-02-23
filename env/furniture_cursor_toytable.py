""" Define cursor environment class FurnitureCursorEnv. """

from collections import OrderedDict

import numpy as np

from env.models import furniture_name2id
from env.furniture import FurnitureEnv
import env.transform_utils as T
from util.logger import logger

class FurnitureCursorToyTableEnv(FurnitureEnv):
    """
    Cursor environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.agent_type = 'Cursor'
        config.furniture_id = furniture_name2id["toy_table"]

        super().__init__(config)
        # default values
        self._env_config.update({
            "pos_dist": 0.1,
            "rot_dist_up": 0.9,
            "rot_dist_forward": 0.9,
            "project_dist": -1,
            "site_dist_rew": config.site_dist_rew,
            "site_up_rew": config.site_up_rew,
            "aligned_rew": config.aligned_rew,
            "connect_rew": config.connect_rew,
            "success_rew": config.success_rew,
            "pick_rew": config.pick_rew,
            "ctrl_penalty": config.ctrl_penalty,
        })

        # turn on the gravity compensation for selected furniture pieces
        self._gravity_compensation = 1

        # requires multiple connection actions to make connection between two
        # parts.
        self._num_connect_steps = 10

        self._cursor_selected = [None, None]

    @property
    def observation_space(self):
        """
        Returns the observation space.
        """
        ob_space = super().observation_space

        if self._robot_ob:
            ob_space['robot_ob'] = [(3 + 1) * 2]

        return ob_space

    @property
    def dof(self):
        """
        Returns the DoF of the cursor agent.
        """
        assert self._control_type == 'ik'
        dof = (3 + 3 + 1) * 2 + 1  # (move, rotate, select) * 2 + connect
        return dof

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """

        ob, _, done, _ = super()._step(a)

        reward, done, info = self._compute_reward(a)

        if self._success:
            logger.info('Success!')


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

        # history for reward function
        self._top_picked = False
        self._leg_picked = False

        top_site_name = "top-leg,,conn_site4"
        leg_site_name = "leg-top,,conn_site4"
        top_site_xpos = self._site_xpos_xquat(top_site_name)
        leg_site_xpos = self._site_xpos_xquat(leg_site_name)

        top_site_pos = top_site_xpos[:3]
        self._prev_top_site_pos = top_site_pos

        up1 = self._get_up_vector(top_site_name)
        up2 = self._get_up_vector(leg_site_name)
        forward1 = self._get_forward_vector(top_site_name)
        forward2 = self._get_forward_vector(leg_site_name)
        # calculate distance between site + z-offset and other site
        point_above_topsite = top_site_xpos[:3] + np.array([0,0,0.125])
        pos_dist = T.l2_dist(point_above_topsite, leg_site_xpos[:3])
        site_dist = T.l2_dist(top_site_xpos[:3], leg_site_xpos[:3])
        rot_dist_up = T.cos_dist(up1, up2)

        # offset site dist
        self._prev_pos_dist = pos_dist
        # actual site dist
        self._prev_site_dist = site_dist
        self._prev_rot_dist_up = rot_dist_up

        self._phase = 'align_eucl'

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

        pos_init = [[-0.30491682, -0.09045364,  0.03429339],[ 0.12432612, -0.13662719,  0.02096991]]
        quat_init = [[ 0.00000009, -0.99874362, -0.05011164,  0.00000002],  [0.70610751, 0.7061078 , 0.03757641, 0.03757635]]

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
            robot_states["cursor_state"] = np.array([self._cursor_selected[0] is not None,
                                                     self._cursor_selected[1] is not None])

            state['robot_ob'] = np.concatenate(
                [x.ravel() for _, x in robot_states.items()]
            )

        return state

    def _initialize_robot_pos(self):
        """
        Initializes cursor position to be on top of parts
        """
        self._set_pos('cursor0', [-0.35, -0.125, 0.0125])
        self._set_pos('cursor1', [0.075, -0.175, 0.0375])

    def _compute_reward(self, action):
        """
        Two stage reward.
        The first stage gives reward for picking up a table top and a leg.

        The next stage gives reward for bringing the leg connection site close to the table
        connection site.
        """
        rew = pick_rew = site_dist_rew = site_up_rew = connect_rew = success_rew = \
            aligned_rew = ctrl_penalty = 0
        done = self._num_connected > 0
        info = {}
        holding_top = self._cursor_selected[0] == '4_part4'
        holding_leg = self._cursor_selected[1] == '2_part2'
        c0_action, c1_action = action[:7], action[7:14]
        c0_moverotate, c1_moverotate = c0_action[:-1], c1_action[:-1]
        c0_ctrl_penalty, c1_ctrl_penalty = 2 * np.linalg.norm(c0_moverotate, 2), np.linalg.norm(c1_moverotate, 2)
        ctrl_penalty = -self._env_config['ctrl_penalty'] * (c0_ctrl_penalty + c1_ctrl_penalty)

        top_site_name = "top-leg,,conn_site4"
        leg_site_name = "leg-top,,conn_site4"
        top_site_xpos = self._site_xpos_xquat(top_site_name)
        leg_site_xpos = self._site_xpos_xquat(leg_site_name)

        up1 = self._get_up_vector(top_site_name)
        up2 = self._get_up_vector(leg_site_name)
        forward1 = self._get_forward_vector(top_site_name)
        forward2 = self._get_forward_vector(leg_site_name)
        # calculate distance between site + z-offset and other site
        point_above_topsite = top_site_xpos[:3] + np.array([0,0,0.125])
        pos_dist = T.l2_dist(point_above_topsite, leg_site_xpos[:3])
        site_dist = T.l2_dist(top_site_xpos[:3], leg_site_xpos[:3])
        rot_dist_up = T.cos_dist(up1, up2)

        # cursor 0 select table top
        if holding_top and not self._top_picked:
            pick_rew += self._env_config['pick_rew']
            self._top_picked = True
        # cursor 1 select table leg
        if holding_leg and not self._leg_picked:
            pick_rew += self._env_config['pick_rew']
            self._leg_picked = True

        if self._num_connected > 0:
            success_rew = self._env_config['success_rew']

        # if parts are in hand, then give reward for moving parts closer
        elif holding_top and holding_leg:

            project1_2 = np.dot(up1, T.unit_vector(leg_site_xpos[:3] - top_site_xpos[:3]))
            project2_1 = np.dot(up2, T.unit_vector(top_site_xpos[:3] - leg_site_xpos[:3]))

            angles_aligned = rot_dist_up > self._env_config['rot_dist_up']
            dist_aligned = pos_dist < 0.1
            sites_aligned = site_dist < self._env_config['pos_dist']

            if angles_aligned and dist_aligned and sites_aligned:
                self._phase = 'connect'
            elif angles_aligned and dist_aligned:
                self._phase = 'align_eucl_2'
            elif dist_aligned:
                self._phase = 'align_rot'
            elif angles_aligned:
                self._phase = 'align_eucl'

            if self._phase == 'connect':
                # give reward for getting alignment right
                aligned_rew = self._env_config['aligned_rew']
                connect = action[14]
                if connect > 0:
                    connect_rew +=  5 * self._env_config['connect_rew']

            elif self._phase == 'align_rot':
                # First phase: bring angle distance close together
                # give bonus for being done with align_eucl
                aligned_rew = self._env_config['aligned_rew']/10
                # give rew for making angular dist between sites 1
                site_up_diff = rot_dist_up - self._prev_rot_dist_up
                if not abs(site_up_diff) < 0.01:
                    site_up_rew = self._env_config['site_up_rew'] * site_up_diff

            elif self._phase == 'align_eucl':
                # Second phase: bring eucl distance close
                aligned_rew = self._env_config['aligned_rew']/10
                # give rew for minimizing eucl distance between sites
                site_dist_diff = self._prev_pos_dist - pos_dist
                if not abs(site_dist_diff) < 0.01:
                    site_dist_rew = self._env_config['site_dist_rew'] * site_dist_diff

            elif self._phase == 'align_eucl_2':
                # Third phase: bring sites close together
                # give reward for getting alignment right
                aligned_rew = self._env_config['aligned_rew']/5
                # give reward for minimizing eucl distance between sites
                site_dist_diff = self._prev_site_dist - site_dist
                if not abs(site_dist_diff) < 0.01:
                    site_dist_rew = self._env_config['site_dist_rew'] * site_dist_diff


            self._prev_site_dist = site_dist
            self._prev_pos_dist = pos_dist
            self._prev_rot_dist_up = rot_dist_up
        elif (not holding_top and self._top_picked) or (not holding_leg and self._leg_picked):
            # give penalty for dropping top or leg
            #pick_rew = -2
            done = True

        info['phase'] = self._phase
        info['leg_picked'] = self._leg_picked
        info['top_picked'] = self._top_picked
        info['pick_rew'] = pick_rew
        info['site_dist_rew'] = site_dist_rew
        info['site_up_rew'] = site_up_rew
        info['aligned_rew'] = aligned_rew
        info['connect_rew'] = connect_rew
        info['success_rew'] = success_rew
        info['ctrl_penalty'] = ctrl_penalty
        info['c0_ctrl_penalty'] = c0_ctrl_penalty
        info['c1_ctrl_penalty'] = c1_ctrl_penalty
        info['rot_dist_up'] = rot_dist_up
        info['site_dist'] = site_dist
        info['pos_dist'] = pos_dist

        rew = pick_rew + site_dist_rew + site_up_rew + connect_rew + \
                + aligned_rew + success_rew + ctrl_penalty
        return rew, done, info


def main():
    from config import create_parser
    parser = create_parser(env="FurnitureCursorToyTableEnv")
    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Cursor environment
    env = FurnitureCursorToyTableEnv(config)
    env.run_manual(config)


if __name__ == "__main__":
    main()
