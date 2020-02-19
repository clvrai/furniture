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
            "connect_rew": config.connect_rew,
            "success_rew": config.success_rew,
            "pick_rew": config.pick_rew,
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
        Returns the DoF of the curosr agent.
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

        up1 = self._get_up_vector(top_site_name)
        up2 = self._get_up_vector(leg_site_name)
        forward1 = self._get_forward_vector(top_site_name)
        forward2 = self._get_forward_vector(leg_site_name)
        pos_dist = T.l2_dist(top_site_xpos[:3], leg_site_xpos[:3])
        rot_dist_up = T.cos_dist(up1, up2)

        self._prev_pos_dist = pos_dist
        self._prev_rot_dist_up = rot_dist_up

    def _place_objects(self):
        """
        Returns fixed initial position and rotations of the toy table.
        The first case has the table top on the left and legs on the right.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        pos_init = [[ 0.21250838, -0.1163671 ,  0.02096991], [-0.30491682, -0.09045364,  0.03429339],[ 0.38134436, -0.11249256,  0.02096991],[ 0.12432612, -0.13662719,  0.02096991],[ 0.29537311, -0.12992911,  0.02096991]]
        quat_init = [[0.706332  , 0.70633192, 0.03309327, 0.03309326], [ 0.00000009, -0.99874362, -0.05011164,  0.00000002], [ 0.70658149,  0.70706735, -0.00748174,  0.0272467 ], [0.70610751, 0.7061078 , 0.03757641, 0.03757635], [0.70668613, 0.70668642, 0.02438253, 0.02438249]]

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

    def _compute_reward(self, action):
        """
        Two stage reward.
        The first stage gives reward for picking up a table top and a leg.

        The next stage gives reward for bringing the leg connection site close to the table
        connection site.
        """
        rew = pick_rew = site_dist_rew = site_up_rew = connect_rew = success_rew = 0
        done = self._num_connected > 0
        info = {}
        holding_top = self._cursor_selected[0] == '4_part4'
        holding_leg = self._cursor_selected[1] == '2_part2'
        # cursor 0 select table top
        if holding_top and not self._top_picked:
            pick_rew += self._env_config['pick_rew']
            self._top_picked = True
            logger.debug('picked up tabletop')
        # cursor 1 select table leg
        if holding_leg and not self._leg_picked:
            pick_rew += self._env_config['pick_rew']
            self._leg_picked = True
            logger.debug('picked up leg')

        if self._num_connected > 0:
            success_rew = self._env_config['success_rew']
            logger.debug(f'success rew: {success_rew}')

        # if parts are in hand, then give reward for moving parts closer
        elif holding_top and holding_leg:
            top_site_name = "top-leg,,conn_site4"
            leg_site_name = "leg-top,,conn_site4"
            top_site_xpos = self._site_xpos_xquat(top_site_name)
            leg_site_xpos = self._site_xpos_xquat(leg_site_name)

            up1 = self._get_up_vector(top_site_name)
            up2 = self._get_up_vector(leg_site_name)
            forward1 = self._get_forward_vector(top_site_name)
            forward2 = self._get_forward_vector(leg_site_name)
            pos_dist = T.l2_dist(top_site_xpos[:3], leg_site_xpos[:3])
            rot_dist_up = T.cos_dist(up1, up2)

            project1_2 = np.dot(up1, T.unit_vector(leg_site_xpos[:3] - top_site_xpos[:3]))
            project2_1 = np.dot(up2, T.unit_vector(top_site_xpos[:3] - leg_site_xpos[:3]))

            #logger.debug(f'pos_dist: {pos_dist:.2f}, '+f'rot_dist_up: {rot_dist_up:.2f}, '+
            #            f'project: {project1_2:.2f}, {project2_1:.2f}')

            # if parts are close together, press connect
            if pos_dist < 0.03 and rot_dist_up > self._env_config['rot_dist_up']:
                connect = action[14]
                if connect > 0:
                    connect_rew = self._env_config['connect_rew']
                logger.debug(f'connect rew: {connect_rew}')
            else: # else bring parts closer together
                # give rew for minimizing eucl distance between sites
                site_dist_diff = self._prev_pos_dist - pos_dist
                if not abs(site_dist_diff) < 0.01:
                    site_dist_rew = self._env_config['site_dist_rew'] * site_dist_diff
                logger.debug(f'site_dist_rew: {site_dist_rew}')
                self._prev_pos_dist = pos_dist

                # give rew for making angular dist between sites 1
                site_up_diff = rot_dist_up - self._prev_rot_dist_up
                if not abs(site_up_diff) < 0.01:
                    site_up_rew = self._env_config['site_up_rew'] * site_up_diff
                logger.debug(f'site_up_rew: {site_up_rew}')
                self._prev_rot_dist_up = rot_dist_up

        elif (not holding_top and self._top_picked) or (not holding_leg and self._leg_picked):
            # give penalty for dropping top or leg
            #pick_rew = -2
            done = True


        rew = pick_rew + site_dist_rew + site_up_rew + connect_rew + success_rew
        return rew, done, info


def main():
    import argparse
    import config.furniture as furniture_config
    from util import str2bool

    parser = argparse.ArgumentParser()
    furniture_config.add_argument(parser)

    # change default config for Cursors
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)

    parser.set_defaults(render=True)

    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Cursor environment
    env = FurnitureCursorToyTableEnv(config)
    env.run_manual(config)


if __name__ == "__main__":
    main()
