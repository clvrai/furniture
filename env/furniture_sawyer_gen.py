import numpy as np
from tqdm import tqdm
import time 
import math

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from config import create_parser
from env.furniture_sawyer import FurnitureSawyerEnv
from env.models import background_names, furniture_name2id, furniture_xmls
from util.logger import logger
from util.video_recorder import VideoRecorder
import env.transform_utils as T

class FurnitureSawyerGenEnv(FurnitureSawyerEnv):
    """
    Sawyer environment for assemblying furniture programmatically
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.max_episode_steps = 4000
        config.record_demo = True
        super().__init__(config)

        self._phase = None
        self._num_connected_prev = 0
        self._config.furniture_id = furniture_name2id[config.furniture_name]
        self._phases = ['xy_move_g', 'align_g', 'z_move_g',
                        'z_move_safepos', 'xy_move_t', 'align_conn',
                        'xy_move_conn', 'z_move_conn', 'align_conn_fine',
                        'z_move_conn_fine', 'z_return_safepos', 'part_done']
        """
        Abbreviations:
        grip ~ gripper
        g ~ gripped body 
        t ~ target body
        conn ~ connection site 

        Phases Descrption:
        1. xy_move_g:
                move to xy pos. of gripper with gbody
        2. align_g:
                move to xyplane rot. of gripper vector with gbody_gripsites,
                rotate up-vector ofgripper to down vector (0,0,-1)
        3. z_move_g:
                move gripper down to z-pos of gbody
        4. z_move_safepos:
                grip gbody then move up to the grip_safepos
        5. xy_move_t:
                move to xy pos of gripper with tbody
        6. align_conn:
                rotate up-vector of gbody connsite to tbody connsite up-vector
        7. xy_move_conn:
                move to xy position of gbody connsite w.r.t. tbody connsite
        8. z_move_conn:
                move to xyz position of gbody connsite w.r.t. tbody connsite
        9. align_conn_fine:
                finely rotate up-vector of gbody connsite to up-vector of tbody connsite
        10. z_move_conn_fine:
                finely move to xyz position of gbody connsite w.r.t. tbody connsite,
                then try connecting
        11. z_return_safepos:
                release gripper and move up to nogrip_safepos
        12  part_done:
                set part_done = True, and part is connected

        """

    def norm_rot_action(self, action, cap=1):
        if 'fine' in self._phase:
            for a in range(3, 7):
                if 0 < abs(action[a]) < self.min_rot_action_fine:
                    action[a] *= self.min_rot_action_fine / abs(action[a])
        else:
            for a in range(3, 7):
                if 0 < abs(action[a]) < self.min_rot_action:
                    action[a] *= self.min_rot_action / abs(action[a])
        return action

    def _cap_action(self, action, cap=1):
        for a in range(len(action)):
            if action[a] > cap:
                action[a] = cap
            elif action[a] < -cap:
                action[a] = -cap
        return action

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        ob, reward, done, info = super(FurnitureSawyerEnv, self)._step(a)
        if self._num_connected > self._num_connected_prev:
            self._conn_success = True
            self._num_connected_prev = self._num_connected

        self._part_done = self._phase == 'part_done'
        # info["ac"] = a
        return ob, reward, done, info

    def get_bodyiterator(self, bodyname):
        for body in self.mujoco_objects[bodyname].root.find("worldbody"):
            if 'name' in body.attrib and bodyname == body.attrib['name']:
                return body.getiterator()
        return None 

    def _get_groupname(self, bodyname):
        bodyiterator = self.get_bodyiterator(bodyname)
        for child in bodyiterator:
            if child.tag == "site":
                if 'name' in child.attrib and "conn_site" in child.attrib['name']:
                    return child.attrib['name'].split('-')[0]
        return None

    def get_connsites(self, gbody_name, tbody_name):
        gripbody_connsite, tbody_connsite = [], []
        group1 = self._get_groupname(gbody_name)
        group2 = self._get_groupname(tbody_name)
        iter1 = self.get_bodyiterator(gbody_name)
        iter2 = self.get_bodyiterator(tbody_name)
        griptag = group1 + '-' + group2
        tgttag = group2 + '-' + group1
        for child in iter1:
            if child.tag == "site":
                if ('name' in child.attrib and "conn_site" in child.attrib['name']
                    and griptag in child.attrib['name'] 
                    and child.attrib['name'] not in self._used_connsites):
                    gripbody_connsite.append(child.attrib['name'])
        for child in iter2:
            if child.tag == "site":
                if ('name' in child.attrib and "conn_site" in child.attrib['name']
                    and tgttag in child.attrib['name']
                    and child.attrib['name'] not in self._used_connsites):
                    tbody_connsite.append(child.attrib['name'])
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

    # def get_closest_connsite(self, conn_sites, gripper_pos):
    #     closest = None
    #     min_dist = None
    #     for name in conn_sites:
    #         pos = self.sim.data.get_site_xpos(name)
    #         dist = T.l2_dist(gripper_pos, pos)
    #         if closest is None:
    #             closest = name
    #             min_dist = dist
    #         else:
    #             if dist < min_dist:
    #                 closest = name
    #                 min_dist = dist
    #     return closest


    def align_gripsites(self, gripvec, gbodyvec):
        if T.angle_between(-gripvec, gbodyvec) < T.angle_between(gripvec, gbodyvec):
            gripvec = -gripvec
        xyaction = T.angle_between2D(gripvec, gbodyvec)
        return xyaction

    def align2D(self, vec, targetvec):
        angle = T.angle_between2D(vec, targetvec)
        if -(2*np.pi) < angle <= -np.pi:
            action = -(2*np.pi + angle)
        if -np.pi < angle <= 0:
            action = -angle
        if 0 < angle <= np.pi:
             action = -angle
        if np.pi < angle <= 2*np.pi:
             action = 2*np.pi - angle
        return action

    def generate_demos(self, n_demos):

        '''
        Issues:
            1. Only works for toy_table 
            2. Once any collision occurs, impossible to recover
            3. Sawyer arm sometimes hits table and rotates awkwardly
            4. change fine_factor to some threshold capping
        '''
        # toytable xyz parameters
        fine_factor = 4
        z_finedist = 0.11                   # distance between connsites at which to start fine adjustment
        z_conn_dist = -0.02                 # distance between connsites at which to connect
        lat_magnitude_factor = 20           # keep movespeed constant at 0.025
        z_nogrip_safepos = 0.44             # z height to raise gripper to, to ensure no collisions
        z_grip_safepos = 0.4
        epsilon = 0.01                      # max acceptable x,y,z difference
        epsilon_fine = 0.001                # max acceptable x,y,z difference

        #toytable rot parameters
        rot_magnitude_factor = 0.2
        rot_epsilon = 0.05
        rot_epsilon_fine = 0.01
        self.min_rot_action = 0.05
        self.min_rot_action_fine = 0.01

        #toytable parameters
        max_success_steps = 2200            # max # of steps a successful demo will take
        self._n_connects = 5                # how many times parts must be connected for completion
        recipe = [('0_part0', '4_part4'),('1_part1', '4_part4'),
                ('2_part2', '4_part4'),('3_part3', '4_part4')]
        #Sawyer, two_finger_gripper sites
        griptip_site = 'griptip_site'
        gripbase_site = 'right_gripper_base_collision'
        grip_site = 'grip_site'

        for demo_num in tqdm(range(n_demos)):
            ob = self.reset()
            self._used_connsites = set()
            for j in range(len(recipe)):
                failed = False
                z_move_g_prev = None
                phase_num = 0
                self._phase = self._phases[phase_num]
                self._part_done = False
                gbody_name, tbody_name = recipe[j]
                gconn_names, tconn_names = self.get_connsites(gbody_name, tbody_name)
                grip_pos = self._get_pos(grip_site).copy()
                gconn = self.get_furthest_connsite(gconn_names, grip_pos)
                g_pos = self._get_pos(gbody_name)
                g_l = gbody_name + '_ltgt_site'     #gripped body left gripsite
                g_r = gbody_name + '_rtgt_site'     #gripped body right gripsite

                if self._config.render:
                    self.render()
                if self._config.record_vid:
                    self.vid_rec.capture_frame(self.render("rgb_array")[0])
                while not self._part_done:
                    action = np.zeros((8,))

                    if self._phase == 'xy_move_g':
                        grip_pos = self._get_pos(grip_site).copy()
                        g_pos = (self._get_pos(g_l)+self._get_pos(g_r))/2
                        d = (g_pos[0:2]-grip_pos[0:2])
                        if abs(d[0]) > epsilon or abs(d[1]) > epsilon:
                            if abs(d[0]) > epsilon:
                                action[0] = d[0]
                            if abs(d[1]) > epsilon:
                                action[1] = d[1]
                        else:
                            phase_num+=1

                    elif self._phase == 'align_g':
                        # align gripper fingers with grip sites
                        xy_gripvec = self._get_forward_vector(grip_site).copy()
                        xy_gvec = self._get_pos(g_r).copy() - self._get_pos(g_l).copy()
                        xy_gripvec = xy_gripvec[0:2]
                        xy_gvec = xy_gvec[0:2]
                        xy_ac = self.align_gripsites(xy_gripvec, xy_gvec)
                        # point gripper z downwards
                        gripvec =  self._get_up_vector(grip_site).copy()
                        target = np.array([0, -1])
                        yz_ac = self.align2D(np.array([gripvec[1], gripvec[2]]), target)
                        xz_ac = self.align2D(np.array([gripvec[0], gripvec[2]]), target)
                        rot_action = [xy_ac, yz_ac, xz_ac]
                        rot_action = [0 if abs(act) < rot_epsilon_fine else act for act in rot_action]
                        if rot_action == [0,0,0]:
                            grip_pos = self._get_pos(grip_site).copy()
                            g_pos = (self._get_pos(g_l)+self._get_pos(g_r))/2
                            d = (g_pos[0:2]-grip_pos[0:2])
                            if abs(d[0]) > epsilon or abs(d[1]) > epsilon:
                                if abs(d[0]) > epsilon:
                                    action[0] = d[0]
                                if abs(d[1]) > epsilon:
                                    action[1] = d[1]
                            else:
                                phase_num+=1
                        else:
                            action[3:6] = rot_action

                    elif self._phase == 'z_move_g':
                        action[6] = -1
                        grip_pos = self._get_pos(grip_site).copy()
                        g_pos = (self._get_pos(g_l)+self._get_pos(g_r))/2
                        d = (g_pos)-grip_pos
                        if abs(d[2]) > epsilon:
                            action[0:3] = d
                        else:
                            phase_num+=1

                    elif self._phase == 'z_move_safepos':
                        action[6] = 1
                        gripbase_pos = self._get_pos(gripbase_site)
                        d = z_grip_safepos - gripbase_pos[2]
                        if abs(d) > epsilon:
                            action[2] = d
                        else:
                            phase_num+=1
                            tconn = self.get_furthest_connsite(tconn_names, grip_pos)
                            tconn_pos = self.sim.data.get_site_xpos(tconn)

                    elif self._phase == 'xy_move_t':
                        action[6] = 1
                        grip_pos = self._get_pos(grip_site).copy()
                        d = (tconn_pos[0:2] - grip_pos[0:2])
                        if abs(d[0]) > epsilon or abs(d[1]) > epsilon:
                            if abs(d[0]) > epsilon:
                                action[0] = d[0]
                            if abs(d[1]) > epsilon:
                                action[1] = d[1]
                        else:
                            phase_num+=1

                    elif self._phase == 'align_conn':
                        action[6] = 1
                        g_up = self._get_up_vector(gconn).copy()
                        t_up = self._get_up_vector(tconn).copy()
                        yz_ac = self.align2D(np.array([g_up[1], g_up[2]]), np.array([t_up[1], t_up[2]]))
                        xz_ac = self.align2D(np.array([g_up[0], g_up[2]]), np.array([t_up[0], t_up[2]]))
                        rot_action = [0, yz_ac, xz_ac]
                        rot_action = [0 if abs(act) < rot_epsilon else act for act in rot_action]
                        if rot_action == [0,0,0]:
                            phase_num+=1
                        else:
                            action[3:6] = rot_action

                    elif self._phase == 'xy_move_conn':
                        action[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        d = (tconn_pos[0:2] - gconn_pos[0:2])
                        if abs(d[0]) > epsilon or abs(d[1]) > epsilon:
                            if abs(d[0]) > epsilon:
                                action[0] = d[0]
                            if abs(d[1]) > epsilon:
                                action[1] = d[1]
                        else:
                            phase_num+=1
                    
                    elif self._phase == 'z_move_conn':
                        action[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        d = tconn_pos - gconn_pos
                        d[2] += z_conn_dist
                        if abs(d[2]) > z_finedist:
                            action[0:3] = d
                        else:
                            phase_num+=1

                    elif self._phase == 'align_conn_fine':
                        action[6] = 1
                        g_up = self._get_up_vector(gconn).copy()
                        t_up = self._get_up_vector(tconn).copy()
                        yz_ac = self.align2D(np.array([g_up[1], g_up[2]]), np.array([t_up[1], t_up[2]]))
                        xz_ac = self.align2D(np.array([g_up[0], g_up[2]]), np.array([t_up[0], t_up[2]]))
                        rot_action = [0, yz_ac, xz_ac]
                        rot_action = [0 if abs(act) < rot_epsilon_fine else act for act in rot_action]
                        if rot_action == [0,0,0]:
                            gconn_pos = self.sim.data.get_site_xpos(gconn)
                            tconn_pos = self.sim.data.get_site_xpos(tconn)
                            d = (tconn_pos[0:2] - gconn_pos[0:2])
                            if abs(d[0]) > epsilon_fine or abs(d[1]) > epsilon_fine:
                                if abs(d[0]) > epsilon_fine:
                                    action[0] = d[0]
                                if abs(d[1]) > epsilon_fine:
                                    action[1] = d[1]
                            else:
                                phase_num+=1
                        else:
                            action[3:6] = rot_action

                    elif self._phase == 'z_move_conn_fine':
                        action[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        d = tconn_pos - gconn_pos
                        d[2] += z_conn_dist
                        if abs(d[2]) > epsilon:
                            action[0:3] = d[0:3]
                            action[2] = action[2] / fine_factor
                        else:
                            action[7] = 1 
                            phase_num+=1

                    elif self._phase == 'z_return_safepos':
                        gripbase_pos = self._get_pos(gripbase_site)
                        action[6] = -1
                        d = (z_nogrip_safepos - gripbase_pos[2])
                        if abs(d) > epsilon:
                            action[2] = d
                        else:
                            phase_num+=1

                    self._phase = self._phases[phase_num]
                    action[0:3] = lat_magnitude_factor * action[0:3]
                    action[3:6] = rot_magnitude_factor * action[3:6]
                    action = self.norm_rot_action(action)
                    action = self._cap_action(action)
                    ob, reward, done, info = self.step(action)
                    if self._config.render:
                        self.render()
                    if self._config.record_vid:
                        self.vid_rec.capture_frame(self.render("rgb_array")[0])
                    if self._episode_length > max_success_steps:
                    # if demo generation fails restart
                        failed = True
                        break

                if self._part_done:
                    self._used_connsites.add(gconn)
                    self._used_connsites.add(tconn)

                if failed:
                    print('failed to assemble')
                    if self._config.record_vid:
                        self.vid_rec.close(success=False)
                    self._demo.reset()
                    break
                elif done:
                    print('assembled in', self._episode_length, 'steps!')
                    self._demo.save(self.file_prefix)
                    if self._config.record_vid:
                        self.vid_rec.close()
                    break



def main():

    parser = create_parser(env="FurnitureSawyerGenEnv")
    config, unparsed = parser.parse_known_args()

    env = FurnitureSawyerGenEnv(config)
    #env.run_manual(config)
    env.generate_demos(10)
    #env.run_demo(config)

if __name__ == "__main__":
    main()
