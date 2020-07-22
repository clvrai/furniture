import numpy as np
from tqdm import tqdm
import time 
import math
import yaml
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from config import create_parser
from env.furniture_sawyer import FurnitureSawyerEnv
from env.models import background_names, furniture_name2id, furniture_xmls
from util.logger import logger
from util.video_recorder import VideoRecorder
import env.transform_utils as T

class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

class FurnitureSawyerGenEnv(FurnitureSawyerEnv):
    """
    Sawyer environment for assemblying furniture programmatically
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.record_demo = True
        super().__init__(config)

        self._phase = None
        self._num_connected_prev = 0
        self._config.furniture_id = furniture_name2id[config.furniture_name]
        self._phases = ['xy_move_g', 'align_g', 'z_move_g',
                        'move_grip_safepos', 'xy_move_t', 'align_conn',
                        'xy_move_conn', 'z_move_conn', 'align_conn_fine',
                        'z_move_conn_fine', 'move_nogrip_safepos', 'part_done']

        self._phase_noise = {
            'xy_move_g': [-self._config.furn_xyz_rand, self._config.furn_xyz_rand],
            'xy_move_t': [-self._config.furn_xyz_rand, self._config.furn_xyz_rand],
            'move_grip_safepos': [0, 2*self._config.furn_xyz_rand],
            'move_nogrip_safepos': [0, 2*self._config.furn_xyz_rand]
        }
        # self._phase_noise = {
        #     'xy_move_g': [-self._config.furn_xyz_rand/2, self._config.furn_xyz_rand/2],
        #     'xy_move_t': [-self._config.furn_xyz_rand/2, self._config.furn_xyz_rand/2],
        #     'move_grip_safepos': [0, self._config.furn_xyz_rand],
        #     'move_nogrip_safepos': [0, self._config.furn_xyz_rand]
        # }
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
        4. move_grip_safepos:
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
        11. move_nogrip_safepos:
                release gripper and move up to nogrip_safepos
        12  part_done:
                set part_done = True, and part is connected

        """

    def get_random_noise(self, phase, size):
        minimum, maximum = self._phase_noise[phase]
        return self._rng.uniform(low=minimum, high=maximum, size=size)

    def norm_rot_action(self, action, cap=1):
        if 'fine' in self._phase:
            for a in range(3, 7):
                if 0 < abs(action[a]) < self.min_rot_act_fine:
                    action[a] *= self.min_rot_act_fine / abs(action[a])
        else:
            for a in range(3, 7):
                if 0 < abs(action[a]) < self.min_rot_act:
                    action[a] *= self.min_rot_act / abs(action[a])
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
        if (
            self._num_connected == self._success_num_conn
            and len(self._object_names) > 1
        ):
            self._success = True

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


    def align_gripsites(self, gripvec, gbodyvec):
        if T.angle_between(-gripvec, gbodyvec) < T.angle_between(gripvec, gbodyvec):
            gripvec = -gripvec
        xyaction = T.angle_between2D(gripvec, gbodyvec)
        return xyaction

        # if abs(T.angle_between2D(-gripvec, gbodyvec)) < abs(T.angle_between2D(gripvec, gbodyvec)):
        #     gripvec = -gripvec
        # if T.angle_between(-gripvec, gbodyvec) < T.angle_between(gripvec, gbodyvec):
        #     gripvec = -gripvec
        # xyaction = self.align2D(gripvec, gbodyvec)
        return xyaction

    def get_closest_xy_fwd(self, allowed_angles, gconn, tconn):
            if len(allowed_angles) == 0:
                #no need for xy-alignment, all angles are acceptable
                return self._get_forward_vector(gconn)[0:2]
            # get closest forward vector
            gfwd = self._get_forward_vector(gconn)[0:2]
            tfwd = self._get_forward_vector(tconn)[0:2]
            min_angle = min(abs(T.angle_between2D(gfwd, tfwd)), abs((2*np.pi)+T.angle_between2D(gfwd, tfwd)))
            min_all_angle = 0
            min_tfwd = tfwd
            for angle in allowed_angles:
                tfwd_rotated = T.rotate_vector2D(tfwd, angle*(np.pi/180))
                xy_angle = T.angle_between2D(gfwd, tfwd_rotated)
                if np.pi <= xy_angle < 2*np.pi:
                    xy_angle = 2*np.pi - xy_angle
                elif -(2*np.pi) <= xy_angle < -np.pi:
                    xy_angle = 2*np.pi + xy_angle
                if abs(xy_angle) < min_angle: 
                    min_angle = abs(xy_angle)
                    min_tfwd = tfwd_rotated.copy()
                    min_all_angle = angle
            return min_tfwd

    def align2D(self, vec, targetvec):
        if abs(vec[0]) + abs(vec[1]) < 0.5:
            return 0 #unlikely current orientation allows for helpful rotation action
        angle = T.angle_between2D(vec, targetvec)
        # move in direction that gets closer to closest of (-2pi, 0, or 2pi)
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
            1. Only works for furniture with vertical connections sites
            2. Once any collision occurs, impossible to recover
            3. Sawyer arm sometimes hits table and rotates awkwardly
        '''
        with open('demos/recipes/' + self._config.furniture_name +'.yaml', 'r') as stream:
            p = yaml.load(stream, Loader=PrettySafeLoader)

        self._success_num_conn = None

        self.min_rot_act = p['min_rot_act']
        self.min_rot_act_fine = p['min_rot_act_fine']
        recipe = p['recipe']
        grip_angles = None
        if 'grip_angles' in p.keys():
            grip_angles = p['grip_angles'] 

        #Sawyer, two_finger_gripper sites
        griptip_site = 'griptip_site'
        gripbase_site = 'right_gripper_base_collision'
        grip_site = 'grip_site'
        
        z_down_prev = None
        xy_angle = None
        z_move_g_prev = None
        failed = False
        ground_offset = 0.0001
        self._config.max_episode_steps = p['max_success_steps'] + 1
        n_successful_demos = 0
        n_failed_demos = 0

        with tqdm(total=n_demos) as pbar:
            while n_successful_demos < n_demos:
                ob = self.reset()
                if 'num_connects' in p.keys():
                    self._success_num_conn = p['num_connects']
                else:
                    self._success_num_conn = len(self._object_names) - 1

                self._used_connsites = set()
                for j in range(len(recipe)):
                    grip_safepos = p['grip_safepos'][j]
                    nogrip_safepos = p['nogrip_safepos'][j]
                    safepos_count = 0
                    t_fwd = None
                    phase_num = 0
                    twice = False
                    self._phase = self._phases[phase_num]
                    self._part_done = False
                    gbody_name, tbody_name = recipe[j]
                    gconn_names, tconn_names = self.get_connsites(gbody_name, tbody_name)
                    grip_pos = self._get_pos(grip_site).copy()
                    noise = None
                    if p['use_closest']:
                        gconn = self.get_closest_connsite(gconn_names, grip_pos) #'leg-top,0,90,180,270,conn_site3' 
                    else:
                        gconn = self.get_furthest_connsite(gconn_names, grip_pos) #'leg-top,0,90,180,270,conn_site3' 
                    g_pos = self._get_pos(gbody_name)
                    allowed_angles = [float(x) for x in gconn.split(",")[1:-1] if x]
                    for i in range(len(recipe)):
                        g_l = gbody_name + '_ltgt_site' + str(i)     #gripped body left gripsite
                        g_r = gbody_name + '_rtgt_site' + str(i)    #gripped body right gripsite
                        if  g_l in self._used_connsites or g_r in self._used_connsites:
                            continue
                        else:
                            self._used_connsites.add(g_l)
                            self._used_connsites.add(g_r)
                            break
                    if self._config.render:
                        self.render()
                    if self._config.record_vid:
                        self.vid_rec.capture_frame(self.render("rgb_array")[0])
                    while not self._part_done:
                        action = np.zeros((8,))

                        if self._phase == 'xy_move_g':
                            grip_pos = self._get_pos(grip_site).copy() 
                            g_pos = (self._get_pos(g_l)+self._get_pos(g_r))/2
                            if noise is None:
                                noise = self.get_random_noise(self._phase, 2)
                            d = noise + (g_pos[0:2]-grip_pos[0:2]) 
                            if abs(d[0]) > p['eps'] or abs(d[1]) > p['eps']:
                                if abs(d[0]) > p['eps']:
                                    action[0] = d[0]
                                if abs(d[1]) > p['eps']:
                                    action[1] = d[1]
                            else:
                                phase_num+=1
                                noise = None

                        elif self._phase == 'align_g':
                            action[6] = -1
                            if grip_angles is None or grip_angles[j] is not None:
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
                                rot_action = [0 if abs(act) < p['rot_eps'] else act for act in rot_action]
                                if rot_action == [0,0,0]:
                                    grip_pos = self._get_pos(grip_site).copy()
                                    g_pos = (self._get_pos(g_l)+self._get_pos(g_r))/2
                                    d = (g_pos[0:2]-grip_pos[0:2])
                                    if abs(d[0]) > p['eps'] or abs(d[1]) > p['eps']:
                                        if abs(d[0]) > p['eps']:
                                            action[0] = d[0]
                                        if abs(d[1]) > p['eps']:
                                            action[1] = d[1]
                                    else:
                                        phase_num+=1
                                else:
                                    action[3:6] = rot_action
                            else:
                                phase_num+=1


                        elif self._phase == 'z_move_g':
                            action[6] = -1
                            grip_pos = self._get_pos(grip_site).copy()
                            grip_tip = self._get_pos(griptip_site)
                            g_pos = (self._get_pos(g_l)+self._get_pos(g_r))/2
                            d = (g_pos)-grip_pos
                            if z_down_prev is None:
                                z_down_prev = grip_tip[2] + ground_offset
                            if abs(d[2]) > p['eps'] and grip_tip[2] < z_down_prev:
                                action[0:3] = d
                                z_down_prev = grip_tip[2].copy() - ground_offset
                            else:
                                phase_num+=1
                                z_down_prev = None

                        elif self._phase == 'move_grip_safepos':
                            action[6] = 1
                            if safepos_count >= len(grip_safepos):
                                safepos_count = 0
                                if len(grip_safepos) >= 3:
                                # skip xy_move_t if specified movements > 3
                                    phase_num += 2
                                else:
                                    phase_num+=1
                                gconn_pos = self.sim.data.get_site_xpos(gconn)
                                if p['use_closest']:
                                    tconn = self.get_closest_connsite(tconn_names, gconn_pos) 
                                else:                            
                                    tconn = self.get_furthest_connsite(tconn_names, gconn_pos) 
                                tconn_pos = self.sim.data.get_site_xpos(tconn)
                            else:
                                # move to safepos in the order specified
                                d = np.zeros((3,))
                                gripbase_pos = self._get_pos(gripbase_site)
                                axis, val = grip_safepos[safepos_count]
                                if noise is None:
                                    noise = self.get_random_noise(self._phase, 1)
                                if axis == 'z':
                                    d[2] = noise + val - gripbase_pos[2]
                                elif axis == 'y':
                                    d[1] = noise + val - gripbase_pos[1]
                                else:
                                    d[0] = noise + val - gripbase_pos[0]
                                if np.sum(np.absolute(d)) > p['eps']:
                                    action[0:3] = d
                                else:
                                    safepos_count += 1
                                    noise = None

                        elif self._phase == 'xy_move_t':
                            action[6] = 1
                            grip_pos = self._get_pos(grip_site).copy()
                            if noise is None:
                                noise = self.get_random_noise(self._phase, 2)                        
                            d = noise + (tconn_pos[0:2] - grip_pos[0:2])
                            if abs(d[0]) > p['eps'] or abs(d[1]) > p['eps']:
                                if abs(d[0]) > p['eps']:
                                    action[0] = d[0]
                                if abs(d[1]) > p['eps']:
                                    action[1] = d[1]
                            else:
                                phase_num+=1
                                noise = None

                        elif self._phase == 'align_conn':
                            action[6] = 1
                            g_up = self._get_up_vector(gconn).copy()
                            t_up = self._get_up_vector(tconn).copy()
                            yz_ac = self.align2D(np.array([g_up[1], g_up[2]]), np.array([t_up[1], t_up[2]]))
                            xz_ac = self.align2D(np.array([g_up[0], g_up[2]]), np.array([t_up[0], t_up[2]]))
                            rot_action = [0, yz_ac, xz_ac]
                            rot_action = [0 if abs(act) < p['rot_eps'] else act for act in rot_action]
                            if rot_action == [0,0,0]:
                                g_xy_fwd = self._get_forward_vector(gconn).copy()
                                if t_fwd is None:
                                    t_fwd = self.get_closest_xy_fwd(allowed_angles, gconn, tconn)
                                t_xy_fwd = t_fwd[0:2]
                                xy_ac = self.align2D(g_xy_fwd, t_xy_fwd)
                                xy_ac = 0 if abs(xy_ac) < p['rot_eps'] else xy_ac
                                if xy_ac == 0:
                                    phase_num+=1
                                else:
                                    action[3] = -xy_ac
                            else:
                                action[3:6] = rot_action

                        elif self._phase == 'xy_move_conn':
                            action[6] = 1
                            gconn_pos = self.sim.data.get_site_xpos(gconn)
                            tconn_pos = self.sim.data.get_site_xpos(tconn)
                            d = (tconn_pos[0:2] - gconn_pos[0:2])
                            if abs(d[0]) > p['eps'] or abs(d[1]) > p['eps']:
                                if abs(d[0]) > p['eps']:
                                    action[0] = d[0]
                                if abs(d[1]) > p['eps']:
                                    action[1] = d[1]
                            else:
                                phase_num+=1
                        
                        elif self._phase == 'z_move_conn':
                            action[6] = 1
                            gconn_pos = self.sim.data.get_site_xpos(gconn)
                            tconn_pos = self.sim.data.get_site_xpos(tconn)
                            d = tconn_pos - gconn_pos
                            d[2] += p['z_conn_dist']
                            if abs(d[2]) > p['z_finedist']:
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
                            rot_action = [0 if abs(act) < p['rot_eps_fine'] else act for act in rot_action]
                            if rot_action == [0,0,0]:
                                g_xy_fwd = self._get_forward_vector(gconn).copy()
                                if t_fwd is None:
                                    t_fwd = self.get_closest_xy_fwd(allowed_angles, gconn, tconn)
                                t_xy_fwd = t_fwd[0:2]
                                xy_ac = self.align2D(g_xy_fwd, t_xy_fwd)
                                xy_ac = 0 if abs(xy_ac) < p['rot_eps_fine'] else xy_ac
                                if xy_ac == 0:
                                    d = (tconn_pos[0:2] - gconn_pos[0:2])
                                    if abs(d[0]) > p['eps_fine'] or abs(d[1]) > p['eps_fine']:
                                        if abs(d[0]) > p['eps_fine']:
                                            action[0] = d[0]
                                        if abs(d[1]) > p['eps_fine']:
                                            action[1] = d[1]
                                    else:
                                        t_fwd = None
                                        phase_num+=1
                                else:
                                    action[3] = -xy_ac
                            else:
                                action[3:6] = rot_action

                        elif self._phase == 'z_move_conn_fine':
                            action[6] = 1
                            gconn_pos = self.sim.data.get_site_xpos(gconn)
                            tconn_pos = self.sim.data.get_site_xpos(tconn)
                            d = tconn_pos - gconn_pos
                            d[2] += p['z_conn_dist']
                            if abs(d[2]) > p['eps']:
                                action[0:3] = d[0:3]
                                action[2] = action[2] / p['fine_magnitude']
                            else:
                                action[7] = 1 
                                if 'connect_twice' in p.keys() and j in p['connect_twice'] and twice is False:
                                    twice = True
                                else:
                                    twice = False
                                    phase_num+=1

                        elif self._phase == 'move_nogrip_safepos':
                            action[6] = -1
                            if safepos_count >= len(nogrip_safepos):
                                safepos_count = 0
                                phase_num+=1
                            else:
                                # move to safepos in the order specified
                                d = np.zeros((3,))
                                gripbase_pos = self._get_pos(gripbase_site)
                                axis, val = nogrip_safepos[safepos_count]
                                if noise is None:
                                    noise = self.get_random_noise(self._phase, 1)
                                if axis == 'z':
                                    d[2] = val - gripbase_pos[2]
                                elif axis == 'y':
                                    d[1] = val - gripbase_pos[1]
                                else:
                                    d[0] = val - gripbase_pos[0]
                                if np.sum(np.absolute(d)) > p['eps']:
                                    action[0:3] = d
                                else:
                                    safepos_count += 1
                                    noise = None
                        self._phase = self._phases[phase_num]
                        action[0:3] = p['lat_magnitude'] * action[0:3]
                        action[3:6] = p['rot_magnitude'] * action[3:6]
                        action = self.norm_rot_action(action)
                        action = self._cap_action(action)
                        ob, reward, _, info = self.step(action)
                        if self._config.render:
                            self.render()
                        if self._config.record_vid:
                            self.vid_rec.capture_frame(self.render("rgb_array")[0])
                        if self._episode_length > p['max_success_steps']:
                        # if demo generation fails
                            failed = True
                            break
                        if self._success:
                        # if assembly finished
                            break

                    if self._part_done:
                        self._used_connsites.add(gconn)
                        self._used_connsites.add(tconn)

                    if failed:
                        print('failed to assemble')
                        if self._config.record_vid:
                            self.vid_rec.close(success=False)
                        self._demo.reset()
                        failed = False
                        n_failed_demos += 1
                        break
                        
                    elif self._success:
                        print('assembled', self._config.furniture_name, 'in', self._episode_length, 'steps!')
                        if self._config.start_count:
                            demo_count = self._config.start_count + n_successful_demos
                            fname = self.file_prefix + "{:04d}.pkl".format(demo_count)
                            self._demo.save(self.file_prefix, count=demo_count)
                        else:
                            self._demo.save(self.file_prefix)
                        n_successful_demos += 1
                        pbar.update(1)
                        if self._config.record_vid:
                            self.vid_rec.close()
                        break
        print('n_failed_demos', n_failed_demos)


def main():

    parser = create_parser(env="FurnitureSawyerGenEnv")
    config, unparsed = parser.parse_known_args()

    env = FurnitureSawyerGenEnv(config)
    #env.run_manual(config)
    env.generate_demos(100)
    #env.run_demo(config)

if __name__ == "__main__":
    main()
