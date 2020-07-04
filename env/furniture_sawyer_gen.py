import numpy as np
from tqdm import tqdm
import time 

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


from env.furniture_sawyer import FurnitureSawyerEnv
from env.models import background_names, furniture_name2id, furniture_xmls
from util.logger import logger
from util.video_recorder import VideoRecorder
import env.transform_utils as T

class FurnitureSawyerGenEnv(FurnitureSawyerEnv):
    """
    Sawyer environment for placing a block onto table.
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
        self._phases = ['pregrip_xy_align', 'pregrip_rot_align', 'pregrip_xy_align2', 'z_down', 
                'z_finedown', 'gripped_z_up', 'xy_tgt', 'conn_align_rot', 
                'conn_align_xy', 'conn_align_z', 'conn_align_xy_fine', 
                'conn_align_z_fine', 'z_return', 'part_done'] 
        """
        Phases Descrption:

        grip ~ gripper
        g ~ gripped body 
        t ~ target body
        conn ~ connection site 

        1. pregrip_xy_align: 
                align xy pos. of gripper with gbody
        2. pregrip_rot_align:
                align xyplane rot. of gripper vector with gbody_gripsites,
                align xz, yz gripper rot. with down vector (0,0,-1)   
        3. pregrip_xy_align2:
                finely align xy pos. of gripper with gbody
        3. z_down:
                move gripper down 
        4. z_finedown:
                finely move gripper down, then grasp
        4. gripped_z_up:
                grip gbody then move up to z_safepos
        5. xy_tgt:
                align xy pos of gripper with tbody  
        6. conn_align_rot:
                align upvectors of gbody connsite and tbody connsite
        7. conn_align_xy:
                align xy position of gbody connsite with tbody connsite
        8. conn_align_z:
                align z position of gbody connsite with tbody connsite
        9. conn_align_xy_fine:
                finely align xy position of gbody connsite with tbody connsite
        10. conn_align_z_fine:
                finely align z position of gbody connsite with tbody connsite,
            then try connecting
        11. z_return:
                release gripper and move up to z_safepos
        12  part_done:
                set part_done = True, and part is connected

        """

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


    def align_plane(self, vec1, vec2, plane='xy'):
        if plane == 'xy':
            vec1[2] = 0
            vec2[2] = 0
        elif plane == 'yz':
            vec1[0] = 0
            vec2[0] = 0
        elif plane == 'xz':
            vec1[1] = 0
            vec2[1] = 0

        angle = T.angle_between(vec1, vec2)
        
        if angle < 0:
            angle2 = 2*np.pi + angle
        else:
            angle2 = angle - 2*np.pi

        if min(abs(angle), abs(angle2)) < 0.04: 
            return 0 
        if abs(angle2) < abs(angle):
            return angle2
        return angle

    def align_rot(self, gvec, tvec):
        xy_act = 0#self.align_plane(gvec.copy(), tvec.copy(), plane='xy')
        yz_act = self.align_plane(gvec.copy(), tvec.copy(), plane='yz')
        xz_act = self.align_plane(gvec.copy(), tvec.copy(), plane='xz')
        rot_action = [xy_act, yz_act, xz_act]
        return rot_action


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
        closest = None
        min_dist = None
        for name in conn_sites:
            pos = self.sim.data.get_site_xpos(name)
            dist = T.l2_dist(gripper_pos, pos)
            if closest is None:
                closest = name
                min_dist = dist
            else:
                if dist > min_dist:
                    closest = name
                    min_dist = dist
        return closest


    def generate_demos(self, n_demos):

        '''
        Issues:
            1. Only works for toy_table 
            2. rotation alignment code doesn't always properly account for direction, 
                rotational magnitude is fine
            3. Once any collision occurs, impossible to recover
            4. Sawyer arm sometimes hits table and rotates awkwardly 
        '''

        action_magnitude_factor = 20    # keep movespeed constant at 0.025
        z_safepos = 0.4                 # z height to raise gripper to, to ensure no collisions 
        z_finedist = 0.06               # distance between connsites at which to start fine adjustment 
        z_conn_dist = 0.04              # distance between connsites at which to connect
        epsilon = 0.0005                # max acceptable x,y,z difference 
        fine_fact = 2                   # action magnitude reduction for fine adjustment phases         
        n_connects = 5                  # how many times parts must be connected for completion 
        recipe = [('0_part0', '4_part4'),('1_part1', '4_part4'),
                ('2_part2', '4_part4'),('3_part3', '4_part4')]

        lgriptip_site = 'lgriptip_site'
        rgriptip_site = 'rgriptip_site'
        griptip_site = 'griptip_site'
        for demo_num in tqdm(range(n_demos)):
            ob = self.reset(self._config.furniture_id)
            self._used_connsites = set()
            for j in range(len(recipe)):
                z_down_prev = None
                phase_num = 0
                self._phase = self._phases[phase_num]
                self._part_done = False
                gbody_name, tbody_name = recipe[j]
                gconn_names, tconn_names = self.get_connsites(gbody_name, tbody_name)
                grippos = self._get_pos("grip_site")
                gconn = self.get_furthest_connsite(gconn_names, grippos)
                gbody_pos = self._get_pos(gbody_name)
                ltgt_site = gbody_name + '_ltgt_site'
                rtgt_site = gbody_name + '_rtgt_site'

                if self._config.render:
                    self.render()
                if self._config.record_vid:
                    self.vid_rec.capture_frame(self.render("rgb_array")[0])
                while not self._part_done:
                    action = np.zeros((8,))
                    grippos = self._get_pos("grip_site")
                    gbody_pos = self._get_pos(gbody_name)
                    if self._phase == 'pregrip_xy_align':
                        d = (gbody_pos[0:2]-grippos[0:2])
                        if abs(d[0]) > epsilon or abs(d[1]) > epsilon:
                            if abs(d[0]) > epsilon:
                                action[0] = d[0]  
                            if abs(d[1]) > epsilon:
                                action[1] = d[1]  
                        else:
                            phase_num+=1
                    elif self._phase == 'pregrip_rot_align':
                        xy_gvec = self._get_pos(rgriptip_site) - self._get_pos(lgriptip_site)
                        xy_tvec = self._get_pos(rtgt_site) - self._get_pos(ltgt_site)
                        xy_ac = self.align_plane(xy_gvec, xy_tvec)
                        # always orient gripper to be point down before grip attempt 
                        tvec = [0, 0, -1]  
                        gvec = self._get_pos(griptip_site) - grippos
                        rot_action = self.align_rot(gvec, tvec)
                        rot_action[0] = xy_ac
                        rot_action[1] = -rot_action[1]
                        rot_action[2] = -rot_action[2]
                        if rot_action == [0,0,0]:
                            phase_num+=1
                        else:
                            action[3:6] = rot_action
                    elif  self._phase == 'pregrip_xy_align2':
                        d = (gbody_pos[0:2]-grippos[0:2])
                        if abs(d[0]) > epsilon or abs(d[1]) > epsilon:
                            if abs(d[0]) > epsilon:
                                action[0] = d[0]  
                            if abs(d[1]) > epsilon:
                                action[1] = d[1]  
                        else:
                            phase_num+=1
                    elif self._phase == 'z_down':
                        action[6] = -1
                        d = (gbody_pos)-grippos
                        grip_tip = self._get_pos("griptip_site")
                        if z_down_prev is None:
                            z_down_prev = grip_tip[2] + epsilon
                        if grip_tip[2] < z_down_prev and abs(d[2]) > z_finedist:
                            action[0:3] = d
                            z_down_prev = grip_tip[2]
                        else:
                            phase_num+=1
                            z_down_prev = None
                    elif self._phase == 'z_finedown':
                        action[6] = -1
                        d = (gbody_pos-grippos)
                        grip_tip = self._get_pos("griptip_site")
                        if z_down_prev is None:
                            z_down_prev = grip_tip[2] + epsilon
                        if grip_tip[2] < z_down_prev:
                            action[0:3] = d / fine_fact
                            z_down_prev = grip_tip[2]
                        else:
                            phase_num+=1
                    elif self._phase == 'gripped_z_up':
                        action[6] = 1
                        d = z_safepos - grippos[2]
                        if abs(d) > epsilon:
                            action[2] = d
                        else:
                            phase_num+=1
                            tconn = self.get_furthest_connsite(tconn_names, grippos)
                            tconn_pos = self.sim.data.get_site_xpos(tconn)
                    elif self._phase == 'xy_tgt':
                        action[6] = 1
                        d = (tconn_pos[0:2] - grippos[0:2])
                        if abs(d[0]) > epsilon or abs(d[1]) > epsilon:
                            if abs(d[0]) > epsilon:
                                action[0] = d[0]  
                            if abs(d[1]) > epsilon:
                                action[1] = d[1]  
                        else:
                            phase_num+=1

                    elif self._phase == 'conn_align_rot':
                        action[6] = 1
                        g_up = self._get_up_vector(gconn).copy()
                        t_up = self._get_up_vector(tconn).copy()
                        rot_action = self.align_rot(g_up, t_up)
                        if rot_action == [0,0,0]:
                            gconn_pos = self.sim.data.get_site_xpos(gconn)
                            tconn_pos = self.sim.data.get_site_xpos(tconn)
                            d = (tconn_pos[0:2] - gconn_pos[0:2])
                            if abs(d[0]) > epsilon or abs(d[1]) > epsilon:
                                phase_num+=1
                            else:
                                phase_num+=2
                        else:
                            action[3:6] = rot_action

                    elif self._phase == 'conn_align_xy':
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
                            phase_num-=1
                    
                    elif self._phase == 'conn_align_z':
                        action[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        d = tconn_pos - gconn_pos
                        d[2] += z_conn_dist
                        if abs(d[2]) > z_finedist: 
                            action[0:3] = d
                        else:
                            phase_num+=1

                    elif self._phase == 'conn_align_xy_fine':
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

                    elif self._phase == 'conn_align_z_fine':
                        action[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        d = tconn_pos - gconn_pos
                        d[2] += z_conn_dist
                        if abs(d[2]) > epsilon: 
                            action[0:3] = d / fine_fact
                        else:
                            action[7] = 1 
                            phase_num+=1

                    elif self._phase == 'z_return':
                        action[6] = -1
                        d = (z_safepos - grippos[2])
                        if abs(d) > epsilon:
                            action[2] = d
                        else:
                            phase_num+=1
                    self._phase = self._phases[phase_num]
                    action[0:3] = action_magnitude_factor * action[0:3]
                    action = self._cap_action(action)
                    ob, reward, done, info = self.step(action)
                    if done:
                        print('assembled!')
                    self.render()
                    if self._config.record_vid:
                        self.vid_rec.capture_frame(self.render("rgb_array")[0])
                if self._part_done:
                    self._used_connsites.add(gconn)
                    self._used_connsites.add(tconn)

            #self._demo.save(self.file_prefix)
            if self._config.record_vid:
                self.vid_rec.close(name=self._config.furniture_name+f"{demo_num}.mp4")




def main():
    from config import create_parser

    parser = create_parser(env="FurnitureSawyerGenEnv")
    config, unparsed = parser.parse_known_args()

    # generate placing demonstrations
    env = FurnitureSawyerGenEnv(config)
    
    #env.run_manual(config)
    env.generate_demos(10)
    #env.test()

if __name__ == "__main__":
    main()
