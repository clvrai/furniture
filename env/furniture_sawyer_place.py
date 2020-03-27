import numpy as np

from env.furniture_sawyer import FurnitureSawyerEnv
from env.models import furniture_name2id
from util.logger import logger


class FurnitureSawyerPlaceEnv(FurnitureSawyerEnv):
    """
    Sawyer environment for placing a block onto table.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.furniture_id = furniture_name2id["placeblock"]

        super().__init__(config)
        # default values for rew function
        self._env_config.update(
            {
                "pos_dist": 0.04,
                "rot_dist_up": 0.95,
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
                "topsite_z_offset": config.topsite_z_offset,
                "hold_duration": config.hold_duration,
                "grip_penalty": config.grip_penalty,
                "xy_dist_rew": config.xy_dist_rew,
                "z_dist_rew": config.z_dist_rew,
            }
        )
        self._gravity_compensation = 1
        # requires multiple connection actions to make connection between two
        # parts.
        self._num_connect_steps = 0
        self._discretize_grip = config.discretize_grip

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        # discretize gripper action
        if self._discretize_grip:
            a = a.copy()
            a[-2] = -1 if a[-2] < 0 else 1

        ob, _, done, _ = super(FurnitureSawyerEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)

        for i, body in enumerate(self._object_names):
            pose = self._get_qpos(body)
            logger.debug(f"{body} {pose[:3]} {pose[3:]}")

        info["ac"] = a

        return ob, reward, done, info

    def _reset(self, furniture_id=None, background=None):
        """
        Resets simulation.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        super()._reset(furniture_id, background)

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
        pos_init = [0.04521311, 0.04596679, 0.11724173]
        quat_init = [0.51919501, 0.52560512, 0.47367611, 0.47938163]

        return pos_init, quat_init

    def _ctrl_reward(self, action):
        if self._config.control_type == "ik":
            a = np.linalg.norm(action[:6])
        elif self._config.control_type == "impedance":
            a = np.linalg.norm(action[:7])

        # grasp_offset, grasp_leg, grip_leg
        ctrl_penalty = -self._env_config["ctrl_penalty"] * a
        if self._phase in [
            "move_leg_up",
            "move_leg",
            "connect",
        ]:  # move slower when moving leg
            ctrl_penalty *= 1

        return ctrl_penalty

    def _compute_reward(self, action):
        rew = 0
        done = False
        info = {}
        return rew, done, info

    def place_block(self, config):
        """
        1. Move to xy location and release block
        2. Move gripper in z direction above block to avoid collision
        2. Move to original starting point
        """
        ob = self.reset(config.furniture_id, config.background)
        if config.render:
            self.render()

        done = False
        original_hand_pos = self._get_pos("grip_site")
        ground_pos = np.random.uniform(
            low=[-0.01, -0.01, 0.005], high=[0.01, 0.01, 0.015], size=3
        )
        above_block_pos = None
        phase = 1
        while not done:
            action = np.zeros((8,))
            hand_pos = self._get_pos("grip_site")
            if phase == 1:
                action[6] = 1  # always grip
                d = ground_pos - hand_pos
                if np.linalg.norm(d) > 0.005:
                    action[:3] = d
                else:
                    phase = 2
                    above_block_pos = self._get_pos("1_block_l") + [0, 0, 0.03]
            elif phase == 2:
                action[6] = -1  # release block
                d = above_block_pos - hand_pos
                if np.linalg.norm(d) > 0.005:
                    action[:3] = d
                else:
                    phase = 3
            elif phase == 3:
                action[6] = -1  # release block
                d = original_hand_pos - hand_pos
                if np.linalg.norm(d) > 0.005:
                    action[:3] = d
                else:
                    phase = 4
            ob, reward, done, info = self.step(action)
            self.render()


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureSawyerPlaceEnv")
    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Sawyer environment
    env = FurnitureSawyerPlaceEnv(config)
    env.place_block(config)

    # import pickle
    # with open("demos/Sawyer_toy_table_0022.pkl", "rb") as f:
    #     demo = pickle.load(f)
    # env.reset()
    # print(len(demo['actions']))

    # from util.video_recorder import VideoRecorder
    # vr = VideoRecorder()
    # vr.add(env.render('rgb_array')[0])
    # for ac in demo['actions']:
    #     env.step(ac)
    #     vr.add(env.render('rgb_array')[0])
    # vr.save_video('test.mp4')


if __name__ == "__main__":
    main()
