""" Define Baxter environment class FurnitureBaxterToyTableEnv. """
from typing import Tuple

import numpy as np

from env.furniture_baxter import FurnitureBaxterEnv
from env.models import furniture_name2id
from util.logger import logger


class FurnitureBaxterToyTableEnv(FurnitureBaxterEnv):
    """
    Baxter environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.furniture_id = furniture_name2id["toy_table"]

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
            a[-3] = -1 if a[-3] < 0 else 1

        ob, _, done, _ = super(FurnitureBaxterEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)

        if self._debug:
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

        # set two bodies for picking or assemblying
        id1 = self.sim.model.eq_obj1id[0]
        id2 = self.sim.model.eq_obj2id[0]
        self._target_body1 = self.sim.model.body_id2name(id1)
        self._target_body2 = self.sim.model.body_id2name(id2)

        # reward variables
        self._phase = "grip_leg"
        self._leg_gripped = False
        self._orig_left_hand_pos = self._get_pos("l_g_grip_site")

    def _place_objects(self):
        """
        Returns fixed initial position and rotations of the toy table.
        The first case has the table top on the left and legs on the right.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        pos_init = [
            [-0.34684698 + 0.05, -0.12887974, 0.03418991],
            [0.03472849 - 0.0285, 0.11868485 - 0.05, 0.02096991],
        ]
        noise = self._init_random(3 * len(pos_init), "furniture")
        for i in range(len(pos_init)):
            for j in range(3):
                pos_init[i][j] += noise[3 * i + j]
        quat_init = [
            [0, 1, 0, 0],
            [0.707, 0.707, 0, 0],
        ]

        return pos_init, quat_init

    def _compute_reward(self, action) -> Tuple[float, bool, dict]:
        """
        phase 1: grip leg
        phase 2: disconnect leg
        phase 3: move leg up with left gripper
        """
        rew = leg_grip_rew = 0
        info = {}
        done = False
        leg_name = "2_part2"

        # phase 1: grip leg and disconnect leg
        left_hand_grip_leg = self._gripper_contact(leg_name, ["left"])["left"]

        if left_hand_grip_leg and not self._leg_gripped and self._phase == "grip_leg":
            leg_grip_rew = 1
            self._leg_gripped = True
            self._phase = "disconnect_leg"

        # phase 2: move leg upwards
        # leg_pos = self._get_pos(leg_name)

        # phase 3: move leg to table

        rew = leg_grip_rew
        return rew, done, info

    def _try_connect(self, part1=None, part2=None):
        """
        Disconnects all parts attached to part1
        part1, part2 are names of the body
        """
        assert part1 is not None and part2 is None
        for i, (id1, id2) in enumerate(
            zip(self.sim.model.eq_obj1id, self.sim.model.eq_obj2id)
        ):
            p1 = self.sim.model.body_id2name(id1)
            p2 = self.sim.model.body_id2name(id2)
            if part1 in [p1, p2]:
                self.sim.model.eq_active[i] = 0

    def generate_demos(self, num_demos):
        """
        Close left hand gripper and move gripper and table leg up
        Set point slightly above hand as target, repeat

        @a is a 6 + 6 + 2 + 1 = 15 dim array; 0:6 are change in x,y,z and rx ry rz for
        right hand, 6:12 are change for left hand, 12 is select for right hand, 13 is
        select for left hand, 14 is connect action
        """
        cfg = self._config

        done = False
        action = np.zeros((15,))
        ob = self.reset(cfg.furniture_id, cfg.background)
        disconnected = False
        while not done:
            # keep left and right hand closed, disconnect
            action[-2] = action[-1] = 1
            if not disconnected:
                action[-1] = 1
                disconnected = True
            else:
                # move left hand up
                left_hand_pos = self._get_pos("l_g_grip_site")
                above_left_hand_pos = left_hand_pos + [0, 0, 0.01]
                d = above_left_hand_pos - left_hand_pos
                action[6:9] = d
            ob, reward, done, info = self.step(action)
            self.render()


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureBaxterToyTableEnv")
    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterToyTableEnv(config)
    # env.run_manual(config)
    env.generate_demos(1)

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
