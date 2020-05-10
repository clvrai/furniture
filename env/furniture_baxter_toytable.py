""" Define Baxter environment class FurnitureBaxterToyTableEnv. """
import numpy as np

import env.transform_utils as T
from env.furniture_baxter import FurnitureBaxterEnv
from env.models import furniture_name2id
from util import clamp
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
        config.furniture_id = furniture_name2id["swivel_chair_0700"]

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
        ob, _, done, _ = super(FurnitureBaxterEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)

        # for i, body in enumerate(self._object_names):
        #     pose = self._get_qpos(body)
        #     logger.debug(f"{body} {pose[:3]} {pose[3:]}")

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

    # def _place_objects(self):
    #     """
    #     Returns fixed initial position and rotations of the toy table.
    #     The first case has the table top on the left and legs on the right.

    #     Returns:
    #         xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
    #         xquat((float * 4) * n_obj): quaternion of the objects
    #     """
    #     # pos_init = [[ 0.21250838, -0.1163671 ,  0.02096991], [-0.30491682, -0.09045364,  0.03429339],[ 0.38134436, -0.11249256,  0.02096991],[ 0.12432612, -0.13662719,  0.02096991],[ 0.29537311, -0.12992911,  0.02096991]]
    #     # quat_init = [[0.706332  , 0.70633192, 0.03309327, 0.03309326], [ 0.00000009, -0.99874362, -0.05011164,  0.00000002], [ 0.70658149,  0.70706735, -0.00748174,  0.0272467 ], [0.70610751, 0.7061078 , 0.03757641, 0.03757635], [0.70668613, 0.70668642, 0.02438253, 0.02438249]]
    #     pos_init = [
    #         [-0.34684698 + 0.05, -0.12887974, 0.03418991],
    #         [0.03472849 - 0.0285, 0.11868485 - 0.05, 0.02096991],
    #     ]
    #     noise = self._init_random(3 * len(pos_init), "furniture")
    #     for i in range(len(pos_init)):
    #         for j in range(3):
    #             pos_init[i][j] += noise[3 * i + j]
    #     quat_init = [
    #         [0.00000009, -0.99874362, -0.05011164, 0.00000002],
    #         [-0.70610751, 0.7061078, -0.03757641, 0.03757635],
    #     ]

    #     return pos_init, quat_init

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
                # setup eq_data
                # self.sim.model.eq_data[i] = T.rel_pose(
                #     self._get_qpos(p1), self._get_qpos(p2)
                # )
                self.sim.model.eq_active[i] = 0

    def _compute_reward(self, action):
        return 0, False, {}

    def _load_model(self):
        """
        Loads the Task, which is composed of arena, robot, objects, equality
        """
        # task includes arena, robot, and objects of interest
        from env.models.tasks import FloorTask

        self.mujoco_model = FloorTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.mujoco_equality,
            self._rng,
        )

def main():
    from config import create_parser

    parser = create_parser(env="FurnitureBaxterToyTableEnv")
    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterToyTableEnv(config)
    env.run_manual(config)

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
