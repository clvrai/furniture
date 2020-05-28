""" Define Baxter environment class FurnitureBaxterToyTableEnv. """
from typing import Tuple

import numpy as np
from tqdm import tqdm

from env.furniture_baxter import FurnitureBaxterEnv
from env.models import background_names, furniture_name2id, furniture_xmls
from util.logger import logger
from util.video_recorder import VideoRecorder


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
        self.sim.model.opt.gravity[-1] = -1
        # discretize gripper action
        if self._discretize_grip:
            a = a.copy()
            a[-2] = -1 if a[-2] < 0 else 1
            a[-3] = -1 if a[-3] < 0 else 1

        ob, _, done, _ = super(FurnitureBaxterEnv, self)._step(a)
        reward, done, info = self._compute_reward(a)

        # if self._debug:
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
        if self._config.furniture_name == "Random":
            furniture_id = self._rng.randint(len(furniture_xmls))
        if self._furniture_id is None or (
            self._furniture_id != furniture_id and furniture_id is not None
        ):
            # construct mujoco xml for furniture_id
            if furniture_id is None:
                self._furniture_id = self._config.furniture_id
            else:
                self._furniture_id = furniture_id
            self._reset_internal()

        # reset simulation data and clear buffers
        self.sim.reset()

        # store robot's contype, conaffinity (search MuJoCo XML API for details)
        # disable robot collision
        robot_col = {}
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name not in self._object_names and self.mujoco_robot.is_robot_part(
                geom_name
            ):
                robot_col[geom_name] = (
                    self.sim.model.geom_contype[geom_id],
                    self.sim.model.geom_conaffinity[geom_id],
                )
                self.sim.model.geom_contype[geom_id] = 0
                self.sim.model.geom_conaffinity[geom_id] = 0

        # initialize collision for non-mesh geoms
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name in self._object_names and "collision" in geom_name:
                self.sim.model.geom_contype[geom_id] = 1
                self.sim.model.geom_conaffinity[geom_id] = 1

        # initialize group
        self._object_group = list(range(len(self._object_names)))

        # initialize member variables
        self._connect_step = 0
        self._connected_sites = set()
        self._connected_body1 = None
        self._connected_body1_pos = None
        self._connected_body1_quat = None
        self._num_connected = 0

        # initialize weld constraints
        eq_obj1id = self.sim.model.eq_obj1id
        eq_obj2id = self.sim.model.eq_obj2id
        p = self._preassembled  # list of weld equality ids to activate
        if len(p) > 0:
            for eq_id in p:
                self.sim.model.eq_active[eq_id] = 1
                object_body_id1 = eq_obj1id[eq_id]
                object_body_id2 = eq_obj2id[eq_id]
                object_name1 = self._object_body_id2name[object_body_id1]
                object_name2 = self._object_body_id2name[object_body_id2]
                self._merge_groups(object_name1, object_name2)
        elif eq_obj1id is not None:
            for i, (id1, id2) in enumerate(zip(eq_obj1id, eq_obj2id)):
                self.sim.model.eq_active[i] = 1 if self._config.assembled else 0

        self._do_simulation(None)
        # stablize furniture pieces
        for _ in range(100):
            for obj_name in self._object_names:
                self._stop_object(obj_name, gravity=0)
            self.sim.forward()
            self.sim.step()

        logger.debug("*** furniture initialization ***")
        # initialize the robot and block to initial demonstraiton state
        self._init_qpos = {
            "qpos": [
                0.15259831,
                -0.24533181,
                0.68495219,
                2.28670809,
                -0.39624288,
                -1.54411427,
                -0.75568118,
                -1.12699962,
                -0.16681518,
                0.05078415,
                0.89042369,
                -0.07658486,
                1.0002326,
                -1.9998653,
            ],
            "4_part4": [
                -0.11734456,
                -0.26209947,
                0.24811555,
                -0.4469388,
                0.47928162,
                0.52778792,
                -0.54034688,
            ],
            "2_part2": [
                -0.00578973,
                -0.05484689,
                0.04273277,
                0.44693876,
                -0.47928137,
                -0.52778823,
                0.54034683,
            ],
            "r_gripper": [-0.01900776, 0.01889891],
            "l_gripper": [-0.01304315, 0.01261245],
        }
        # set toy table pose
        pos_init = []
        quat_init = []
        for body in self._object_names:
            qpos = self._init_qpos[body]
            pos_init.append(qpos[:3])
            quat_init.append(qpos[3:])
        # set baxter pose
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self._init_qpos["qpos"]
        self.sim.data.qpos[self._ref_gripper_right_joint_pos_indexes] = self._init_qpos[
            "r_gripper"
        ]
        self.sim.data.qpos[self._ref_gripper_left_joint_pos_indexes] = self._init_qpos[
            "l_gripper"
        ]

        # enable robot collision
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name not in self._object_names and self.mujoco_robot.is_robot_part(
                geom_name
            ):
                contype, conaffinity = robot_col[geom_name]
                self.sim.model.geom_contype[geom_id] = contype
                self.sim.model.geom_conaffinity[geom_id] = conaffinity

        # set furniture positions
        for i, body in enumerate(self._object_names):
            logger.debug(f"{body} {pos_init[i]} {quat_init[i]}")
            if self._config.assembled:
                self._object_group[i] = 0
            else:
                self._set_qpos(body, pos_init[i], quat_init[i])

        self.sim.forward()

        # store qpos of furniture and robot
        if self._record_demo:
            self._store_qpos()

        if self._agent_type in ["Sawyer", "Panda", "Jaco", "Baxter"]:
            self._initial_right_hand_quat = self._right_hand_quat
            if self._agent_type == "Baxter":
                self._initial_left_hand_quat = self._left_hand_quat

            if self._control_type == "ik":
                # set up ik controller
                self._controller.sync_state()

        # set next subtask
        self._get_next_subtask()

        # set object positions in unity
        if self._unity:
            if background is None and self._background is None:
                background = self._config.background
            if self._config.background == "Random":
                background = self._rng.choice(background_names)
            if background and background != self._background:
                self._background = background
                self._unity.set_background(background)

        # set two bodies for picking or assemblying
        id1 = self.sim.model.eq_obj1id[0]
        id2 = self.sim.model.eq_obj2id[0]
        self._target_body1 = self.sim.model.body_id2name(id1)
        self._target_body2 = self.sim.model.body_id2name(id2)

    def _compute_reward(self, action) -> Tuple[float, bool, dict]:
        """
        phase 1: grip leg
        phase 2: disconnect leg
        phase 3: move leg up with left gripper
        """
        rew = 0
        info = {}
        done = False
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
        for i in tqdm(range(num_demos)):
            done = False
            ob = self.reset(cfg.furniture_id, cfg.background)
            if cfg.render:
                self.render()
            vr = None
            if cfg.record:
                vr = VideoRecorder()
                vr.capture_frame(self.render("rgb_array")[0])
            phase = 0
            move_steps = 0

            while not done:
                action = np.zeros((15,))
                # keep left and right hand closed, disconnect
                action[-3] = action[-2] = 1
                # if phase == 0:
                #     action[-1] = 1
                #     phase = 1
                if phase == 0:
                    # move left to the right
                    action[6] = 0.2
                    # move right hand to the left
                    action[0] = -0.05
                    move_steps += 1
                    # add some random noise to peg gripper
                    if move_steps > 16:
                        r = 0.8
                        action[6] += self._rng.uniform(-r, r)
                        action[7] += self._rng.uniform(-r, r)
                    if move_steps == 30:
                        phase = 2
                        move_steps = 0
                elif phase == 2:
                    action[-2] = -1
                    # action[-3] = -1
                    move_steps += 1
                    if move_steps == 2:
                        phase = 3

                ob, reward, done, info = self.step(action)
                if cfg.render:
                    self.render()
                if cfg.record:
                    vr.capture_frame(self.render("rgb_array")[0])
                if phase == 3:
                    done = True
                    if cfg.record_demo:
                        self.save_demo()
                    if cfg.record:
                        vr.close(f"baxtertoytabledis_{i}.mp4")


def main():
    from config import create_parser
    from multiprocessing import Process

    parser = create_parser(env="FurnitureBaxterToyTableEnv")
    config, unparsed = parser.parse_known_args()

    def generate_demos(rank, config):
        config.seed = config.seed + rank
        # create an environment and run manual control of Baxter environment
        env = FurnitureBaxterToyTableEnv(config)
        # env.run_manual(config)
        env.generate_demos(config.num_demos)

    num_workers = 10
    workers = []
    for rank in range(num_workers):
        p = Process(target=generate_demos, args=(rank, config), daemon=True)
        workers.append(p)
        p.start()

    for w in workers:
        p.join()
    # import pickle

    # with open("demos/Baxter_toy_table_0009.pkl", "rb") as f:
    #     demo = pickle.load(f)
    #     print(demo["qpos"][-1])
    #     import ipdb

    #     ipdb.set_trace()
    # env.reset()
    # print(len(demo["actions"]))

    # from util.video_recorder import VideoRecorder
    # vr = VideoRecorder()
    # vr.add(env.render('rgb_array')[0])
    # for ac in demo['actions']:
    #     env.step(ac)
    #     vr.add(env.render('rgb_array')[0])
    # vr.save_video('test.mp4')


if __name__ == "__main__":
    main()
