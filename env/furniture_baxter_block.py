""" Define baxter block picking up environment class FurnitureBaxterBlockEnv. """
import numpy as np

import env.transform_utils as T
from env.furniture_baxter import FurnitureBaxterEnv
from env.models import furniture_name2id
from util.logger import logger


class FurnitureBaxterBlockEnv(FurnitureBaxterEnv):
    """
    Baxter robot environment with a block picking up task.
    Fix initialization of objects and only use right arm to make the task simple.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        # set the furniture to be always the simple blocks
        config.furniture_id = furniture_name2id["block"]

        super().__init__(config)

        self._env_config.update(
            {
                "max_episode_steps": 50,
                "success_reward": 10,
                "gripper_rot_reward": 5,
                "arm_move_reward": 500,
                "obj_hand_dist_reward": 200,
                "gripper_open_reward": 2,
                "arm_stable_reward": 20,
                "finger_pos_reward": 30,
                "gripper_height_reward": 500,
                "obj_pos_reward": 100,
                "obj_rot_reward": 1,
                "pass_reward": 400,
                "furn_xyz_rand": 0.0,
                "train": [True, False],
            }
        )

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        # zero out left arm's action and only use right arm
        a[6:12] = 0
        ob, _, done, _ = super(FurnitureBaxterEnv, self)._step(a)

        reward, done, info = self._compute_reward(a)

        if self._success:
            logger.info("Success!")

        info["right_action"] = a[0:6]
        info["left_action"] = a[6:12]
        info["gripper"] = a[12:]

        return ob, reward, done, info

    def _reset(self, furniture_id=None, background=None):
        """
        Resets simulation and variables to compute reward.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        super()._reset(furniture_id, background)

        # set two bodies for picking or assemblying
        id1 = self.sim.model.eq_obj1id[0]
        id2 = self.sim.model.eq_obj2id[0]
        self._target_body = [
            self.sim.model.body_id2name(id1),
            self.sim.model.body_id2name(id2),
        ]

        # subtask phase
        self._phase = 0

        # intermediate and target positions
        self._above_target = [[-0.3, -0.2, 0.15], [0.1, -0.2, 0.15]]
        self._target_pos = [[-0.1, -0.2, 0.1], [0.1, -0.2, 0.05]]

        # target rotations
        self._target_up = [
            self._get_up_vector(self._target_body[0]),
            self._get_up_vector(self._target_body[1]),
        ]
        self._target_forward = [
            self._get_forward_vector(self._target_body[0]),
            self._get_forward_vector(self._target_body[1]),
        ]

        # initial rotation of grippers
        self._gripper_up = [
            self._get_up_vector("r_fingertip_g0"),
            self._get_up_vector("l_g_r_fingertip_g0"),
        ]
        self._gripper_forward = [
            self._get_forward_vector("r_fingertip_g0"),
            self._get_forward_vector("l_g_r_fingertip_g0"),
        ]

        # initial position of grippers
        hand_pos = [
            np.array(self.sim.data.site_xpos[self.right_eef_site_id]),
            np.array(self.sim.data.site_xpos[self.left_eef_site_id]),
        ]
        self._dist = [
            T.l2_dist(hand_pos[0], self._above_target[0]),
            T.l2_dist(hand_pos[1], self._above_target[1]),
        ]

        self._height = 0.05

    def _place_objects(self):
        """
        Returns the fixed initial positions and rotations of furniture parts.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        pos_init = {'1_block_l' : [-0.3, -0.2, 0.05],
                    '2_block_r' : [0.1, -0.2, 0.05]
                   }
        quat_init = {'1_block_l' : [1, 0, 0, 0],
                     '2_block_r' : [1, 0, 0, 0]
                    }
        return pos_init, quat_init

    def _compute_reward(self, a):
        """
        Computes reward for the block picking up task.
        """
        info = {}

        # control penalty
        ctrl_reward = self._ctrl_reward(a)
        info["reward_ctrl"] = ctrl_reward

        # reward for successful assembly
        success_reward = self._env_config["success_reward"] * self._num_connected
        info["reward_success"] = success_reward

        reward = ctrl_reward + success_reward

        # compute positions and rotations for reward function
        hand_pos = [
            np.array(self.sim.data.site_xpos[self.right_eef_site_id]),
            np.array(self.sim.data.site_xpos[self.left_eef_site_id]),
        ]
        info["right_hand"] = hand_pos[0]
        info["left_hand"] = hand_pos[1]

        finger_pos = [
            [self._get_pos("r_fingertip_g0"), self._get_pos("l_fingertip_g0")],
            [self._get_pos("l_g_r_fingertip_g0"), self._get_pos("l_g_l_fingertip_g0")],
        ]
        info["right_r_finger"] = finger_pos[0][0]
        info["right_l_finger"] = finger_pos[0][1]

        gripper_up = [
            self._get_up_vector("r_fingertip_g0"),
            self._get_up_vector("l_g_r_fingertip_g0"),
        ]
        gripper_forward = [
            self._get_forward_vector("r_fingertip_g0"),
            self._get_forward_vector("l_g_r_fingertip_g0"),
        ]

        # check whether grippers touch the blocks or not
        touch_left_finger = [False, False]
        touch_right_finger = [False, False]
        for j in range(self.sim.data.ncon):
            c = self.sim.data.contact[j]
            body1 = self.sim.model.geom_bodyid[c.geom1]
            body2 = self.sim.model.geom_bodyid[c.geom2]
            body1_name = self.sim.model.body_id2name(body1)
            body2_name = self.sim.model.body_id2name(body2)

            for i, arm in enumerate(self._arms):
                if (
                    c.geom1 in self.l_finger_geom_ids[arm]
                    and body2_name == self._target_body[i]
                ):
                    touch_left_finger[i] = True
                if (
                    body1_name == self._target_body[i]
                    and c.geom2 in self.l_finger_geom_ids[arm]
                ):
                    touch_left_finger[i] = True

                if (
                    c.geom1 in self.r_finger_geom_ids[arm]
                    and body2_name == self._target_body[i]
                ):
                    touch_right_finger[i] = True
                if (
                    body1_name == self._target_body[i]
                    and c.geom2 in self.r_finger_geom_ids[arm]
                ):
                    touch_right_finger[i] = True

        # compute reward for each arm
        for arm_i in range(2):
            if not self._env_config["train"][arm_i]:
                continue

            gripper_rot_reward = 0
            arm_move_reward = 0
            arm_stable_reward = 0
            finger_pos_reward = 0
            gripper_height_reward = 0
            gripper_open_reward = 0
            obj_stable_reward = 0
            obj_hand_dist_reward = 0
            obj_pos_reward = 0
            obj_rot_reward = 0
            pass_reward = 0

            # block position
            pos = self._get_pos(self._target_body[arm_i])
            info["target_pos_%d" % arm_i] = pos

            # encourage gripper not to rotate
            gripper_rot_reward = self._env_config["gripper_rot_reward"] * (
                T.cos_dist(self._gripper_up[arm_i], gripper_up[arm_i])
                - 0.8
                + T.cos_dist(self._gripper_forward[arm_i], gripper_forward[arm_i])
                - 0.8
            )
            gripper_rot_reward = max(gripper_rot_reward, -2)

            if self._phase == 0:
                # put arm above the block
                dist = T.l2_dist(hand_pos[arm_i], self._above_target[arm_i])
                arm_move_reward = self._env_config["arm_move_reward"] * (
                    self._dist[arm_i] - dist
                )
                self._dist[arm_i] = dist

                gripper_open_reward -= self._env_config["gripper_open_reward"] * a[-3]

                if dist < 0.025:
                    self._phase = 1
                    pass_reward = self._env_config["pass_reward"]
                    self._dist[arm_i] = hand_pos[arm_i][2]

            elif self._phase == 1:
                # lower down arm closer to the block
                dist = T.l2_dist(hand_pos[arm_i][:2], self._above_target[arm_i][:2])
                arm_stable_reward = -self._env_config["arm_stable_reward"] * min(
                    dist, 0.2
                )

                r_finger_dis = pos[1] - 0.021 - finger_pos[arm_i][0][1]
                l_finger_dis = finger_pos[arm_i][1][1] - (pos[1] + 0.021)

                finger_pos_reward += self._env_config["finger_pos_reward"] * (
                    np.clip(r_finger_dis, -0.05, 0.02)
                )
                finger_pos_reward += self._env_config["finger_pos_reward"] * (
                    np.clip(l_finger_dis, -0.05, 0.02)
                )

                if r_finger_dis > 0 and l_finger_dis > 0:
                    finger_pos_reward = 2

                gripper_height_reward += (
                    self._dist[arm_i] - min(0.2, max(0.02, hand_pos[arm_i][2]))
                ) * self._env_config["gripper_height_reward"]
                self._dist[arm_i] = min(0.2, max(0.02, hand_pos[arm_i][2]))

                gripper_open_reward -= self._env_config["gripper_open_reward"] * a[-3]

                if (
                    r_finger_dis > 0
                    and l_finger_dis > 0
                    and dist < 0.05
                    and hand_pos[arm_i][2] < 0.11
                    and pos[2] > 0.04
                ):
                    self._phase = 2
                    pass_reward = self._env_config["pass_reward"]
                    self._dist[arm_i] = hand_pos[arm_i][2]

            elif self._phase == 2:
                # lower down arm and put the block between fingers
                dist = T.l2_dist(hand_pos[arm_i][:2], self._above_target[arm_i][:2])
                arm_stable_reward = -self._env_config["arm_stable_reward"] * min(
                    dist, 0.1
                )

                r_finger_dis = pos[1] - 0.025 - finger_pos[arm_i][0][1]
                l_finger_dis = finger_pos[arm_i][1][1] - (pos[1] + 0.025)

                finger_pos_reward += self._env_config["finger_pos_reward"] * (
                    np.clip(r_finger_dis, -0.05, 0.02)
                )
                finger_pos_reward += self._env_config["finger_pos_reward"] * (
                    np.clip(l_finger_dis, -0.05, 0.02)
                )

                if r_finger_dis > 0 and l_finger_dis > 0:
                    finger_pos_reward = 2

                gripper_height_reward += (
                    self._dist[arm_i] - min(0.2, max(0.02, hand_pos[arm_i][2]))
                ) * self._env_config["gripper_height_reward"]
                self._dist[arm_i] = min(0.2, max(0.02, hand_pos[arm_i][2]))

                gripper_open_reward -= self._env_config["gripper_open_reward"] * a[-3]

                if (
                    r_finger_dis > 0
                    and l_finger_dis > 0
                    and dist < 0.05
                    and hand_pos[arm_i][2] < 0.09
                    and pos[2] > 0.04
                ):
                    self._phase = 3
                    pass_reward = self._env_config["pass_reward"]
                    self._dist[arm_i] = hand_pos[arm_i][2]

            elif self._phase == 3:
                # lower down arm even more while place the block between fingers
                dist = T.l2_dist(hand_pos[arm_i][:2], self._above_target[arm_i][:2])
                arm_stable_reward = -self._env_config["arm_stable_reward"] * min(
                    dist, 0.1
                )

                r_finger_dis = pos[1] - 0.025 - finger_pos[arm_i][0][1]
                l_finger_dis = finger_pos[arm_i][1][1] - (pos[1] + 0.025)

                finger_pos_reward += self._env_config["finger_pos_reward"] * (
                    np.clip(r_finger_dis, -0.05, 0.02)
                )
                finger_pos_reward += self._env_config["finger_pos_reward"] * (
                    np.clip(l_finger_dis, -0.05, 0.02)
                )

                if r_finger_dis > 0 and l_finger_dis > 0:
                    finger_pos_reward = 2

                gripper_height_reward += (
                    self._dist[arm_i] - min(0.2, max(0.02, hand_pos[arm_i][2]))
                ) * self._env_config["gripper_height_reward"]
                self._dist[arm_i] = min(0.2, max(0.02, hand_pos[arm_i][2]))

                gripper_open_reward -= self._env_config["gripper_open_reward"] * a[-3]

                if (
                    r_finger_dis > 0
                    and l_finger_dis > 0
                    and dist < 0.05
                    and hand_pos[arm_i][2] < 0.05
                    and pos[2] > 0.04
                ):
                    self._phase = 4
                    pass_reward = self._env_config["pass_reward"]

            elif self._phase == 4:
                # hold the block
                gripper_open_reward += (
                    self._env_config["gripper_open_reward"] * a[-3] * 5
                )

                dist = T.l2_dist(hand_pos[arm_i], pos)
                obj_hand_dist_reward -= self._env_config["obj_hand_dist_reward"] * max(
                    0, dist - 0.02
                )

                if a[-3] > 0.5:
                    self._phase = 5
                    pass_reward = self._env_config["pass_reward"]
                    self._dist[arm_i] = T.l2_dist(self._above_target[arm_i], pos)

            elif self._phase == 5:
                # pick up the block
                dist = T.l2_dist(self._above_target[arm_i], pos)
                obj_pos_reward = (
                    self._env_config["obj_pos_reward"] * (self._dist[arm_i] - dist) * 10
                )
                self._dist[arm_i] = dist

                obj_pos_reward += (
                    self._env_config["obj_pos_reward"]
                    * max(min(pos[2], 0.15) - self._height, 0)
                    * 40
                )
                self._height = max(self._height, pos[2])

                gripper_open_reward += (
                    self._env_config["gripper_open_reward"] * a[-3] * 5
                )

                dist = T.l2_dist(hand_pos[arm_i], pos)
                obj_hand_dist_reward -= self._env_config["obj_hand_dist_reward"] * max(
                    0, dist - 0.02
                )

                dist = T.l2_dist(self._above_target[arm_i], pos)
                if dist < 0.05:
                    self._phase = 6
                    pass_reward = self._env_config["pass_reward"]
                    self._dist[arm_i] = T.l2_dist(pos, self._target_pos[arm_i])

            elif self._phase == 6:
                # move the block to the target position
                gripper_open_reward += self._env_config["gripper_open_reward"] * a[-1]
                gripper_open_reward += self._env_config["gripper_open_reward"] * a[-3]

                dist = T.l2_dist(pos, self._target_pos[arm_i])
                obj_pos_reward = (
                    self._env_config["obj_pos_reward"] * (self._dist[arm_i] - dist) * 10
                )
                self._dist[arm_i] = dist

                dist = T.l2_dist(hand_pos[arm_i], pos)
                obj_hand_dist_reward -= self._env_config["obj_hand_dist_reward"] * max(
                    0, dist - 0.02
                )

                obj_rot_reward = self._env_config["obj_rot_reward"] * (
                    T.cos_dist(
                        self._get_up_vector(self._target_body[arm_i]),
                        self._target_up[arm_i],
                    )
                    + T.cos_dist(
                        self._get_forward_vector(self._target_body[arm_i]),
                        self._target_forward[arm_i],
                    )
                )

            if self._phase == 10:
                # failed to manipulate gripper
                reward += -1
            else:
                if self._phase < 3 and a[-3] > -0.5:
                    # if gripper opens before picking, it fails
                    self._phase = 10
                    gripper_open_reward = -200
                elif self._phase == 6 and a[-3] < 0.3:
                    # if gripper release the block, it fails
                    self._phase = 10
                    gripper_open_reward = -200

                if pos[2] < 0.04:
                    obj_stable_reward = -1

                reward += (
                    gripper_rot_reward
                    + arm_move_reward
                    + gripper_open_reward
                    + arm_stable_reward
                    + finger_pos_reward
                    + gripper_height_reward
                    + obj_hand_dist_reward
                    + obj_pos_reward
                    + obj_rot_reward
                    + pass_reward
                )

            info["phase"] = self._phase
            info["reward_gripper_rot_%d" % arm_i] = gripper_rot_reward
            info["reward_arm_move_%d" % arm_i] = arm_move_reward
            info["reward_gripper_open_%d" % arm_i] = gripper_open_reward
            info["reward_arm_stable_%d" % arm_i] = arm_stable_reward
            info["reward_finger_pos_%d" % arm_i] = finger_pos_reward
            info["reward_gripper_height_%d" % arm_i] = gripper_height_reward
            info["reward_obj_stable_%d" % arm_i] = obj_stable_reward
            info["reward_obj_hand_dist_%d" % arm_i] = obj_hand_dist_reward
            info["reward_obj_pos_%d" % arm_i] = obj_pos_reward
            info["reward_obj_rot_%d" % arm_i] = obj_rot_reward
            info["reward_pass_%d" % arm_i] = pass_reward

        done = False

        return reward, done, info


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureBaxterEnv")
    parser.set_defaults(render=True, record_demo=True)

    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterEnv(config)
    env.run_manual(config)


if __name__ == "__main__":
    main()
