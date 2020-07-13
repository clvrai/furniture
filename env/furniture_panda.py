""" Define Panda environment class FurniturePandaEnv. """

from collections import OrderedDict

import numpy as np
import gym.spaces

from env.furniture import FurnitureEnv
import env.transform_utils as T
from util.logger import logger


class FurniturePandaEnv(FurnitureEnv):
    """
    Panda environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.agent_type = "Panda"

        super().__init__(config)

        self._env_config.update({"success_reward": 100})

    @property
    def observation_space(self):
        """
        Returns the observation space.
        """
        ob_space = super().observation_space

        if self._robot_ob:
            if self._control_type == "impedance":
                ob_space.spaces["robot_ob"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(32,),
                )
            elif self._control_type == "ik":
                ob_space.spaces["robot_ob"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3 + 4 + 3 + 3 + 1,),  # pos, quat, vel, rot_vel, gripper
                )

        return ob_space

    @property
    def dof(self):
        """
        Returns the DoF of the robot.
        """
        dof = 0  # 'No' Agent
        if self._control_type == "impedance":
            dof = 7 + 2
        elif self._control_type == "ik":
            dof = 3 + 3 + 1 + 1  # move, rotate, select, connect
        elif self._control_type == "ik_quaternion":
            dof = 3 + 4 + 1 + 1  # move, rotate, select, connect
        return dof

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        prev_reward, _, old_info = self._compute_reward()

        ob, _, done, _ = super()._step(a)

        reward, done, info = self._compute_reward()

        ctrl_reward = self._ctrl_reward(a)
        info["reward_ctrl"] = ctrl_reward

        connect_reward = reward - prev_reward
        info["reward_connect"] = connect_reward

        if self._success:
            logger.info("Success!")

        reward = ctrl_reward + connect_reward

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

    def _get_obs(self):
        """
        Returns the current observation.
        """
        state = super()._get_obs()

        # proprioceptive features
        if self._robot_ob:
            robot_states = OrderedDict()
            if self._control_type == "impedance":
                robot_states["joint_pos"] = np.array(
                    [
                        self.sim.data.qpos[x]
                        for x in self._ref_joint_pos_indexes["right"]
                    ]
                )
                robot_states["joint_vel"] = np.array(
                    [
                        self.sim.data.qvel[x]
                        for x in self._ref_joint_vel_indexes["right"]
                    ]
                )
                robot_states["gripper_qpos"] = np.array(
                    [
                        self.sim.data.qpos[x]
                        for x in self._ref_gripper_joint_pos_indexes["right"]
                    ]
                )
                robot_states["gripper_qvel"] = np.array(
                    [
                        self.sim.data.qvel[x]
                        for x in self._ref_gripper_joint_vel_indexes["right"]
                    ]
                )

            gripper_qpos = [
                self.sim.data.qpos[x]
                for x in self._ref_gripper_joint_pos_indexes["right"]
            ]
            robot_states["gripper_dis"] = np.array(
                [(gripper_qpos[0] + 0.0115) - (gripper_qpos[1] - 0.0115)]
            )  # range of grippers are [-0.0115, 0.0208] and [-0.0208, 0.0115]
            robot_states["eef_pos"] = np.array(
                self.sim.data.site_xpos[self.eef_site_id["right"]]
            )
            robot_states["eef_velp"] = np.array(
                self.sim.data.site_xvelp[self.eef_site_id["right"]]
            )  # 3-dim
            robot_states["eef_velr"] = self.sim.data.site_xvelr[
                self.eef_site_id["right"]
            ]  # 3-dim

            robot_states["eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("right_hand"), to="xyzw"
            )

            state["robot_ob"] = np.concatenate(
                [x.ravel() for _, x in robot_states.items()]
            )

        return state

    def _get_reference(self):
        """
        Sets up references to robot joints and objects.
        """
        super()._get_reference()

        self.l_finger_geom_ids = {
            "right": [
                self.sim.model.geom_name2id(x) for x in self.gripper["right"].left_finger_geoms
            ]
        }
        self.r_finger_geom_ids = {
            "right": [
                self.sim.model.geom_name2id(x) for x in self.gripper["right"].right_finger_geoms
            ]
        }

        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        self._ref_joint_pos_indexes_all = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_vel_indexes_all = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_pos_indexes = {
            "right": self._ref_joint_pos_indexes_all,
            "left": [],
        }
        self._ref_joint_vel_indexes = {
            "right": self._ref_joint_vel_indexes_all,
            "left": [],
        }

        # indices for grippers in qpos, qvel
        self.gripper_joints = list(self.gripper["right"].joints)
        self._ref_gripper_joint_pos_indexes = {
            "right": [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
            ]
        }
        self._ref_gripper_joint_vel_indexes = {
            "right": [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
            ]
        }

        # IDs of sites for gripper visualization
        self.eef_site_id = {"right": self.sim.model.site_name2id("grip_site")}
        self.eef_cylinder_id = {
            "right": self.sim.model.site_name2id("grip_site_cylinder")
        }

    def _compute_reward(self):
        """
        Computes reward of the current state.
        """
        return super()._compute_reward()


def main():
    import argparse
    import config.furniture as furniture_config
    from util import str2bool

    parser = argparse.ArgumentParser()
    furniture_config.add_argument(parser)

    # change default config for Panda
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", type=str2bool, default=False)

    parser.set_defaults(render=True)

    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Panda environment
    env = FurniturePandaEnv(config)
    env.run_manual(config)


if __name__ == "__main__":
    main()
