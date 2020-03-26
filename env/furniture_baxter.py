""" Define baxter environment class FurnitureBaxterEnv. """

from collections import OrderedDict

import numpy as np

from env.furniture import FurnitureEnv
import env.transform_utils as T
from util.logger import logger


class FurnitureBaxterEnv(FurnitureEnv):
    """
    Baxter robot environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.agent_type = 'Baxter'

        super().__init__(config)

        self._env_config.update({
            "success_reward": 100,
        })

    @property
    def observation_space(self):
        """
        Returns the observation space.
        """
        ob_space = super().observation_space

        if self._robot_ob:
            if self._control_type == 'impedance':
                ob_space['robot_ob'] = [64]
            elif self._control_type == 'ik':
                ob_space['robot_ob'] = [(3 + 4 + 3 + 3 + 1) * 2]

        return ob_space

    @property
    def dof(self):
        """
        Returns the DoF of the robot.
        """
        dof = 0  # 'No' Agent
        if self._control_type == 'impedance':
            dof = (7 + 2) * 2
        elif self._control_type == 'ik':
            dof = (3 + 3 + 1) * 2 + 1 # (move, rotate, select) * 2 + connect
        return dof

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        prev_reward, _, old_info = self._compute_reward()

        ob, _, done, _ = super()._step(a)

        reward, done, info = self._compute_reward()

        ctrl_reward = self._ctrl_reward(a)
        info['reward_ctrl'] = ctrl_reward

        connect_reward = reward - prev_reward
        info['reward_connect'] = connect_reward

        if self._success:
            logger.info('Success!')

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
            if self._control_type == 'impedance':
                robot_states["joint_pos"] = np.array(
                    [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
                )
                robot_states["joint_vel"] = np.array(
                    [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
                )
                robot_states["right_gripper_qpos"] = np.array(
                    [self.sim.data.qpos[x] for x in self._ref_gripper_right_joint_pos_indexes]
                )
                robot_states["right_gripper_qvel"] = np.array(
                    [self.sim.data.qvel[x] for x in self._ref_gripper_right_joint_vel_indexes]
                )
                robot_states["left_gripper_qpos"] = np.array(
                    [self.sim.data.qpos[x] for x in self._ref_gripper_left_joint_pos_indexes]
                )
                robot_states["left_gripper_qvel"] = np.array(
                    [self.sim.data.qvel[x] for x in self._ref_gripper_left_joint_vel_indexes]
                )

            right_gripper_qpos = [self.sim.data.qpos[x] for x in self._ref_gripper_right_joint_pos_indexes]
            left_gripper_qpos = [self.sim.data.qpos[x] for x in self._ref_gripper_left_joint_pos_indexes]
            robot_states["right_gripper_dis"] = np.array(
                [abs(right_gripper_qpos[0] - right_gripper_qpos[1])]
            )
            robot_states["left_gripper_dis"] = np.array(
                [abs(left_gripper_qpos[0] - left_gripper_qpos[1])]
            )
            robot_states["right_eef_pos"] = np.array(self.sim.data.site_xpos[self.right_eef_site_id])
            robot_states["right_eef_velp"] = np.array(self.sim.data.site_xvelp[self.right_eef_site_id]) # 3-dim
            robot_states["right_eef_velr"] = self.sim.data.site_xvelr[self.right_eef_site_id] # 3-dim
            robot_states["left_eef_pos"] = np.array(self.sim.data.site_xpos[self.left_eef_site_id])
            robot_states["left_eef_velp"] = np.array(self.sim.data.site_xvelp[self.left_eef_site_id]) # 3-dim
            robot_states["left_eef_velr"] = self.sim.data.site_xvelr[self.left_eef_site_id] # 3-dim

            robot_states["right_eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("right_hand"), to="xyzw"
            )
            robot_states["left_eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("left_hand"), to="xyzw"
            )

            state['robot_ob'] = np.concatenate(
                [x.ravel() for _, x in robot_states.items()]
            )

        return state

    def _get_reference(self):
        """
        Sets up references to robot joints and objects.
        """
        super()._get_reference()

        self.l_finger_geom_ids = [
            [self.sim.model.geom_name2id(x) for x in self.gripper_left.left_finger_geoms],
            [self.sim.model.geom_name2id(x) for x in self.gripper_right.left_finger_geoms]
        ]
        self.r_finger_geom_ids = [
            [self.sim.model.geom_name2id(x) for x in self.gripper_left.right_finger_geoms],
            [self.sim.model.geom_name2id(x) for x in self.gripper_right.right_finger_geoms]
        ]

        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        self._ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]

        # indices for grippers in qpos, qvel
        self.gripper_left_joints = list(self.gripper_left.joints)
        self._ref_gripper_left_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_left_joints
        ]
        self._ref_gripper_left_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_left_joints
        ]
        self.left_eef_site_id = self.sim.model.site_name2id("l_g_grip_site")

        self.gripper_right_joints = list(self.gripper_right.joints)
        self._ref_gripper_right_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_right_joints
        ]
        self._ref_gripper_right_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_right_joints
        ]
        self.right_eef_site_id = self.sim.model.site_name2id("grip_site")

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_pos_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("pos")
        ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("vel")
        ]

        self._ref_joint_gripper_left_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("gripper_l")
        ]

        self._ref_joint_gripper_right_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("gripper_r")
        ]

    def _compute_reward(self):
        """
        Computes reward of the current state.
        """
        return super()._compute_reward()


def main():
    import argparse
    from config import create_parser
    from util import str2bool

    parser = create_parser(env='FurnitureBaxterEnv')
    parser.set_defaults(render=True, record_demo=True)

    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterEnv(config)
    env.run_manual(config)


if __name__ == "__main__":
    main()
