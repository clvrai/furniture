""" Define Sawyer environment class FurnitureSawyerEnv. """

from collections import OrderedDict

import gym.spaces
import numpy as np

import env.transform_utils as T
from env.furniture import FurnitureEnv
from util.logger import logger


class FurnitureSawyerEnv(FurnitureEnv):
    """
    Sawyer environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.agent_type = "Sawyer"

        super().__init__(config)

        self._env_config.update({"success_reward": 100})

    @property
    def observation_space(self):
        """
        Returns the observation space.
        """
        ob_space = super().observation_space

        if self._robot_ob:
            if self._control_type in ["impedance", "torque"]:
                ob_space.spaces["robot_ob"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        7 + 7 + 2 + 3 + 4 + 3 + 3,
                    ),  # qpos, qvel, gripper, eefp, eefq, velp, velr
                )
            elif self._control_type in ["ik", "ik_quaternion"]:
                ob_space.spaces["robot_ob"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3 + 4 + 3 + 3 + 2,),  # pos, quat, velp, velr, gripper
                )

        return ob_space

    @property
    def dof(self):
        """
        Returns the DoF of the robot.
        """
        dof = 0  # 'No' Agent
        if self._control_type in ["impedance", "torque"]:
            dof = 7 + 2  # 7 joints, select, connect
        elif self._control_type == "ik":
            dof = 3 + 3 + 1 + 1  # move, rotate, select, connect
        elif self._control_type == "ik_quaternion":
            dof = 3 + 4 + 1 + 1  # move, rotate, select, connect
        return dof

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        ob, _, done, _ = super()._step(a)

        reward, _, info = self._compute_reward(a)

        if self._success:
            logger.info("Success!")

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
        if self.sim.model.eq_obj1id is not None:
            id1 = self.sim.model.eq_obj1id[0]
            id2 = self.sim.model.eq_obj2id[0]
            self._target_body1 = self.sim.model.body_id2name(id1)
            self._target_body2 = self.sim.model.body_id2name(id2)

    def _get_obs(self, include_qpos=False):
        """
        Returns the current observation.
        """
        state = super()._get_obs()

        # proprioceptive features
        if self._robot_ob:
            robot_states = OrderedDict()
            if self._control_type in ["impedance", "torque"] or include_qpos:
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
            # gripper_qpos = [
            #     self.sim.data.qpos[x]
            #     for x in self._ref_gripper_joint_pos_indexes["right"]
            # ]
            # robot_states["gripper_dis"] = np.array(
            #     [(gripper_qpos[0] + 0.0115) - (gripper_qpos[1] - 0.0115)]
            # )  # range of grippers are [-0.0115, 0.0208] and [-0.0208, 0.0115]
            robot_states["gripper_qpos"] = np.array(
                [
                    self.sim.data.qpos[x]
                    for x in self._ref_gripper_joint_pos_indexes["right"]
                ]
            )
            robot_states["eef_pos"] = np.array(
                self.sim.data.site_xpos[self.eef_site_id["right"]]
            )
            robot_states["eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("right_hand"), to="xyzw"
            )
            robot_states["eef_velp"] = np.array(
                self.sim.data.site_xvelp[self.eef_site_id["right"]]
            )  # 3-dim
            robot_states["eef_velr"] = self.sim.data.site_xvelr[
                self.eef_site_id["right"]
            ]  # 3-dim

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
                self.sim.model.geom_name2id(x)
                for x in self.gripper["right"].left_finger_geoms
            ]
        }
        self.r_finger_geom_ids = {
            "right": [
                self.sim.model.geom_name2id(x)
                for x in self.gripper["right"].right_finger_geoms
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

    def _compute_reward(self, ac):
        """
        Computes reward of the current state.
        """
        return super()._compute_reward(ac)

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

            if c.geom1 in self.l_finger_geom_ids["right"] and body2_name == obj:
                touch_left_finger = True
            if c.geom2 in self.l_finger_geom_ids["right"] and body1_name == obj:
                touch_left_finger = True

            if c.geom1 in self.r_finger_geom_ids["right"] and body2_name == obj:
                touch_right_finger = True
            if c.geom2 in self.r_finger_geom_ids["right"] and body1_name == obj:
                touch_right_finger = True

        return touch_left_finger, touch_right_finger


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureSawyerEnv")
    parser.set_defaults(max_episode_steps=2000)
    parser.add_argument(
        "--run_mode", type=str, default="manual", choices=["manual", "vr", "demo"]
    )
    config, unparsed = parser.parse_known_args()
    if len(unparsed):
        logger.error("Unparsed argument is detected:\n%s", unparsed)
        return

    # create an environment and run manual control of Sawyer environment
    env = FurnitureSawyerEnv(config)
    if config.run_mode == "manual":
        env.run_manual(config)
    elif config.run_mode == "vr":
        env.run_vr(config)
    elif config.run_mode == "demo":
        env.run_demo_actions(config)


if __name__ == "__main__":
    main()
