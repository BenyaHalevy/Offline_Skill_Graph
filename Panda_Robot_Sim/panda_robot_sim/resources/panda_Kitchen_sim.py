import time
import numpy as np
import math
import os
import pybullet as pb

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11  # 8
pandaNumDofs = 7

# lower limits for null space
# ll = [-10]*pandaNumDofs
# upper limits for null space (todo: set them to proper range)
# ul = [10]*pandaNumDofs
# joint ranges for null space (todo: set them to proper range)
# jr = [10]*pandaNumDofs
# restposes for null space
jointPositions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
# rp = jointPositions
pos = np.array([0.3, 0.5, -0.3])
orn = np.array([math.pi/2., 0, 0])


class PandaSim(object):
    def __init__(self, bullet_client, offset, jointPositions = jointPositions, pos = pos, orn = orn):
        """ Loads the Panda robot into the simulation environment """

        self.bullet_client = bullet_client
        self.rp = jointPositions
        self.offset = np.array(offset)
        self.start_pos = [self.offset[0]+pos[0], self.offset[1]+pos[1], self.offset[2]+pos[2]]
        self.start_orn = [orn[0], orn[1], orn[2]]
        self.pos = []  # self.start_pos
        self.orn = []  # self.start_orn
        self.jointCommands = []
        self.jointPoses = []
        self.id_joints = list(range(11))
        self.finger_open = 1
        self.finger_closed = 0
        self.in_position = False
        self.grip_command = False
        self.grip_closed = False
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        ''' Load environment objects '''
        self.bullet_client.loadURDF("table/table.urdf", np.array([-0.4, -0.65, 0.65]) + self.offset, [-0.5, -0.5, -0.5, 0.5], useFixedBase=True, flags=flags)
        urdfpath = str(__file__).replace('panda_Kitchen_sim.py', 'kitchen\\urdf\kitchen.urdf')
        self.kitchen = self.bullet_client.loadURDF(urdfpath, np.array([0.4, -0.6, -0.5]) + self.offset, [0.5, 0.5, 0.5, -0.5], useFixedBase=True)

        ''' Load Panda Robot '''
        robot_orn = [-0.707107, 0.0, 0.0, 0.707107]  # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        # eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]) + self.offset,
                                                 robot_orn, useFixedBase=True, flags=flags)
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)

            jointName = info[1]
            jointType = info[2]

            if jointType == self.bullet_client.JOINT_PRISMATIC:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
            if jointType == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1

        self.reset()

    def reset(self):
        """ resets Panda robot initial position """
        self.pos = self.start_pos
        self.orn = self.bullet_client.getQuaternionFromEuler(self.start_orn)
        self.jointCommands = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, self.pos,
                                                                           self.orn, residualThreshold=0.002)
        pb.enableJointForceTorqueSensor(self.panda, 9, True)
        pb.enableJointForceTorqueSensor(self.panda, 10, True)
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.jointCommands[i], force=10 * 240.)
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_open, force=6*10)
        pass

    def inverse_kinematics(self, pos, orn):
        """Sets robot position and orientation using inverse kinematics.
        the inverse kinematics function calculates the joint position commands and sets them to the motor controllers

        :param pos: a 3 dimensional vector of the end effector absolute position
        :param orn: a 3 dimensional vector of the 3 angles of the end effector orientation
        :return: no return values, the function executes the motion from the calculation

        """
        self.in_position = False
        self.pos = pos
        self.orn = self.bullet_client.getQuaternionFromEuler(orn)
        self.jointCommands = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, self.pos,
                                                                           self.orn, residualThreshold=0.002)
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.jointCommands[i], force=10 * 240.)
        # for i in [9, 10]:
        #     self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
        #                                              self.finger_target, force=10)
        pass

    def get_IK_joint_commands(self, pos, orn):
        """
        returns the joint position commands calculated by the inverse kinematics function

        :param pos: a 3 dimensional vector of the end effector absolute position
        :param orn: a 3 dimensional vector of the 3 angles of the end effector orientation
        :return: a 7 dimentional vector of the 7 joint angle commands

        """
        self.in_position = False
        self.pos = pos
        self.orn = self.bullet_client.getQuaternionFromEuler(orn)
        self.jointCommands = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, self.pos,
                                                                           self.orn, residualThreshold=0.002)
        return self.jointCommands


    def forward_kinematics(self, joint_poses):
        """Sets robot position and orientation using forward kinematics.
        the forward kinematics function takes a vector of position commands and sets them to the motor controllers.

        :param joint_poses: a vector of joint poses that are set as position commands to the motor controllers
        :return: no return values

        """
        self.in_position = False
        self.jointCommands = joint_poses
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.jointCommands[i], force=10 * 240.)
        pass

    def check_in_position(self):
        """ checks if the robot is in the position it was set to.
        if the robot is in position, the in_position class variable is set to True

        :return: boolean: in_position

        """
        # TODO: add the gripper state to the in position check
        self.jointPoses = [i[0] for i in self.bullet_client.getJointStates(self.panda, self.id_joints)]
        errors = np.array(self.jointCommands[0:7]) - np.array(self.jointPoses[0:7])
        # print(self.jointPoses)
        if all(j < 0.02 for j in errors):
            self.in_position = True
            # print(self.in_position)
        return self.in_position

    def get_joint_poses(self):
        """ get angle values for all robot joints

        :return: list: jointPoses

        """
        # TODO: add the gripper state to the in position check
        self.jointPoses = [format(i[0], '.4f') for i in self.bullet_client.getJointStates(self.panda, self.id_joints)]

        return self.jointPoses

    def close_gripper(self):
        """Sets the robot gripper to a closed position.

        :return: no return values

        """
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_closed, force=10*100)
        self.grip_command = True
        pass

    def open_gripper(self):
        """Sets the robot gripper to an open position.

        :return: no return values

        """
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_open, force=6*10)
        self.grip_command = False
        pass

    def get_gripper_state(self):
        """ check if the gripper is gripping onto something by checking the forces on the gripper

        :return: Boolean: True for sending a grip
        """
        grip_forces = [i[2][0:3] for i in pb.getJointStates(self.panda, [9, 10])]
        force = np.linalg.norm(grip_forces)
        if self.grip_command == 1.0 and force > 2.5:
            self.grip_closed = 1.0
        else:
            self.grip_closed = 0.0

        return self.grip_closed

    def get_grippers_distance(self):
        """ check if the gripper is gripping onto something by checking the forces on the gripper

        :return: float: distance between grippers
        """
        grip_position = [i[0] for i in pb.getJointStates(self.panda, [9, 10])]
        distance = np.round(sum(grip_position),2)

        return distance

    def get_Kitchen_state(self):
        """ get values for all Kitchen joints

        :return: list: jointPoses

        """
        Kitchen_state = [i[0] for i in self.bullet_client.getJointStates(self.kitchen, [8, 10, 12, 14, 17, 20])]

        return Kitchen_state