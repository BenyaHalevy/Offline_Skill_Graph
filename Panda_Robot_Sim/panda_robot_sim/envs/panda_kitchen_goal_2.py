import gym
import math
import numpy as np
import random as rd
import pybullet as p
import pybullet_data as pd
from panda_robot_sim.resources.panda_Kitchen_sim import PandaSim
from scripts.data_collection.State_class_rdc import*

joint_positions = np.array([0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02])
pos = np.array([0.3, 0.5, -0.3])
orn = np.array([math.pi/2., 0, 0])
grip_com = 0.0
np.set_printoptions(precision=3)

class PandaKitchenEnv_goal_2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui=False, joint_positions=joint_positions, pos=pos, orn=orn, grip_com=grip_com, steps=60, pos_tolerance=0.08, orn_tolerance=0.15, random=False, cube=False):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1,  1,  1, 1, 1,  1,  1], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1, 0, 0, 0,
                          -1, -1, -1, -1, -1, -1, -1, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8,
                           1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32))
        self.random = random
        self.start_pos = pos
        self.start_orn = orn
        self.grip_command = grip_com
        self.grip_state = None
        self.jointPositions = joint_positions
        self.obs = None

        self.target_pos = None
        self.target_orn = None
        self.target_grip = None
        self.target_grip_state = None
        self.target_slide = None
        self.target_state = None

        self.cube = cube
        self.cube_id = None

        self.pos_tolerance = pos_tolerance
        self.orn_tolerance = orn_tolerance
        self.pos_scale = 0.2
        self.orn_scale = 0.5

        self.steps = steps
        self.step_count = 0

        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        p.setAdditionalSearchPath(pd.getDataPath())

        self.time_step = 1. / 60.
        p.setTimeStep(self.time_step, self.client)

        self.panda_sim = None
        self.goal = None
        self.done = False
        # self.reset()

    def step(self, action):
        self.step_count += 1
        # get pos and orn actions and current pos and orn to create an absolute action command
        action = np.clip(action, -1, 1)

        pos_act = action[:3]
        orn_act = action[3:6]
        gripper_act = action[6]
        current_pos = self.obs[:3]  # p.getLinkStates(self.panda_sim.panda, [11])[0][0]
        current_orn = p.getEulerFromQuaternion(self.obs[3:7])  # p.getEulerFromQuaternion(p.getLinkStates(self.panda_sim.panda, [11])[0][1])
        target_pos_act = current_pos + pos_act * self.pos_scale
        target_orn_act = current_orn + orn_act * self.orn_scale

        # get gripper state
        grip_state = self.panda_sim.get_gripper_state()
        grip_distance = self.panda_sim.get_grippers_distance()

        # apply actions and step simulation
        self.panda_sim.inverse_kinematics(target_pos_act, target_orn_act)
        if gripper_act > 0.5:
            self.panda_sim.close_gripper()
            self.grip_command = 1.0
        elif gripper_act < -0.5:
            self.panda_sim.open_gripper()
            self.grip_command = 0.0

        p.stepSimulation()

        # get new pos and orn for observation
        current_pos = np.array(p.getLinkStates(self.panda_sim.panda, [11])[0][0]).round(3)
        current_orn = np.array(p.getLinkStates(self.panda_sim.panda, [11])[0][1])

        ''' check Kitchen environment status '''
        slide_door = self.panda_sim.get_Kitchen_state()[1]  # not in use

        self.obs = np.array(list(current_pos) + list(current_orn) + [self.grip_command] + [grip_state] + [grip_distance]
                            + self.target_state)

        # calculate new pos and orn errors
        delta_pos = np.array(self.target_pos) - current_pos
        delta_orn = np.array(self.target_orn) - p.getEulerFromQuaternion(current_orn)
        pos_err = np.max(abs(delta_pos))
        orn_err = np.max(abs(delta_orn))

        reward = 0.0
        self.done = False
        # generate reward
        # if pos_err < self.pos_tolerance and orn_err < self.orn_tolerance:
        #     reward = 1.0
        #     self.done = True
        # else:
        #     reward = 0.0

        # if slide_door > 0.26:
        #     reward = 1.0
        #     self.done = True

        # # if self.cube:
        # if (0.2 - p.getBasePositionAndOrientation(self.cube_id)[0][1]) < 0.03 and np.max(np.abs(current_pos - p.getBasePositionAndOrientation(self.cube_id)[0])) < 0.02:
        #     reward = 1.0
        #     self.done = True

        # if grip_state == 1.0 and np.max(np.abs(current_pos - p.getBasePositionAndOrientation(self.cube_id)[0])) < 0.025:
        #     reward = 1.0
        #     self.done = True

        # if np.abs(0.2 - p.getBasePositionAndOrientation(self.cube_id)[0][1]) < 0.025:
        #     reward = 1.0

        if self.step_count == self.steps-1:
            self.done = True

        return self.obs, reward, self.done, {}

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, -9.8, 0)
        self.panda_sim = PandaSim(p, [0, 0, 0], self.jointPositions, self.start_pos, self.start_orn)
        if self.cube:
            self.cube_id = cube()
        current_pos = self.start_pos
        current_orn = p.getQuaternionFromEuler(self.start_orn)
        grip_state = self.panda_sim.get_gripper_state()
        grip_distance = self.panda_sim.get_grippers_distance()

        self.target_state = list(self.target_pos) + list(p.getQuaternionFromEuler(self.target_orn)) + [self.target_grip] + [self.target_grip_state]

        self.obs = np.array(list(current_pos) + list(current_orn) + [self.grip_command] + [grip_state] + [grip_distance] + self.target_state)
        self.step_count = 0
        self.done = False
        return self.obs

    # def render(self):
    #     pass

    def close(self):
        p.disconnect(self.client)


def cube():
    cube_id = p.loadURDF("cube_small.urdf", np.array([0.45, 0.1, 0.0]), [-0.5, -0.5, -0.5, 0.5], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    return cube_id



