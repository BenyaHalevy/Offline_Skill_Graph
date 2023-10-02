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

sequence = False
sequence_list = [s9,s11,s12,s13,s15,s16,s17,s19]


class PandaKitchenEnv_goal(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui=False, joint_positions=joint_positions, pos=pos, orn=orn, grip_com=grip_com, steps=60, pos_tolerance=0.08, orn_tolerance=0.15, random=False, cube=False):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1, -1, -math.pi, -math.pi, -math.pi, -1], dtype=np.float32),
            high=np.array([1,  1,  1,  math.pi,  math.pi,  math.pi,  1], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1, -1, -1, -math.pi, -math.pi, -math.pi, 0, 0,
                          -1, -1, -1, -math.pi, -math.pi, -math.pi, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, math.pi, math.pi, math.pi, 1, 1,
                           1, 1, 1, math.pi, math.pi, math.pi, 1, 1], dtype=np.float32))
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
        self.sequence_count = 0

        self.pos_tolerance = pos_tolerance
        self.orn_tolerance = orn_tolerance

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
        # p.setGravity(0, -9.8, 0)

        self.panda_sim = None
        self.goal = None
        self.done = False
        # self.reset()

    def step(self, action):
        self.step_count += 1
        # get pos and orn actions and current pos and orn to create an absolute action command
        pos_act = np.clip(action[:3],-1, 1)
        # if min(abs(pos_act)) < 1:
        #     print(pos_act)
        orn_act = action[3:6]
        for idx, orn in enumerate(orn_act):
            if orn > math.pi:
                orn_act[idx] = orn - 2*math.pi
            elif orn < -math.pi:
                orn_act[idx] = orn + 2*math.pi
        current_pos = self.obs[:3]  # p.getLinkStates(self.panda_sim.panda, [11])[0][0]
        current_orn = self.obs[3:6]  # p.getEulerFromQuaternion(p.getLinkStates(self.panda_sim.panda, [11])[0][1])
        for idx, orn in enumerate(current_orn):
            if orn > math.pi:
                current_orn[idx] = orn - 2*math.pi
            elif orn < -math.pi:
                current_orn[idx] = orn + 2*math.pi
        pos = current_pos + pos_act
        orn = current_orn + orn_act

        # get gripper state
        grip_state = self.panda_sim.get_gripper_state()

        # apply actions and step simulation
        self.panda_sim.inverse_kinematics(pos, orn)
        if action[6] > 0.5:
            self.panda_sim.close_gripper()
            self.grip_command = 1.0
        elif action[6] < -0.5:
            self.panda_sim.open_gripper()
            self.grip_command = 0.0

        p.stepSimulation()

        # get new pos and orn for observation
        current_pos = np.array(p.getLinkStates(self.panda_sim.panda, [11])[0][0]).round(2)
        current_orn = np.array(p.getEulerFromQuaternion(p.getLinkStates(self.panda_sim.panda, [11])[0][1])).round(2)

        ''' check Kitchen environment status '''
        slide_door = self.panda_sim.get_Kitchen_state()[1]
        if slide_door < 0.1:
            slide_obs = -1.0
        elif slide_door > 0.23:
            slide_obs = 1.0
        else:
            slide_obs = 0.0

        self.obs = np.array(list(current_pos) + list(current_orn) + [self.grip_command] + [grip_state] + self.target_state)

        # calculate new pos and orn errors
        delta_pos = np.array(self.target_pos) - current_pos
        delta_orn = np.array(self.target_orn) - current_orn
        pos_err = np.max(abs(delta_pos))
        orn_err = np.max(abs(delta_orn))

        reward = 0.0
        self.done = False
        flag =True
        # generate reward
        # if pos_err < self.pos_tolerance and orn_err < self.orn_tolerance:
        #     reward = 1.0
        #     self.done = True
        # else:
        #     reward = 0.0

        if slide_door > 0.25:
            reward = 1.0
        #     self.done = True

        # # if self.cube:
        # if (0.2 - p.getBasePositionAndOrientation(self.cube_id)[0][1]) < 0.03 and np.max(np.abs(current_pos - p.getBasePositionAndOrientation(self.cube_id)[0])) < 0.02:
        #     reward = 1.0
        #     self.done = True

        # if grip_state == 1.0 and np.max(np.abs(current_pos - p.getBasePositionAndOrientation(self.cube_id)[0])) < 0.02:
        #     reward = 1.0
        #     self.done = True

        # if np.abs(0.2 - p.getBasePositionAndOrientation(self.cube_id)[0][1]) < 0.025:
        #     reward = 1.0

        ''' for testing full sequence '''
        if sequence:
            if self.sequence_count == 0:
                if slide_door > 0.26:
                    reward = 1.0
                    self.sequence_count += 1
                    self.switch_target(self.sequence_count)
                    flag = False
            elif flag and self.sequence_count >= 1 and self.sequence_count <= 3:
                if pos_err < self.pos_tolerance and orn_err < self.orn_tolerance:
                    reward = 1.0
                    self.sequence_count += 1
                    self.switch_target(self.sequence_count)
                    flag = False
            elif flag and self.sequence_count == 4:
                if grip_state == 1.0 and np.max(np.abs(current_pos - p.getBasePositionAndOrientation(self.cube_id)[0])) < 0.02:
                    reward = 1.0
                    self.sequence_count += 1
                    self.switch_target(self.sequence_count)
                    flag = False
            elif flag and self.sequence_count == 5:
                if (0.2 - p.getBasePositionAndOrientation(self.cube_id)[0][1]) < 0.03 and np.max(np.abs(current_pos - p.getBasePositionAndOrientation(self.cube_id)[0])) < 0.02:
                    reward = 1.0
                    self.sequence_count += 1
                    self.switch_target(self.sequence_count)
                    flag = False
            elif flag and self.sequence_count == 6:
                if pos_err < self.pos_tolerance and orn_err < self.orn_tolerance and np.max(np.abs(current_pos - p.getBasePositionAndOrientation(self.cube_id)[0])) < 0.02:
                    reward = 1.0
                    self.sequence_count += 1
                    self.switch_target(self.sequence_count)
                    flag = False
            elif flag and self.sequence_count == 7:
                if pos_err < self.pos_tolerance and orn_err < self.orn_tolerance and np.max(np.abs(self.target_pos - p.getBasePositionAndOrientation(self.cube_id)[0])) < 0.06:
                    reward = 1.0
                    self.sequence_count += 1
                    self.switch_target(self.sequence_count)
                    self.done = True

        if self.step_count == self.steps-1:
            self.done = True

        return self.obs, reward, self.done, {}

    def reset(self):
        # if self.random:
        #     start = rd.randint(0, 1)
        #     self.start_pos = state_list[start].pos
        #     self.start_orn = state_list[start].orn
        #     self.jointPositions = state_list[start].jointPositions
        #     if start == 1:
        #         self.target_pos = state_list[2].pos
        #         self.target_orn = state_list[2].orn
        #     else:
        #         target = rd.randint(1, 2)
        #         self.target_pos = state_list[target].pos
        #         self.target_orn = state_list[target].orn

        p.resetSimulation(self.client)
        p.setGravity(0, -9.8, 0)
        self.panda_sim = PandaSim(p, [0, 0, 0], self.jointPositions, self.start_pos, self.start_orn)
        if self.cube:
            self.cube_id = cube()
        current_pos = self.start_pos
        current_orn = self.start_orn
        grip_state = self.panda_sim.get_gripper_state()
        if sequence:
            self.sequence_count = 0
            # self.switch_target(self.sequence_count)
        for idx, orn in enumerate(current_orn):
            if orn > math.pi:
                current_orn[idx] = orn - 2*math.pi
            elif orn < -math.pi:
                current_orn[idx] = orn + 2*math.pi

        ''' check Kitchen environment status '''
        slide_door = 0.0
        if slide_door < 0.1:
            slide_obs = -1.0
        elif slide_door > 2.5:
            slide_obs = 1.0
        else:
            slide_obs = 0.0

        self.target_state = list(self.target_pos) + list(self.target_orn) + [self.target_grip] + [self.target_grip_state]
        self.obs = np.array(list(current_pos) + list(current_orn) + [self.grip_command] + [grip_state] + self.target_state)
        self.step_count = 0
        self.done = False
        return self.obs

    # def render(self):
    #     pass

    def close(self):
        p.disconnect(self.client)

    def switch_target(self, sequence_count):
        self.target_pos = sequence_list[sequence_count].pos
        self.target_orn = sequence_list[sequence_count].orn
        self.target_grip = sequence_list[sequence_count].gripper_command
        self.target_grip_state = sequence_list[sequence_count].gripper_state
        self.target_state = list(self.target_pos) + list(self.target_orn) + [self.target_grip] + [self.target_grip_state]

def cube():
    cube_id = p.loadURDF("cube_small.urdf", np.array([0.45, 0.1, 0.0]), [-0.5, -0.5, -0.5, 0.5], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    return cube_id


