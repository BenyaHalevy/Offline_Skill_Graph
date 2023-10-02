import gym
import math
import numpy as np
import pybullet as p
import pybullet_data as pd
from panda_robot_sim.resources.panda_Kitchen_sim import PandaSim

joint_positions = np.array([0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02])
pos = np.array([0.3, 0.5, -0.3])
orn = np.array([math.pi/2., 0, 0])


class PandaKitchenEnv_v2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui=False, joint_positions=joint_positions, pos=pos, orn=orn, steps=60, pos_tolerance=0.08, orn_tolerance=0.15):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi], dtype=np.float32),
            high=np.array([math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1, -1, -1, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi], dtype=np.float32),
            high=np.array([1, 1, 1, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi], dtype=np.float32))

        self.start_pos = pos
        self.start_orn = orn
        self.jointPositions = joint_positions
        self.target_pos = None
        self.target_orn = None

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
        p.setGravity(0, -9.8, 0)

        self.panda_sim = None
        self.goal = None
        self.done = False
        self.reset()

    def step(self, action):
        self.step_count += 1
        joints = self.panda_sim.get_joint_poses()[:-4]
        joint_commands = np.array(joints) + action

        self.panda_sim.forward_kinematics(joint_commands)
        p.stepSimulation()

        current_pos = p.getLinkStates(self.panda_sim.panda, [11])[0][0]
        current_orn = p.getEulerFromQuaternion(p.getLinkStates(self.panda_sim.panda, [11])[0][1])
        jointPoses = self.panda_sim.get_joint_poses()

        obs = np.array(list(current_pos) + list(current_orn) + list(jointPoses[:-4]))

        delta_pos = np.array(self.target_pos) - current_pos
        delta_orn = np.array(self.target_orn) - current_orn
        pos_err = np.sqrt(sum(np.square(abs(delta_pos))))
        orn_err = np.sqrt(sum(np.square(abs(delta_orn))))

        if pos_err < self.pos_tolerance and orn_err < self.orn_tolerance:
            reward = 1.0
        else:
            reward = 0.0

        # reward = - max(pos_err,orn_err)

        # if reward > 0:
        #     self.done = True

        if self.step_count == self.steps-1:
            self.done = True

        return obs, reward, self.done, {}

    def reset(self):
        p.resetSimulation(self.client)
        self.panda_sim = PandaSim(p, [0, 0, 0], self.jointPositions, self.start_pos, self.start_orn)
        current_pos = self.start_pos
        current_orn = self.start_orn
        obs = np.array(list(current_pos) + list(current_orn) + list(self.jointPositions[:-2]))
        self.step_count = 0
        self.done = False
        return obs

    # def render(self):
    #     pass

    def close(self):
        p.disconnect(self.client)

    # def seed(self, seed=None):
        # pass

    def get_dataset(self):
        # TODO -> create get dataset function that also sets the start position and the target position  of the env accordingly after the simulation was initialized but before running
        pass

