import gym
import time
import math
import panda_robot_sim
import numpy as np
import pybullet as pb
from scripts.data_collection.State_class_rdc import*


deg2rad = math.pi/180
start = s17
end = s18


def main():
    env = gym.make('PandaKitchen-v5', gui=True, joint_positions=start.jointPositions, pos=start.pos, orn=start.orn, cube = True)
    env.target_pos = end.pos
    env.target_orn = end.orn
    time.sleep(3)
    obs = env.reset()
    # cube = pb.loadURDF("cube_small.urdf", np.array([0.4, 0.75, 0.1]), [-0.5, -0.5, -0.5, 0.5])
    while True:
        pos_act = env.target_pos - obs[:3]
        orn_act = env.target_orn - obs[3:6]
        # pos_act = [0, 0, 0]
        orn_act = [0, 0, 0]
        grip_act = [1]
        action = np.concatenate((pos_act, orn_act, grip_act), axis=0)
        obs, _, _, _ = env.step(action)
        time.sleep(1 / 60)


if __name__ == '__main__':
    main()
