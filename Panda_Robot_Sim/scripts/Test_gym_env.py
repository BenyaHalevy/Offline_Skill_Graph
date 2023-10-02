import gym
import time
import math
import panda_robot_sim
import pybullet as p
import numpy as np
from scripts.data_collection.State_class_rdc import*

deg2rad = math.pi/180

start = s16
end = s17
target_pos = end.pos
target_orn = end.orn

target_joints = end.jointPositions[:-2]

v2 = False


def main():
    env = gym.make('PandaKitchen-v5', gui=True, joint_positions=start.jointPositions, pos=start.pos, orn=start.orn, random=False)
    env.target_pos = np.array(target_pos)
    env.target_orn = np.array(target_orn)
    env.target_grip = end.gripper_command
    env.target_grip_state = end.gripper_state
    obs = env.reset()
    # print(obs)
    # exit()
    time.sleep(1)
    while True:
        if not v2:
            pos_act = env.target_pos - obs[:3]
            orn_act = env.target_orn - p.getEulerFromQuaternion(obs[3:7])
            grip_act = 0.0
            # print(max(pos_act))
            # if max(pos_act) < 0.01:
            #     grip_act = 1
            action = np.concatenate((pos_act, orn_act), axis=0)
            action = np.append(action, grip_act)
        else:
            action = np.array(target_joints) - np.array(obs[6:])
        obs, _, _, _ = env.step(action)
        print(obs)
        time.sleep(1 / 60)


if __name__ == '__main__':
    main()
