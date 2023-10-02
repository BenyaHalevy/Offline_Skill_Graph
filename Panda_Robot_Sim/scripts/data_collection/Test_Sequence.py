import gym
import time
import math
import panda_robot_sim
import numpy as np
from scripts.data_collection.State_class_rdc import*

deg2rad = math.pi/180
new_state_list = state_list[16:21]


def main():
    idx = 1
    start = new_state_list[0]
    end = new_state_list[idx]
    env = gym.make('PandaKitchen-v3', gui=True, joint_positions=start.jointPositions, pos=start.pos, orn=start.orn, cube=True)
    env.target_pos = end.pos
    env.target_orn = end.orn
    obs = env.reset()
    time.sleep(5)
    while True:
        pos_act = env.target_pos - obs[:3]
        orn_act = env.target_orn - obs[3:6]
        grip_act = 0.0
        # pos_act = [0, 0, 0]
        # orn_act = [0, 0, 0]
        if max(abs(pos_act)) < 0.005:
            if idx == 1:
                grip_act = 0.0
            elif idx == 2:
                grip_act = 1.0
                if obs[7] == 1:
                    idx += 1
            else:
                grip_act = 0.0
            if idx != 2 and idx < 4:
                idx += 1
            print(idx)
            end = new_state_list[idx]
            env.target_pos = end.pos
            env.target_orn = end.orn
        pos_act = pos_act # + np.random.normal(0, 0.03, 3)
        orn_act = orn_act # + np.random.normal(0, 0.03, 3)
        action = np.concatenate((pos_act, orn_act), axis=0)
        action = np.append(action, grip_act)
        obs, reward, _, _ = env.step(action)
        print(reward)
        time.sleep(1 / 60)


if __name__ == '__main__':
    main()
