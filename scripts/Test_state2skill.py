import gym
import time
import math
import panda_robot_sim
import pybullet as p
import numpy as np
from pathlib import Path
import torch
from scripts.data_collection.State_class_rdc import*
from State2Skill_Net import Net
import TD3_BC as td

root_path = Path(__file__).absolute().parent.parent

device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else

model = Net(10, 256, 256, 6)
model.load_state_dict(torch.load('State2Skill.pth'))
model.eval()

test_s1 = State([0.35,0.625,0.0],[90,0,45],[0.61,-1.1,-0.29,-2.4,-0.59,2.04,1.24,0.04,0.04])
test_s2 = State([0.375,0.35,-0.15],[90,0,0],[0.85,-0.5,-0.42,-2.47,-0.22,2.0,1.35,0.04,0.04])
test_s3 = State([0.425,0.575,0.15],[90,0,45],[0.57,-1.0,-0.66,-2.35,-0.65,2.26,0.95,0.04,0.04])

tested_states = [s6, s10, s11, s12, s13, test_s1, test_s2, test_s3]
states_pairs = [[s11, s12],[s12,s6],[s12,s13],[s13,s11],[s10,s11],[s6,s11]]
skill_states = ['S11-S12','S12-S6','S12-S13','S13-S11','S10-S11','S6-S11']
mean = []
std = []
policies = []
targets = []
for idx, states in enumerate(skill_states):
    normalize = np.load(f"{root_path}\TD3_BC-main\models\\{states}\\normalize\\normalize.npz")

    mean = mean + [normalize['mean'][0]]
    std = std + [normalize['std'][0]]
    policies = policies + [torch.load(f"{root_path}\TD3_BC-main\models\\{states}\\policy\Policy.pth")]
    targets = targets + [states_pairs[idx][1]]


def main():
    test_success = 0
    total_success = 0
    num_of_tests = 1000
    flag = True
    for idx, start in enumerate(tested_states):
        for test in range(num_of_tests):
            # time.sleep(2)
            if flag:
                env = gym.make('PandaKitchen-v5', gui=False, joint_positions=start.jointPositions, pos=start.pos, orn=start.orn,
                               pos_tolerance=0.04, orn_tolerance=0.11, cube=False)
                flag = False
            else:
                env.start_pos = start.pos
                env.start_orn = start.orn
                env.jointPositions = start.jointPositions

            env.reset()

            noise = np.random.normal(0, 0.15, 3)
            norm = np.linalg.norm(noise)
            if norm > 0.15:
                noise = (noise/np.linalg.norm(noise)) * 0.15
                # norm = np.linalg.norm(noise)
            # print(norm, noise)
            v1 = start.pos + noise
            v2 = start.orn + np.random.normal(0, 0.2, 3)
            if np.linalg.norm(v1) > 1:
                new_start_pos = v1 / np.linalg.norm(v1)
            else:
                new_start_pos = v1
            new_start_orn = v2

            env.jointPositions = p.calculateInverseKinematics(env.panda_sim.panda, 11, new_start_pos, new_start_orn, residualThreshold=0.002)
            env.start_pos = new_start_pos
            env.start_orn = new_start_orn

            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])

            obs = env.reset()
            # time.sleep(2)
            # print("--- Calculating first skill ---")
            # time.sleep(1)

            min = 999
            for j, target in enumerate(targets):
                dist = math.sqrt((target.pos[0]-obs[0])**2 + (target.pos[1]-obs[1])**2 + (target.pos[2]-obs[2])**2)
                if dist< min:
                    min = dist
                    skill = j
                # print(min, skill, j)

            # exit()

            # obs = obs.reshape(-1, obs.shape[0]).astype('float32')
            # obs = torch.from_numpy(obs)
            # output = model(obs)
            # _, skill = torch.max(output.data, 1)
            # skill = int(skill[0])

            # print("Init_skill = skill-", skill)
            # time.sleep(1)

            target = targets[skill]
            target_pos = np.ndarray.tolist(target.pos)
            quat = p.getQuaternionFromEuler(target.orn)
            target_orn = [round(quat[0], 3), round(quat[1], 3), round(quat[2], 3), round(quat[3], 3)]
            robot_initial = tuple(target_pos+target_orn+[target.gripper_command]+[target.gripper_state]+[target.jointPositions[-2]+target.jointPositions[-2]])

            # print("Initial graph node target:", robot_initial)
            # time.sleep(1)

            actor = td.Actor(obs_dim, action_dim, max_action).to(device)
            actor.load_state_dict(policies[skill])
            actor.eval()

            success = 0
            sub_goal = robot_initial
            # print("---Executing plan---")
            # time.sleep(1)
            for i in range(200):
                if not success:
                    state = (np.array(obs).reshape(1, -1) - mean[skill]) / std[skill]
                    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                    action = actor(state).cpu().data.numpy().flatten()
                else:
                    action = np.array([0, 0, 0, 0, 0, 0, 0])
                    test_success += 1
                    # print("---Goal Achieved---")
                    break

                obs, _, _, _ = env.step(action)
                max_err = max(abs(sub_goal - obs))
                success = max_err < 0.03

                # time.sleep(1 / 60)

            success_rate = test_success / (test+1)
        print(f"test-{idx}: {test} success rate is: {success_rate}")
        total_success = total_success + test_success
        test_success = 0
    total_success_rate = total_success/(num_of_tests * (idx+1))
    print(f"total success rate is: {total_success_rate}")
    input("Press Enter to terminate")


if __name__ == '__main__':
    main()
