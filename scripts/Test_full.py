import gym
import time
import math
import panda_robot_sim
import pybullet as p
import numpy as np
from pathlib import Path
import torch
from Panda_Robot_Sim.scripts.data_collection.State_class_rdc import*
from State2Skill_Net import Net
import TD3_BC as td
from Planning.check import *

root_path = Path(__file__).absolute().parent.parent

device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else
gui = True

model = Net(10, 256, 256, 6)
model.load_state_dict(torch.load('State2Skill.pth'))
model.eval()

test_s1 = State([0.35,0.625,0.0],[90,0,45],[0.61,-1.1,-0.29,-2.4,-0.59,2.04,1.24,0.04,0.04])
test_s2 = State([0.375,0.35,-0.15],[90,0,0],[0.85,-0.5,-0.42,-2.47,-0.22,2.0,1.35,0.04,0.04])
test_s3 = State([0.425,0.575,0.15],[90,0,45],[0.57,-1.0,-0.66,-2.35,-0.65,2.26,0.95,0.04,0.04])

tested_states = [s6, s10, s11, s12, s13, test_s1, test_s2, test_s3]
states_pairs = [[s11,s12],[s12,s6],[s12,s13],[s13,s11],[s10,s11],[s6,s11],[s6,s8],[s8,s10],[s13,s15],[s15,s16],[s16,s17],[s17,s19]]
skill_states = ['S11-S12','S12-S6','S12-S13','S13-S11','S10-S11','S6-S11','S6-S8','S8-S10','S13-S15','S15-S16','S16-S17','S17-S19']
mean = []
std = []
policies = []
targets = []
for idx, states in enumerate(skill_states):
    normalize = np.load(f"{root_path}\\TD3_BC-main\models\\{states}\\normalize\\normalize.npz")

    mean = mean + [normalize['mean'][0]]
    std = std + [normalize['std'][0]]
    policies = policies + [torch.load(f"{root_path}\\TD3_BC-main\models\\{states}\\policy\Policy.pth")]
    targets = targets + [states_pairs[idx][1]]


def main():
    test_success = 0
    total_success = 0
    num_of_tests = 1000
    flag = True
    for idx1, start in enumerate(tested_states):
        for test in range(num_of_tests):
            if flag:
                env = gym.make('PandaKitchen-v5', gui=gui, joint_positions=start.jointPositions, pos=start.pos, orn=start.orn,
                               pos_tolerance=0.04, orn_tolerance=0.11, cube=True)
                flag = False
            else:
                env.start_pos = start.pos
                env.start_orn = start.orn
                env.jointPositions = start.jointPositions

            env.reset()

            noiseflag = 1

            while noiseflag:
                noise = np.random.normal(0, 0.15, 3)
                norm = np.linalg.norm(noise)

                if norm > 0.15:
                    noise = (noise / np.linalg.norm(noise)) * 0.15
                # print(norm, noise)
                v1 = start.pos + noise
                v2 = start.orn + np.random.normal(0, 0.2, 3)
                if np.linalg.norm(v1) > 1:
                    new_start_pos = v1 / np.linalg.norm(v1)
                else:
                    new_start_pos = v1
                noiseflag = (not (new_start_pos[1] > 0.15)) and (not (new_start_pos[0] < 0.5))
                # print(new_start_pos, noiseflag)

            new_start_orn = v2

            env.jointPositions = p.calculateInverseKinematics(env.panda_sim.panda, 11, new_start_pos, new_start_orn, residualThreshold=0.002)
            env.start_pos = new_start_pos
            env.start_orn = new_start_orn

            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])

            obs = env.reset()
            # p.resetJointState(env.panda_sim.kitchen, jointIndex=10, targetValue=0.28)
            # time.sleep(2)
            # print("--- Calculating first skill ---")
            # time.sleep(1)

            obs = obs.reshape(-1, obs.shape[0]).astype('float32')
            obs = torch.from_numpy(obs)
            output = model(obs)
            _, skill = torch.max(output.data, 1)
            skill = int(skill[0])
            # print("Init_skill = skill-", skill)
            # time.sleep(1)

            target = targets[skill]
            target_pos = np.ndarray.tolist(target.pos)
            quat = p.getQuaternionFromEuler(target.orn)
            target_orn = [round(quat[0], 3), round(quat[1], 3), round(quat[2], 3), round(quat[3], 3)]
            robot_initial = tuple(target_pos+target_orn+[target.gripper_command]+[target.gripper_state]+[target.jointPositions[-2]+target.jointPositions[-2]])

            # print("Initial graph node target:", robot_initial)
            # time.sleep(1)

            world_initial = {"door": "closed", "cube": "counter"}
            initial = (robot_initial, world_initial)
            skills = skill_dict
            robot_final_state = s_19
            world_final_state = {"door": "open", "cube": "placed"}
            goal = (robot_final_state, world_final_state)

            # print("--- Initiating planner ---")
            time.sleep(1)

            planner_out = solve_problems(initial, goal, skills)
            plan = [skill]
            for action in planner_out:
                plan = plan + [action[2]]
            # print(plan)

            actor = td.Actor(obs_dim, action_dim, max_action).to(device)

            success = 0
            sub_goal = robot_initial
            # print("---Executing plan---")
            # time.sleep(1)

            for idx, skill in enumerate(plan):
                # print(plan[idx])
                if idx > 0:
                    sub_goal = planner_out[idx-1][0]
                actor.load_state_dict(policies[plan[idx]])
                actor.eval()
                for i in range(250):
                    if not success:
                        state = (np.array(obs).reshape(1, -1) - mean[plan[idx]]) / std[plan[idx]]
                        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                        action = actor(state).cpu().data.numpy().flatten()
                    else:
                        action = np.array([0, 0, 0, 0, 0, 0, 0])
                        obs, _, _, _ = env.step(action)
                        if idx == len(plan)-1:
                            test_success += success
                            # print("---Goal Achieved---")
                            break
                        else:
                            env.panda_sim.forward_kinematics(targets[plan[idx]].jointPositions)
                            for j in range(60):
                                p.stepSimulation()
                                if gui:
                                    time.sleep(1 / 60)
                                # print(env.panda_sim.get_joint_poses())
                            break

                    obs, _, _, _ = env.step(action)
                    # print(obs)
                    # obs[:3] = obs[:3] + np.random.normal(0, 0.008, 3)
                    # max_err = max(abs(sub_goal - obs))
                    # print(max_err)
                    success = check_success(env, plan[idx], sub_goal, obs)
                    # if not success and not flag:
                    #     obs[:3] = obs[:3] + np.random.normal(0, 0.01, 3)
                    if gui:
                        time.sleep(1 / 60)
                if not success:
                    break
                success = 0

            success_rate = test_success / (test+1)
        print(f"test-{idx1}: {test} success rate is: {success_rate}")
        total_success = total_success + test_success
        test_success = 0
    total_success_rate = total_success/(num_of_tests * (idx1+1))
    print(f"total success rate is: {total_success_rate}")
    input("Press Enter to terminate")


def check_success(env, skill_idx, sub_goal, obs):
    success = 0
    if skill_idx < 6:
        success = np.max(abs(sub_goal - obs)) < 0.02
    if skill_idx == 2:
        success = np.max(abs(sub_goal[:3] - obs[:3])) < 0.01
    if skill_idx == 6:
        success = obs[8] == 1 and np.abs(obs[2] - (env.panda_sim.get_Kitchen_state()[1] + 0.05)) < 0.01 and obs[0] < 0.57 and obs[0] > 0.55
    if skill_idx == 7:
        success = obs[8] == 0.0 and env.panda_sim.get_Kitchen_state()[1] > 0.26 and obs[9] > 0.06 and np.abs(obs[1]-0.75) < 0.04
    if skill_idx == 8:
        success = obs[8] == 1 and np.max(np.abs(obs[:3] - np.array(p.getBasePositionAndOrientation(env.cube_id)[0]))) < 0.015
    if skill_idx == 9:
        success = obs[8] == 1 and np.abs(0.22 - p.getBasePositionAndOrientation(env.cube_id)[0][1]) < 0.02
    if skill_idx == 10:
        success = obs[7] == 1 and np.abs(0.75 - p.getBasePositionAndOrientation(env.cube_id)[0][1]) < 0.03
    if skill_idx == 11:
        success = obs[7] == 0 and obs[9] > 0.06 and np.max(np.abs(np.array([0.7, 0.6, 0.2]) - p.getBasePositionAndOrientation(env.cube_id)[0])) < 0.03
    return success


if __name__ == '__main__':
    main()

