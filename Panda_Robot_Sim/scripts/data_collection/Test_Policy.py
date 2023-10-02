import gym
import time
import math
import panda_robot_sim
import numpy as np
import torch
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from scripts.data_collection.State_class_rdc import*
# from TD3_BC import Actor
import TD3_BC as td
from graph_components import *
from pathlib import Path

root_path = Path(__file__).absolute().parent.parent.parent.parent

states_pairs = [[s11,s12],[s12,s6],[s12,s13],[s13,s11],[s10,s11],[s6,s11],[s6,s8],[s8,s10],[s13,s15],[s15,s16],[s16,s17],[s17,s19]]

movement = "S10-S11"
subgoal = s_11
skill_idx = 4
device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else

normalize = np.load(f"{root_path}\TD3_BC-main\models\\{movement}\\normalize\\normalize.npz")

mean = normalize['mean'][0]
std = normalize['std'][0]

def main():

    start = s10
    target = s11
    gui = False

    env = gym.make('PandaKitchen-v5', gui=gui, joint_positions=start.jointPositions, pos=start.pos, orn=start.orn, pos_tolerance=0.02, orn_tolerance=0.1, cube=True)
    env.target_pos = np.array(target.pos)
    env.target_orn = np.array(target.orn)
    env.target_grip = target.gripper_command
    env.target_grip_state = target.gripper_state
    # env.target_slide = 1.0
    # M = 256
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = td.Actor(obs_dim, action_dim, max_action).to(device)
    actor.load_state_dict(torch.load(f"{root_path}\TD3_BC-main\models\\{movement}\policy\Policy.pth"))
    actor.eval()
    reward = 0
    obs = env.reset()

    change_flag = True
    counter = 9
    flag = 0
    timer = 0

    numOfTests = 500
    test_success = 0
    for test in range(numOfTests):
        obs = env.reset()
        # print(obs)
        if gui:
            time.sleep(5)
        for step in range(100):
            if not reward:
                state = (np.array(obs).reshape(1, -1) - mean) / std
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action = actor(state).cpu().data.numpy().flatten()
                pass
            else:
                action = np.array([0, 0, 0, 0, 0, 0, 0])
                obs, _, _, _ = env.step(action)
                test_success += reward
                # print("---Goal Achieved---")

                if gui:
                    for j in range(20):
                        p.stepSimulation()
                        time.sleep(1 / 60)
                break

            timer += 1
            # print(action)
            obs, _, _, _ = env.step(action)
            # obs[:3] = obs[:3] + np.random.normal(0, 0.01, 3)
            reward = check_success(env, skill_idx, subgoal, obs)
            # print(reward)

            # print(obs)
            if gui:
                time.sleep(1 / 60)

        reward = 0

    total_success_rate = test_success / numOfTests
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


