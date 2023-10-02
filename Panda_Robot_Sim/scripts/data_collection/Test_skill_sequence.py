import gym
import time
import math
import panda_robot_sim
import pybullet as p
import numpy as np
import torch
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from scripts.data_collection.State_class_rdc import*
# from TD3_BC import Actor
import TD3_BC as td
from pathlib import Path

root_path = Path(__file__).absolute().parent.parent.parent.parent

device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else

states_pairs = [[s11,s12],[s12,s13],[s13,s15],[s15,s16],[s16,s17],[s17,s19]]
skill_states = ['S11-S12','S12-S13','S13-S15','S15-S16','S16-S17','S17-S19']
mean = []
std = []
policies = []
for states in skill_states:
    normalize = np.load(f"{root_path}\TD3_BC-main\models\\{states}\\normalize\\normalize.npz")

    mean = mean + [normalize['mean'][0]]
    std = std + [normalize['std'][0]]
    policies = policies + [torch.load(f"{root_path}\TD3_BC-main\models\\{states}\\policy\Policy.pth")]


def main():
    stage = 2
    start = states_pairs[stage][0]
    target = states_pairs[stage][1]
    env = gym.make('PandaKitchen-v5', gui=True, joint_positions=start.jointPositions, pos=start.pos, orn=start.orn,pos_tolerance=0.02, orn_tolerance=0.11, cube=True)

    env.target_pos = np.array(target.pos)
    env.target_orn = np.array(target.orn)
    env.target_grip = target.gripper_command
    env.target_grip_state = target.gripper_state

    M = 256
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = td.Actor(obs_dim, action_dim, max_action).to(device)
    actor.load_state_dict(policies[stage])
    actor.eval()
    # obs = env.reset()
    # time.sleep(2)
    change_flag = True
    count = 0
    success = 0
    num_of_episodes = 500
    num_of_steps = 400
    for episode in range(num_of_episodes):
        reward = 0
        stage = 2
        set_stage(stage, env, actor)
        env.jointPositions = start.jointPositions
        obs = env.reset()
        p.resetJointState(env.panda_sim.kitchen, jointIndex=10, targetValue=0.28)
        # start_pos = start.pos #+ np.random.normal(0.0, 0.05, 3)
        # env.jointPositions = env.panda_sim.get_IK_joint_commands(start_pos, start.orn)
        # obs = env.reset()
        # obs[:3] = obs[:3] + np.random.normal(0.0, 0.01, 3)
        for step in range(num_of_steps):
            if not reward:
                state = (np.array(obs).reshape(1, -1) - mean[stage]) / std[stage]
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action = actor(state).cpu().data.numpy().flatten()
            else:
                action = np.array([0, 0, 0, 0, 0, 0, 0])
                if stage == 8:
                    success += 1
                    print(step)
                    break

                # count += 1
                if stage < 8 and count == 0:
                    stage += 1
                    target = states_pairs[stage][1]
                    env.target_pos = np.array(target.pos)
                    env.target_orn = np.array(target.orn)
                    env.target_grip = target.gripper_command
                    env.target_grip_state = target.gripper_state
                    actor.load_state_dict(policies[stage])
                    actor.eval()
                    count = 0

            # action[:3] = action[:3]  # + np.random.normal(0.0, 0.2, 3)
            obs, _, _, _ = env.step(action)
            reward = calculate_reward(stage, env, obs[:3], obs[3:7])
            # if not reward:
                # obs[:3] = obs[:3] + np.random.normal(0.0, 0.01, 3)

            time.sleep(1 / 60)

    success_rate = success/num_of_episodes
    print(success_rate)


def calculate_reward(stage, env, current_pos, current_orn):

    delta_pos = np.array(env.target_pos) - current_pos
    delta_orn = np.array(env.target_orn) - p.getEulerFromQuaternion(current_orn)
    pos_err = np.max(abs(delta_pos))
    orn_err = np.max(abs(delta_orn))
    grip_state = env.panda_sim.get_gripper_state()
    grip_distance = env.panda_sim.get_grippers_distance()

    reward = 0
    if stage < 2:
        if pos_err < env.pos_tolerance and orn_err < env.orn_tolerance:
            reward = 1.0
    elif stage == 2:
        if grip_state == 1.0 and grip_distance < 0.05 and np.max(np.abs(current_pos - p.getBasePositionAndOrientation(env.cube_id)[0])) < 0.015:
            reward = 1.0
    elif stage == 3:
        if np.abs(0.2 - p.getBasePositionAndOrientation(env.cube_id)[0][1]) < 0.015:
            reward = 1.0
    elif stage == 4:
        if env.obs[7] == 1 and np.abs(0.75 - p.getBasePositionAndOrientation(env.cube_id)[0][1]) < 0.03:
            reward = 1.0
    elif stage == 5:
        if env.obs[8] == 0 and env.obs[9] > 0.06 and np.max(np.abs(np.array([0.7, 0.6, 0.2]) - p.getBasePositionAndOrientation(env.cube_id)[0])) < 0.02:
            reward = 1.0

    return reward


def set_stage(stage, env, actor):
    target = states_pairs[stage][1]
    env.target_pos = np.array(target.pos)
    env.target_orn = np.array(target.orn)
    env.target_grip = target.gripper_command
    env.target_grip_state = target.gripper_state
    actor.load_state_dict(policies[stage])
    actor.eval()


if __name__ == '__main__':
    main()

