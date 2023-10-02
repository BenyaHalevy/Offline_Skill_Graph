import gym
import pybullet as pb
import pybullet_data as pd
import time
import math
import random
import collections
import pandas
import panda_robot_sim
from scripts.data_collection.State_class_rdc import*

deg2rad = math.pi/180
gui = False

v2 = False
v4 = True

env = gym.make('PandaKitchen-v4', gui=gui, cube=True)
start = 13
end = 11
scenarios = []
num_of_scenarios = 1
legend_df = pandas.DataFrame(columns=['start state', 'start joint positions', 'target state', 'start state index', 'target state index', 'success rate', 'file name'])
for scenario in range(num_of_scenarios):

    # while (start, end) in scenarios or start == end:
    #     start = random.randint(0, 5)
    #     end = random.randint(0, 5)
    scenarios = scenarios + [(start, end)]

    start_state = state_list[start]
    env.start_pos = start_state.pos
    env.start_orn = start_state.orn
    env.grip_state = start_state.gripper_state
    env.grip_command = start_state.gripper_command
    env.jointPositions = start_state.jointPositions

    target_state = state_list[end]
    env.target_pos = target_state.pos
    env.target_orn = target_state.orn
    env.target_grip = target_state.gripper_command
    env.target_grip_state = target_state.gripper_state
    target_joints = target_state.jointPositions[:-2]

    ''' for legend creation'''
    start_obs = env.start_pos.tolist() + env.start_orn.tolist()
    start_joint_positions = env.jointPositions
    target_obs = env.target_pos.tolist() + env.target_orn.tolist()

    num_of_episodes = 10000
    num_of_steps = 70
    total_success = 0

    # print(start_orn, target_orn)

    # df = pandas.DataFrame(columns=['episode', 'step', 'observations', 'actions', 'rewards', 'terminals', 'timeouts'])
    df_dict = collections.defaultdict(list)

    grip_act = 0.0

    for episode in range(num_of_episodes):
        obs = env.reset()
        done = False
        timeout = False
        pos_threshold = abs(np.random.normal(0.02, 0.01))
        orn_threshold = abs(np.random.normal(0.02, 0.01))
        # joint_threshold = abs(np.random.normal(0.02, 0.01))
        success = 0
        for step in range(num_of_steps):

            current_pos = obs[:3]
            if v4:
                current_orn = pb.getEulerFromQuaternion(obs[3:7])
            else:
                current_orn = obs[3:6]
            if v2:
                current_joints = obs[6:]

            ''' state error calculation '''
            delta_pos = env.target_pos - current_pos
            delta_orn = env.target_orn - current_orn
            for idx, orn in enumerate(delta_orn):
                if orn > math.pi:
                    delta_orn[idx] = orn - 2 * math.pi
                elif orn < -math.pi:
                    delta_orn[idx] = orn + 2 * math.pi
            distance = np.max(abs(delta_pos))
            angle = np.max(abs(delta_orn))
            # print(distance, angle)

            ''' generating action '''
            if distance > pos_threshold:
                rel_pos_action = delta_pos
                pos_success = False
            else:
                rel_pos_action = np.array([0, 0, 0])
                pos_success = True

            if angle > orn_threshold:
                rel_orn_action = delta_orn
                orn_success = False
            else:
                rel_orn_action = np.array([0, 0, 0])
                orn_success = True

            if v2 and (distance > pos_threshold or angle > orn_threshold):
                rel_joint_act = target_joints - current_joints

            if pos_success and orn_success:
                reward = 1.0
                if success < 1:
                    success += 1
                # break
            else:
                reward = 0.0
                # print(delta_orn)

            ''' adding noise '''
            if not v2:
                rel_pos_action = 5 * rel_pos_action + np.random.normal(0, 0.4, 3)
                rel_orn_action = 5 * rel_orn_action + np.random.normal(0, 0.35, 3)
                rel_pos_action = np.clip(rel_pos_action, -1, 1)
                rel_orn_action = np.clip(rel_orn_action, -1, 1)
                action = np.concatenate((rel_pos_action, rel_orn_action))
                action = np.append(action, grip_act)
                # print(action)
            else:
                action = rel_joint_act + np.random.normal(0, 0.15, 7)

            ''' setting action to simulation '''

            next_obs, _, _, _ = env.step(action)

            # if step == num_of_steps-1:
                # done = True
                # timeout = True

            # df_row = [episode, step, obs, action, reward, done, timeout]
            # df.loc[episode * num_of_steps + step] = df_row

            df_dict['episode'].append(episode)
            df_dict['step'].append(step)
            df_dict['observations'].append(obs)
            df_dict['actions'].append(action)
            df_dict['rewards'].append(reward)
            df_dict['terminals'].append(done)
            df_dict['timeouts'].append(timeout)

            obs = next_obs
            if gui:
                time.sleep(env.time_step)  # for using GUI

        total_success = total_success + success
        if episode % 100 == 0:
            print(episode, success, total_success/(episode+1), (pos_threshold, orn_threshold), (pos_success, orn_success))

    print(num_of_episodes, total_success/num_of_episodes, (start, end))

    df = pandas.DataFrame.from_dict(df_dict)

    file_name = f'datasets/goal_2/S{start}-S{end}/S{start} to S{end}.pkl'
    df.to_pickle(file_name)

    file_name = f'datasets/goal_2/S{start}-S{end}/csv/S{start} to S{end}.csv'
    df.to_csv(file_name)

    legend_df_row = [start_obs, start_joint_positions, target_obs, start, end, total_success/num_of_episodes, file_name]
    # print(df_row)
    legend_df.loc[scenario+1] = legend_df_row

legend_df.to_pickle(f'datasets/goal_2/S{start}-S{end}/Data Collection Legend.pkl')
legend_df.to_csv(f'datasets/goal_2/S{start}-S{end}/csv/Data Collection Legend.csv')
# print(df)
