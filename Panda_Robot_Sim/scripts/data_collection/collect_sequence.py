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
v4 = True

new_state_list = [s11, s12]

env = gym.make('PandaKitchen-v4', gui=gui, cube=False)
start = 0
goal = 1
scenarios = []
num_of_scenarios = 1
legend_df = pandas.DataFrame(columns=['start state', 'start joint positions', 'target state', 'start state index', 'target state index', 'success rate', 'file name'])
for scenario in range(num_of_scenarios):

    # while (start, end) in scenarios or start == end:
    #     start = random.randint(0, 5)
    #     end = random.randint(0, 5)
    scenarios = scenarios + [(start, goal)]

    start_state = new_state_list[0]  # new_state_list[start]
    env.start_pos = start_state.pos
    env.start_orn = start_state.orn
    env.jointPositions = start_state.jointPositions

    target_state = new_state_list[goal]
    env.target_pos = target_state.pos
    env.target_orn = target_state.orn
    # env.target_slide = 1.0
    env.target_grip = target_state.gripper_command
    env.target_grip_state = target_state.gripper_state
    target_joints = target_state.jointPositions[:-2]


    ''' for legend creation'''
    start_obs = env.start_pos.tolist() + env.start_orn.tolist()
    start_joint_positions = env.jointPositions
    target_obs = env.target_pos.tolist() + env.target_orn.tolist()

    num_of_episodes = 10000
    num_of_steps = 50
    total_success = 0

    # print(start_orn, target_orn)

    # df = pandas.DataFrame(columns=['episode', 'step', 'observations', 'actions', 'rewards', 'terminals', 'timeouts'])
    df_dict = collections.defaultdict(list)

    grip_act = 0.0

    for episode in range(num_of_episodes):
        obs = env.reset()
        cube = pb.loadURDF("cube_small.urdf", np.array([0.45, 0.1, 0.0]), [-0.5, -0.5, -0.5, 0.5])
        pb.resetJointState(env.panda_sim.kitchen, jointIndex=10, targetValue=0.28)
        # for i in range(2000):
        #     pb.stepSimulation()
        mid = 1
        mid_state = new_state_list[mid]
        count = 0
        done = False
        timeout = False
        pos_threshold = 0.015  # abs(np.random.normal(0.02, 0.01))
        orn_threshold = 0.08  # abs(np.random.normal(0.02, 0.01))
        # joint_threshold = abs(np.random.normal(0.02, 0.01))
        success = 0
        step_count = 0
        for step in range(num_of_steps):

            current_pos = obs[:3]
            if v4:
                current_orn = pb.getEulerFromQuaternion(obs[3:7])
            else:
                current_orn = obs[3:6]
            # current_slide = obs[7]

            ''' state error calculation '''
            delta_pos = mid_state.pos - current_pos
            delta_orn = mid_state.orn - current_orn
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

            slide_door = env.panda_sim.get_Kitchen_state()[1]
            state_success = pos_success and orn_success

            reward = 0.0
            # print(pb.getBasePositionAndOrientation(cube)[0][1])
            if pos_success and orn_success:
            # if mid == goal and obs[8] == 1 and slide_door < 0.01 and np.abs(current_pos[2] - (slide_door + 0.05)) < 0.01 and obs[9] > 0.02:
                reward = 1.0
                if success < 1:
                    success += 1
            # if mid == goal and obs[8] == 0.0 and slide_door > 0.26 and obs[9] > 0.06:
            #     reward = 1.0
            #     # print(step)
            #     if success < 1:
            #         success += 1
            # if pos_success and orn_success:
                # if (0.2 - pb.getBasePositionAndOrientation(env.cube_id)[0][1]) < 0.02:
                #     reward = 1
                #     # print(pb.getBasePositionAndOrientation(env.cube_id)[0][1])
                #     if success < 1:
                #         success += 1
            # ''' pickup reward '''
            # if mid == goal and obs[8] == 1 \
            #         and np.max(np.abs(current_pos - np.array(pb.getBasePositionAndOrientation(cube)[0]))) < 0.01 \
            #         and np.max(np.abs(mid_state.pos - current_pos)) < 0.015 and np.max(np.abs(mid_state.orn - current_orn)) < 0.1:
            #     reward = 1.0
            #     if success < 1:
            #         success += 1
            # ''' grab reward '''
            # if mid == goal and state_success and obs[7] == 0 and np.max(np.abs(current_pos - np.array(pb.getBasePositionAndOrientation(env.cube_id)[0]))) < 0.01:
            #     reward = 1.0
            #     if success < 1:
            #         success += 1
            # ''' place reward '''
            # if mid == goal and obs[8] == 0 and state_success and obs[9] > 0.06 and np.max(np.abs(np.array([0.7, 0.6, 0.2]) - pb.getBasePositionAndOrientation(cube)[0])) < 0.02:
            #     reward = 1.0
            #     # print(step)
            #     if success < 1:
            #         success += 1

            elif mid < goal and state_success:
                # print("here")
                if mid != 2 or (obs[7] == 1 and step_count > 10):
                    mid += 1
                    mid_state = new_state_list[mid]
                    # print("ok")
                elif mid == 2:
                    step_count += 1
                    # if step_count > 7:
                    #     reward = 0.3
                else:
                    reward = 0.0
            else:
                reward = 0.0
                # print(delta_orn)
            # if mid == 1 and slide_door > 0.26:
            #     mid += 1
            #     mid_state = new_state_list[mid]


            ''' adding noise '''

            if mid < 2:
                rel_pos_action = 6 * rel_pos_action + np.random.normal(0, 0.49, 3)
                rel_orn_action = 6 * rel_orn_action + np.random.normal(0, 0.45, 3)
                rel_pos_action = np.clip(rel_pos_action, -1, 1)
                rel_orn_action = np.clip(rel_orn_action, -1, 1)
            else:
                rel_pos_action = 6 * rel_pos_action + np.random.normal(0, 0.1, 3)
                rel_orn_action = 6 * rel_orn_action + np.random.normal(0, 0.1, 3)
                rel_pos_action = np.clip(rel_pos_action, -1, 1)
                rel_orn_action = np.clip(rel_orn_action, -1, 1)

            action = np.concatenate((rel_pos_action, rel_orn_action))

            if mid > 0 and mid < 2:
                grip_act = -1.0
            # elif mid == 4:
            #     grip_act = -1.0
            else:
                grip_act = -1.0

            action = np.append(action, grip_act)
            # print(action)

            ''' setting action to simulation '''

            next_obs, r, _, _ = env.step(action)
            # print(r)

            # if step == num_of_steps-1:
            #     done = True
            #     timeout = True

            # df_row = [episode, step, obs, action, reward, done, timeout]
            if mid > 0:
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
        if episode % 1000 == 0:
            print(episode, success, total_success/(episode+1), (pos_threshold, orn_threshold), (pos_success, orn_success))

    print(num_of_episodes, total_success/num_of_episodes, (start, goal))

    df = pandas.DataFrame.from_dict(df_dict)
    file_name = f'datasets/goal_2/S11-S12/S11 to S12.pkl'
    df.to_pickle(file_name)
    #
    file_name = f'datasets/goal_2/S11-S12/csv/S11 to S12.csv'
    df.to_csv(file_name)

#     # overwrite start obs
#     start_obs = s17.pos.tolist() + s16.orn.tolist()
#     start_joint_positions = s17.jointPositions.tolist()
#
#     legend_df_row = [start_obs, start_joint_positions, target_obs, start, goal, total_success/num_of_episodes, file_name]
#     # print(df_row)
#     legend_df.loc[scenario+1] = legend_df_row
# #
# legend_df.to_pickle(f'datasets/goal_2/S13-S15/Data Collection Legend.pkl')
# legend_df.to_csv(f'datasets/goal_2/S13-S15/csv/Data Collection Legend.csv')
# print(df)
