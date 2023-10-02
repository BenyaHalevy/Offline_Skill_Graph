import numpy as np
import torch
import gym
import argparse
import os
from pathlib import Path
import d4rl

import utils
import TD3_BC

import pandas as pd
import panda_robot_sim
from scripts.data_collection.State_class_rdc import*
import random

torch.cuda.empty_cache()
use_board = False
start = s16
end = s17
movement = "S16-S17"
my_data = "S16 to S17"

root_path = Path(__file__).absolute().parent.parent

if use_board:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f"runs\\single\\no_goal\\{movement}_new\\new_batch_3000_eval_10k_plr_3e-5_clr_1e-5_saved")

df = pd.read_pickle(f"{root_path}\\Panda_Robot_Sim\scripts\data_collection\datasets\\no_goal\\{movement}\\{my_data}.pkl")

torch.manual_seed(10)
torch.cuda.manual_seed(10)
np.random.seed(10)
random.seed(10)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, seed, mean, std, seed_offset=100, eval_episodes=5, epoch=0):
	# eval_env = gym.make(env_name)
	# eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	if use_board:
		writer.add_scalar('Average Returns', avg_reward, epoch)
	# d4rl_score = eval_env.get_normalized_score(avg_reward)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--env", default="PandaKitchen-v5")         # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=10000, type=int)      # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=0.25e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", default=True, action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="default")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=3000, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	# os.makedirs(f"./results/{movement}_new")
	# os.makedirs(f"./models/{movement}_new/normalize")
	# os.makedirs(f"./models/{movement}_new/policy")
	# env = gym.make(args.env)

	env = gym.make(args.env, gui=False, joint_positions=start.jointPositions, pos=start.pos,
						orn=start.orn, steps=150, pos_tolerance=0.02, orn_tolerance=0.1, cube=True)
	env.target_pos = end.pos
	env.target_orn = end.orn
	# eval_env.target_slide = 0.0
	env.target_grip = end.gripper_command
	env.target_grip_state = end.gripper_state

	# Set seeds
	# env.seed(args.seed)
	# env.action_space.seed(args.seed)
	# torch.manual_seed(args.seed)
	# np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	# print(state_dim, action_dim, max_action)

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	policy = TD3_BC.TD3_BC(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{movement}_new/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env, dataset=df))
	if args.normalize:
		mean,std = replay_buffer.normalize_states()
		np.savez(f"./models/{movement}_new/normalize/normalize", mean=mean, std=std)
		# print(mean, std)
	else:
		mean,std = 0,1
	
	evaluations = []
	for t in range(int(args.max_timesteps)):
		critic_loss, actor_loss, lmbda, q_mean, mse_loss = policy.train(replay_buffer, args.batch_size)

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			epoch = (t+1)/args.eval_freq + 50
			if use_board:
				writer.add_scalar('critic_loss', critic_loss, epoch)
				writer.add_scalar('actor_loss', actor_loss, epoch)
				writer.add_scalar('lambda', lmbda, epoch)
				writer.add_scalar('Q_mean', q_mean, epoch)
				writer.add_scalar('actor_mse_loss', mse_loss, epoch)

			evaluations.append(eval_policy(policy, env, args.seed, mean, std, epoch=epoch))
			np.save(f"./results/{movement}_new/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{movement}_new/{file_name}")

	torch.save(policy.actor.state_dict(), f"./models/{movement}_new/policy/Policy.pth")
	# np.savez(f"./models/{movement}/normalize/normalize", mean=mean, std=std)
