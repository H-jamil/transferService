import gym
import numpy as np
from utils import plot_learning_curve
import sys
from gym import spaces
import copy
import random
from datetime import datetime
import logging as log
import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback,CheckpointCallback,CallbackList
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# class NormalizedEnv(gym.Wrapper):
#     def __init__(self, env, alpha=0.99, max_abs_reward=None):
#         super(NormalizedEnv, self).__init__(env)
#         self.env = env
#         self.observation_space = env.observation_space
#         self.action_space = env.action_space

#         self.alpha = alpha  # Weight factor for exponential moving average
#         self.obs_mean = np.zeros(self.observation_space.shape)
#         self.obs_var = np.ones(self.observation_space.shape)
#         self.num_steps = 0  # Count of steps to handle initial variance calculation

#         self.max_abs_reward = max_abs_reward if max_abs_reward is not None else 10000

#     def update_obs_stats(self, observation):
#         self.num_steps += 1
#         last_mean = self.obs_mean.copy()
#         self.obs_mean = self.alpha * self.obs_mean + (1 - self.alpha) * observation
#         self.obs_var = self.alpha * self.obs_var + (1 - self.alpha) * np.square(observation - last_mean)

#         if self.num_steps < 100:
#             # Adjust variance for the initial steps
#             self.obs_var = self.obs_var / (1 - self.alpha ** self.num_steps)

#     def normalize_observation(self, observation):
#         obs_std = np.sqrt(self.obs_var)
#         normalized_obs = (observation - self.obs_mean) / obs_std
#         return normalized_obs

#     def normalize_reward(self, reward):
#         normalized_reward = reward / self.max_abs_reward
#         return normalized_reward

#     def step(self, action):
#         observation, reward, done, info = self.env.step(action)
#         self.update_obs_stats(observation)
#         normalized_observation = self.normalize_observation(observation)
#         normalized_reward = self.normalize_reward(reward)
#         return normalized_observation, normalized_reward, done, info

#     def reset(self):
#         observation = self.env.reset()
#         self.update_obs_stats(observation)
#         normalized_observation = self.normalize_observation(observation)
#         return normalized_observation


# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from transferService import transferService
from optimizer_gd import *
from transferClass import *
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

if __name__ == '__main__':
    # env_string = 'CartPole-v0'
    env_string = 'transferService'
    # env = gym.make(env_string)
    optimizer='ppo_test'
    for handler in log.root.handlers[:]:
      log.root.removeHandler(handler)
    log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
    extraString="logFile"
    # Create the directory if it doesn't exist
    directory = f"./logFileDir/{optimizer}/"
    os.makedirs(directory, exist_ok=True)
    log_file = f"logFileDir/{optimizer}/{optimizer}_{extraString}_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    log.basicConfig(
        format=log_FORMAT,
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=log.INFO,
        handlers=[
            log.FileHandler(log_file),
            log.StreamHandler()
        ]
    )
    REMOTE_IP = "129.114.109.231"
    REMOTE_PORT = "80"
    INTERVAL = 1
    INTERFACE = "eno1"
    SERVER_IP = '127.0.0.1'
    SERVER_PORT = 8080
    transfer_service = transferService(REMOTE_IP, REMOTE_PORT, INTERVAL, INTERFACE, SERVER_IP, SERVER_PORT, optimizer, log)
    env = transferClass(transfer_service,optimizer)
    env = Monitor(env)  # Wrap with Monitor for advanced logging
    env = DummyVecEnv([lambda: env])  # Optional: Use a vectorized environment
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # custom_policy = ActorCriticPolicy(
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     lr_schedule=lambda _: 0.0003,  # Learning rate, adjust as needed
    #     features_extractor_class=CustomActorCriticNetwork,
    #     features_extractor_kwargs={"features_dim": 256}
    # )
    policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[{'pi': [128, 128], 'vf': [128, 128]}])
    # model = PPO("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1,tensorboard_log="./ppo_tensorboard/",ent_coef=0.01)
    model = PPO.load("./ppo_checkpoints/ppo_model_4000_steps.zip")
    done = False
    episode_reward = 0
    obs = env.reset()


    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        print("action: ",action)
        obs, reward, done, info = env.step([action])
        print("obs: ", obs,".... reward: ",reward)
        episode_reward += reward[0]

    env.close()
    print(f"Episode reward: {episode_reward}")


    # model = PPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,
    #     tensorboard_log="./ppo_tensorboard/",
    #     ent_coef=0.01
    # )

  #   eval_callback = EvalCallback(env, best_model_save_path='./ppo_best_model/',
  #                              log_path='./ppo_logs/', eval_freq=1000,
  #                              deterministic=True, render=False)

  # # Callback for saving checkpoints every 1000 timesteps
  #   checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./ppo_checkpoints/',
  #                                          name_prefix='ppo_model')

  # # Combine both callbacks
  #   callback = CallbackList([checkpoint_callback, eval_callback])
  #   model.learn(total_timesteps=10000, callback=callback)

  #   # N = 20
  #   # batch_size = 5
  #   # n_epochs = 4
  #   # alpha = 0.0003
  #   # writer = SummaryWriter(f'runs/_ppo_{env_string}_{alpha}_{n_epochs}_{batch_size}')
  #   # agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
  #   #                 alpha=alpha, n_epochs=n_epochs,
  #   #                 input_dims=env.observation_space.shape,writer=writer)
  #   # n_games = 300

  #   # figure_file ='plots/transfer_ppo_300.png'

  #   # best_score = env.reward_range[0]
  #   # score_history = []

  #   # learn_iters = 0
  #   # avg_score = 0
  #   # n_steps = 0

  #   # for i in range(n_games):
  #   #     observation = env.reset()
  #   #     done = False
  #   #     score = 0
  #   #     while not done:
  #   #         action, prob, val = agent.choose_action(observation)
  #   #         observation_, reward, done, info = env.step(action)
  #   #         n_steps += 1
  #   #         score += reward
  #   #         agent.remember(observation, action, prob, val, reward, done)
  #   #         if n_steps % N == 0:
  #   #             agent.learn()
  #   #             learn_iters += 1
  #   #         observation = observation_
  #   #     writer.add_scalar('reward/episode', score, i)
  #   #     score_history.append(score)
  #   #     avg_score = np.mean(score_history[-100:])

  #   #     if avg_score > best_score:
  #   #         best_score = avg_score
  #   #         agent.save_models()

  #   #     print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
  #   #             'time_steps', n_steps, 'learning_steps', learn_iters)
  #   # x = [i+1 for i in range(len(score_history))]
  #   # plot_learning_curve(x, score_history, figure_file)
