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

# class NormalizeObservationAndRewardWrapper(gym.Wrapper):
#     def __init__(self, env, obs_mean, obs_std, reward_scale=1.0):
#         super().__init__(env)
#         self.obs_mean = obs_mean
#         self.obs_std = obs_std
#         self.reward_scale = reward_scale
#         self.old_score=0
#         self.score_difference_positive_threadhold=2
#         self.score_difference_negative_threadhold=-2

#     def reset(self, **kwargs):
#         observation = self.env.reset(**kwargs)
#         return observation

#     def step(self, action):
#         # observation, reward, done, _, info = self.env.step(action)
#         observation, reward, done, info = self.env.step(action)
#         normalized_obs = self.normalize_observation(observation)
#         if reward != 1000000:
#         #   normalized_reward = reward * self.reward_scale
#             score_difference = reward - self.old_score
#             self.old_score = reward
#             if score_difference > self.score_difference_positive_threadhold:
#               normalized_reward = 1
#             elif score_difference < self.score_difference_negative_threadhold:
#               normalized_reward = -3
#             else:
#               normalized_reward = 0
#         else:
#           normalized_reward = 1
#         # return normalized_obs, normalized_reward, done, _ ,info
#         return normalized_obs, normalized_reward, done, info

#     def normalize_observation(self, observation):
#       # try:
#       #   return (observation - self.obs_mean) / self.obs_std
#       # except:
#       #   return np.zeros(40,)
#       EPSILON = 1e-10  # Small constant to prevent division by zero
#       normalized_observation = (observation - self.obs_mean) / (self.obs_std + EPSILON)
#       return normalized_observation
class NormalizeObservationAndRewardWrapper(gym.Wrapper):
    def __init__(self, env, obs_min, obs_max, reward_scale=1.0):
        super().__init__(env)
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.reward_scale = reward_scale
        self.old_score = 0
        self.score_difference_positive_threshold = 5
        self.score_difference_negative_threshold = -1

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.normalize_observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        normalized_obs = self.normalize_observation(observation)

        if reward != 1000000:
            score_difference = reward - self.old_score
            self.old_score = reward
            if score_difference > self.score_difference_positive_threshold:
                # normalized_reward = 1
                normalized_reward = score_difference
            elif score_difference < self.score_difference_negative_threshold:
                # normalized_reward = -1
                normalized_reward = score_difference
            else:
                normalized_reward = 0
        else:
            normalized_reward = 1

        return normalized_obs, normalized_reward, done, info

    def normalize_observation(self, observation):
        EPSILON = 1e-10  # Small constant to prevent division by zero
        normalized_observation = (observation - self.obs_min) / (self.obs_max - self.obs_min + EPSILON)
        return normalized_observation


# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from transferService import transferService
from optimizer_gd import *
from transferClass import *
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def main():
    env_string = 'transferService'
    optimizer='ppo_sb3_max_min'
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
            # log.StreamHandler()
        ]
    )
    REMOTE_IP = "129.114.109.46"
    REMOTE_PORT = "80"
    INTERVAL = 1
    INTERFACE = "eno1"
    SERVER_IP = '127.0.0.1'
    SERVER_PORT = 8080
    transfer_service = transferService(REMOTE_IP, REMOTE_PORT, INTERVAL, INTERFACE, SERVER_IP, SERVER_PORT, optimizer, log)
    env = transferClass(transfer_service,optimizer)

    data = np.load('obs_stats.npz')
    obs_mean = data['mean']
    obs_std = data['std']

    print("obs_mean after loading", obs_mean)
    print("obs_std after loading", obs_std)

    env = NormalizeObservationAndRewardWrapper(env, obs_mean, obs_std, reward_scale=1.0)

    policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[{'pi': [128, 128], 'vf': [128, 128]}])
    # model = PPO("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1,tensorboard_log="./ppo_tensorboard/",ent_coef=0.01)


    # model = PPO.load("./ppo_checkpoints/ppo_model_4000_steps.zip")
    model = PPO.load("./ppo_best_model/best_model.zip")
    done = False
    episode_reward = 0
    obs = env.reset()


    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        print("action: ",action)
        # obs, reward, done, info = env.step([action])
        obs, reward, done, info = env.step(action)
        print("obs: ", obs,".... reward: ",reward)
        # episode_reward += reward[0]
        episode_reward += reward

    env.close()
    print(f"Episode reward: {episode_reward}")

if __name__ == '__main__':
  # run_num = 0
  # while run_num < 5:
  #   main()
  #   run_num += 1


    env_string = 'transferService'
    optimizer='ppo_sb3_max_min'
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
            # log.StreamHandler()
        ]
    )
    REMOTE_IP = "129.114.109.46"
    REMOTE_PORT = "80"
    INTERVAL = 1
    INTERFACE = "eno1"
    SERVER_IP = '127.0.0.1'
    SERVER_PORT = 8080
    transfer_service = transferService(REMOTE_IP, REMOTE_PORT, INTERVAL, INTERFACE, SERVER_IP, SERVER_PORT, optimizer, log)
    env = transferClass(transfer_service,optimizer)

    data = np.load('obs_stats.npz')
    obs_min = data['min']
    obs_max = data['max']

    print("obs_min after loading", obs_min)
    print("obs_max after loading", obs_max)

    env = NormalizeObservationAndRewardWrapper(env, obs_min, obs_max, reward_scale=1.0)

    policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[{'pi': [128, 128], 'vf': [128, 128]}])
    model = PPO("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1,tensorboard_log="./ppo_tensorboard/",ent_coef=0.01)


  #   # model = PPO.load("./ppo_checkpoints/ppo_model_4000_steps.zip")
  #   model = PPO.load("./ppo_best_model/best_model.zip")
  #   done = False
  #   episode_reward = 0
  #   obs = env.reset()


  #   while not done:
  #       action, _ = model.predict(obs, deterministic=True)
  #       # action = env.action_space.sample()
  #       print("action: ",action)
  #       # obs, reward, done, info = env.step([action])
  #       obs, reward, done, info = env.step(action)
  #       print("obs: ", obs,".... reward: ",reward)
  #       # episode_reward += reward[0]
  #       episode_reward += reward

  #   env.close()
  #   print(f"Episode reward: {episode_reward}")




    eval_callback = EvalCallback(env, best_model_save_path='./ppo_best_model/',
                               log_path='./ppo_logs/', eval_freq=100,
                               deterministic=True, render=False)

  # Callback for saving checkpoints every 1000 timesteps
    checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./ppo_checkpoints/',
                                           name_prefix='ppo_model')

  # Combine both callbacks
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(total_timesteps=10000, callback=callback)
