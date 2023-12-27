import gym
import sys
from gym import spaces
import numpy as np
import copy
import random
from datetime import datetime
import logging as log
import time
from transferService import transferService
from optimizer_gd import *
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback,CheckpointCallback,CallbackList


class transferClass(gym.Env):
  metadata = {"render_modes": ["human"], "render_fps": 30}
  def __init__(self,transferServiceObject):
    super().__init__()
    self.action_array=[(1,1),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8)]
    self.transfer_service = transferServiceObject
    self.action_space =spaces.Discrete(9) # example action space
    self.observation_space = spaces.Box(low=0, high=np.inf, shape=(40,), dtype=np.float32) # example observation space
    self.current_observation = np.zeros(40,) # initialize current observation

  def reset(self):
    self.current_observation = self.transfer_service.reset() # get initial observation
    return self.current_observation

  def step(self, action):
    # perform action using transfer_service
    print("Env step action ",action, type(action))
    new_observation,reward=self.transfer_service.step(self.action_array[action][0],self.action_array[action][1])
    if reward==1000000:
      done=True
    else:
      done=False
    self.current_observation=new_observation
    print("Env step observations ",new_observation)
    return new_observation, reward, done, {}

  def bayes_step(self,action):
    params = [1 if x<1 else int(np.round(x)) for x in action]
    print("Bayes Step: ",params)
    if params[0] > 8:
      params[0] = 8
    obs,score_b,done_b,__=self.step(params[0])
    print("Bayes Step Observations: ",obs)
    return score_b

  def render(self, mode="human"):
    pass

  def close(self):
    self.transfer_service.cleanup() # close transfer_service


def main(optimizer):
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
    env = transferClass(transfer_service)

    if optimizer == 'GD':
        initial_state = env.reset()
        optimal_actions = gradient_opt(env)
        print("Optimal Actions: ", optimal_actions)

    elif optimizer == 'BO':
        initial_state = env.reset()
        best_params = bayes_optimizer(env)
        print(f"Optimal parameters: {best_params}")

    elif optimizer == 'RL':
        # env = Monitor(env)  # Wrap the environment with Monitor for training statistics
        env = DummyVecEnv([lambda: env])  # Vectorized environments allow for parallelism
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        obs = env.reset()
        model = PPO.load("./ppo_best_model/best_model.zip")
        done = False
        episode_reward = 0
        while not done:
          action, _ = model.predict(obs, deterministic=True)
          obs, reward, done, info = env.step(action)
          print("obs: ", obs,".... reward: ",reward)
          episode_reward += reward[0]
        print(f"Episode reward: {episode_reward}")

    elif optimizer == 'MIN':
        initial_state = env.reset()
        taken_actions = minimize(env)
        print("Taken Actions: ", taken_actions)

    else:  # Default to 'max'
        initial_state = env.reset()
        taken_actions = maximize(env)
        print("Taken Actions: ", taken_actions)

    env.close()

if __name__ == "__main__":
    # if len(sys.argv) != 2 or sys.argv[1] not in ['gd', 'bo', 'rl', 'max', 'min']:
    #     print("Usage: python script.py <optimizer>")
    #     print("Optimizer must be one of: 'gd', 'bo', 'rl', 'max', 'min'")
    #     sys.exit(1)

    # optimizer = sys.argv[1].upper()
    optimizers = ['GD', 'BO', 'RL', 'MAX', 'MIN']
    for i in range(10):
        for optimizer in optimizers:
            print(f"Iteration: {i}, {optimizer} running")
            main(optimizer)
            time.sleep(2)
