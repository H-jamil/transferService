import gym
from gym import spaces
import numpy as np
import copy
import random
from datetime import datetime
import logging as log
import time
from transferService import transferService
from optimizer_gd import *

class transferClass(gym.Env):
  def __init__(self,transferServiceObject):
    self.action_array=[(1,1),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8)]
    self.transfer_service = transferServiceObject
    self.action_space =spaces.Discrete(9) # example action space
    self.observation_space = spaces.Box(low=0, high=np.inf, shape=(35,), dtype=np.float32) # example observation space
    self.current_observation = np.zeros(35,) # initialize current observation

  def reset(self):
    self.current_observation = self.transfer_service.reset() # get initial observation
    return self.current_observation

  def step(self, action):
    # perform action using transfer_service
    new_observation,reward=self.transfer_service.step(self.action_array[action][0],self.action_array[action][1])
    if reward==1000000:
      done=True
    else:
      done=False
    return new_observation, reward, done, {}

  def bayes_step(self,action):
    params = [1 if x<1 else int(np.round(x)) for x in action]
    print("Bayes Step: ",params)
    if params[0] > 8:
      params[0] = 8
    _,score_b,done_b,__=self.step(params[0])
    return score_b

  # def bayes_step(self, action):
  #   # If action is a list or a numpy array, we want the first value.
  #   # For multi-dimensional action spaces, you might need to handle more elements.
  #   action_val = action[0] if isinstance(action, (list, np.ndarray)) else action

  #   # Round the action value to get an integer action index.
  #   action_idx = int(round(action_val))

  #   # Ensure that the action index doesn't exceed the bounds of the action space.
  #   action_idx = max(0, min(action_idx, len(self.action_array) - 1))

  #   # Get the action from the action array using the index.
  #   chosen_action = self.action_array[action_idx]

  #   # Perform the chosen action using the transfer_service and get the new observation and reward.
  #   new_observation, reward, done, _ = self.step(action_idx)

  #   # In the context of Bayesian optimization, we usually maximize the objective.
  #   # If your reward should be minimized (e.g., an error that should be minimized), negate the reward.
  #   # Otherwise, you can just return the reward.
  #   return reward

  def close(self):
    self.transfer_service.cleanup() # close transfer_service

if __name__ == "__main__":
  log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
  log_file = "logFileDir/" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
  log.basicConfig(
      format=log_FORMAT,
      datefmt='%m/%d/%Y %I:%M:%S %p',
      level=log.INFO,
      handlers=[
          log.FileHandler(log_file),
          log.StreamHandler()
      ]
  )
  REMOTE_IP = "128.205.222.176"
  REMOTE_PORT = "80"
  INTERVAL = 1
  INTERFACE="enp3s0"
  SERVER_IP = '127.0.0.1'
  SERVER_PORT = 8080
  OPTIMIZER="GD"
  transfer_service=transferService(REMOTE_IP,REMOTE_PORT,INTERVAL,INTERFACE,SERVER_IP,SERVER_PORT,OPTIMIZER,log)
  env=transferClass(transfer_service)
  done=False
  initial_state = env.reset()
  # optimal_actions = gradient_opt(env)
  # print("Optimal Actions: ", optimal_actions)
  best_params = bayes_optimizer(env)
  print(f"Optimal parameters: {best_params}")
  env.close()
  # for i in range(5):
  #   print("Trial: ",i)
  #   env=transferClass(transfer_service)
  #   done=False
  #   initial_state = env.reset()
  #   best_params = bayes_optimizer(env)
  #   print(f"Optimal parameters: {best_params}")
  #   # optimal_actions = gradient_opt(env)
  #   # print("Optimal Actions: ", optimal_actions)
  #   env.close()
