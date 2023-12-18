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

  def close(self):
    self.transfer_service.cleanup() # close transfer_service


def main(optimizer):
    log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
    log_file = f"logFileDir/{optimizer}_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
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
    initial_state = env.reset()

    if optimizer == 'GD':
        optimal_actions = gradient_opt(env)
        print("Optimal Actions: ", optimal_actions)
    elif optimizer == 'BO':
        best_params = bayes_optimizer(env)
        print(f"Optimal parameters: {best_params}")
    elif optimizer == 'RL':
        # Assuming you have a function for RL optimizer
        rl_results = rl_optimizer(env)  # Replace with actual RL optimizer function
        print("RL Results: ", rl_results)
    else:  # Default to 'max'
        taken_actions = maximize(env)
        print("Taken Actions: ", taken_actions)

    env.close()

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['gd', 'bo', 'rl', 'max']:
        print("Usage: python script.py <optimizer>")
        print("Optimizer must be one of: 'gd', 'bo', 'rl', 'max'")
        sys.exit(1)

    optimizer = sys.argv[1].upper()
    main(optimizer)
