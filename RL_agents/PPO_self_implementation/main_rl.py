from utils import *
import sys
import copy
import random
import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback,CheckpointCallback,CallbackList
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from optimizer_gd import *
import socket
import multiprocessing
from time import sleep


def handle_client(server_socket, start_signal, response_signal,finish_signal):
    print("Server process is waiting for connection")
    connection, _ = server_socket.accept()
    while True:
      if start_signal.value:
          time.sleep(10)
          connection.sendall(b'start')
          start_signal.value = False
          if connection.recv(1024).decode() == 'done':
              response_signal.value = True
              start_signal.value = False
      if finish_signal.value:
          print("Server process is closing the connection from finish signal")
          connection.sendall(b'finish')
          connection.close()
          break



def main (optim):
  if optim == 'PPO':
    print("ppo is running")
    model = PPO.load("./ppo_1M_simulator/ppo_best_model/best_model.zip")
    env= get_env(env_string='transferService',optimizer=optim)
    env = NormalizeObservationAndRewardWrapper(env)
    done = False
    episode_reward = 0
    action_list=[]
    reward_list=[]
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_list.append(int(action))
        # print("action: ",action)
        obs, reward, done, info = env.step(action)
        # print("obs: ", obs,".... reward: ",reward)
        reward_list.append(reward)
        episode_reward += reward
    env.close()
    print(f"Episode reward: {episode_reward}")
    print(f"actions {action_list},   {len(action_list)}")
    print(f"rewards {reward_list},  {len(reward_list)}")

  elif optim == 'GD':
      print("GD is running")
      env= get_env(env_string='transferService',optimizer=optim)
      initial_state = env.reset()
      optimal_actions = gradient_opt(env)
      env.close()
      print("Optimal Actions: ", optimal_actions)

  elif optim == 'BO':
    print("BO is running")
    env= get_env(env_string='transferService',optimizer=optim)
    initial_state = env.reset()
    best_params = bayes_optimizer(env)
    print(f"Optimal parameters: {best_params}")
    env.close()

  elif optim == 'STATIC':
    print("static is running")
    env= get_env(env_string='transferService',optimizer=optim)
    initial_state = env.reset()
    optimal_actions = static(env,7)
    env.close()
    print("static actions: ", optimal_actions)

if __name__ == '__main__':
  # start_signal = multiprocessing.Value('i', False)
  # response_signal = multiprocessing.Value('i', False)
  # finish_signal = multiprocessing.Value('i', False)
  # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  # server_socket.bind(('10.52.0.239', 12345))
  # server_socket.listen(1)
  # process = multiprocessing.Process(target=handle_client, args=(server_socket, start_signal, response_signal,finish_signal))
  # process.start()
  # # optimizers = ['PPO','GD','BO','STATIC']
  optimizers = ['GD']
  for i in range(0,1):
    for optimizer in optimizers:
      print(f"Iteration: {i}, {optimizer} running")
      # print("Server is running and sending signal to the client to start iperf3")
      # start_signal.value = True
      print("Waiting for client to complete the task")
      main(optimizer)
      # time.sleep(2)
      # while not response_signal.value:
      #     pass
      # print("Client completed the task")
      # response_signal.value = False

  # finish_signal.value = True
  # time.sleep(5)
  # process.terminate()
  # process.join()
