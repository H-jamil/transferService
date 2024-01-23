from utils import *
from datetime import datetime
from PPO import PPO_discrete
# import gymnasium as gym
import gym
import os, shutil
import argparse
import torch

import numpy as np
import copy
import random
import logging as log

from transferService import *
from optimizer_gd import *
from transferClass import *
import time
from pathlib import Path
import pickle



'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
# parser.add_argument('--optimizer', type=str, default='ppo', help='running  optimer: ppo,sb3,GD,BO,MAX,MIN')
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=4000, help='which model to load')
parser.add_argument('--eval_runs', type=int, default=1, help='how many times to evaluate the policy')

parser.add_argument('--seed', type=int, default=209, help='random seed')
parser.add_argument('--T_horizon', type=int, default=128, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=5e3, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=2e2, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e2, help='Model evaluating interval, in steps.')
parser.add_argument('--max_e_steps', type=int, default=1000, help='Max steps in one episode')
parser.add_argument('--max_episodes', type=int, default=5e2, help='Max episodes in one training')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=10, help='Hidden net width')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=0.0 , help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

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



def get_env(env_string='transferService_10_max_thrt', optimizer='ppo_self_norm_max_min_score_max', REMOTE_IP = "129.114.109.104", REMOTE_PORT = "80", INTERVAL = 1,INTERFACE = "eno1",SERVER_IP = '127.0.0.1',SERVER_PORT = 8080):
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
    if optimizer=='sb3':
      print("sb3 transferService is being created")
      transfer_service = transferService_total_score(REMOTE_IP, REMOTE_PORT, INTERVAL, INTERFACE, SERVER_IP, SERVER_PORT, optimizer, log)
    else:
      print("ppo-GD-BO-MAX-MIN transferService is being created")
      transfer_service = transferService(REMOTE_IP, REMOTE_PORT, INTERVAL, INTERFACE, SERVER_IP, SERVER_PORT, optimizer, log)
    env = transferClass(transfer_service,optimizer)
    return env


def main(optim):
    env_string='transferService_10_step'
    opt.max_e_steps = 1000
    env= get_env(env_string=env_string,optimizer=optim)

    if optim == 'ppo' or optim=='sb3':
      # # Load from disk
      data = np.load('obs_stats.npz')
      obs_min = data['min']
      obs_max = data['max']

      print("obs_min after loading", obs_min)
      print("obs_max after loading", obs_max)

      data = np.load('reward_stats.npz')
      reward_mean = data['mean']
      reward_std = data['std']

      print("reward_mean after loading", reward_mean)
      print("reward_std after loading", reward_std)
      env = NormalizeObservationAndRewardWrapper(env, obs_min, obs_max, reward_scale=1.0)

    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n


    # # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Env:',env_string,'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,'   Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps)
    print('\n')

    # Use tensorboard to record training curves
    if optim == 'ppo':
      print("ppo is running")

      if opt.write:
          from torch.utils.tensorboard import SummaryWriter
          timenow = str(datetime.now())[0:-10]
          timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
          writepath = 'runs/{}'.format(env_string) + timenow
          if os.path.exists(writepath): shutil.rmtree(writepath)
          writer = SummaryWriter(log_dir=writepath)

      if not os.path.exists('model'):
        os.mkdir('model')

      agent = PPO_discrete(**vars(opt))

      if opt.Loadmodel:
        agent.load(opt.ModelIdex)

      if opt.render:
          '''Evaluate'''
          ep_r = evaluate_policy(env, agent, turns=1)
          print(f'Env:{env_string}, Episode Reward:{ep_r}')
          # run_no += 1
          # while True:
          #     ep_r = evaluate_policy(env, agent, turns=1)
          #     print(f'Env:{env_string}, Episode Reward:{ep_r}')
      else:
          traj_lenth, total_steps,total_episodes = 0, 0 , 0
          reward_list = []
          action_list = []
          # while total_episodes < opt.max_episodes:
          while total_steps < opt.Max_train_steps:
              '''Reset Env'''
              s = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
              env_seed += 1
              done = False

              '''Interact & trian'''
              while not done:
                  '''Interact with Env'''
                  a, logprob_a = agent.select_action(s, deterministic=False) # use stochastic when training
                  # s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
                  s_next, r, done, info = env.step(a) # dw: dead&win; tr: truncated
                  # if r <=-100: r = -30  #good for LunarLander
                  # done = (dw or tr)
                  action_list.append(a)
                  reward_list.append(r)
                  '''Store the current transition'''
                  # agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_lenth)
                  agent.put_data(s, a, r, s_next, logprob_a, done, done, idx = traj_lenth)
                  s = s_next
                  traj_lenth += 1
                  total_steps += 1

                  '''Update if its time'''
                  # if traj_lenth >= opt.T_horizon:
                  if traj_lenth % opt.T_horizon == 0:
                      print("traj_lenth", traj_lenth, "total_steps", total_steps, "agent training")
                      agent.train()
                      traj_lenth = 0

                  '''Record & log'''
                  if total_steps % opt.eval_interval == 0:
                      env.close()
                      score = evaluate_policy(env, agent, turns=1) # evaluate the policy for 3 times, and get averaged result
                      if opt.write:
                          writer.add_scalar('ep_r', score, global_step=total_steps)
                      print('EnvName:',env_string,'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)

                  '''Save model'''
                  # if total_steps >= opt.save_interval:
                  if total_steps % opt.save_interval==0:
                      agent.save(total_steps)

          total_episodes += 1
          env.close()

    elif optim == 'sb3':
      from pathlib import Path
      from stable_baselines3 import PPO
      from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
      from stable_baselines3.common.monitor import Monitor
      from stable_baselines3.common.callbacks import BaseCallback, EvalCallback,CheckpointCallback,CallbackList
      import torch as th
      import torch.nn as nn
      from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

      policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[{'pi': [128, 128], 'vf': [128, 128]}])
      model = PPO("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1,tensorboard_log="./ppo_tensorboard/",ent_coef=0.01)
      model = PPO.load("./ppo_best_model/best_model.zip")
      done = False
      episode_reward = 0
      obs = env.reset()
      while not done:
          action, _ = model.predict(obs, deterministic=True)
          print("action: ",action)
          obs, reward, done, info = env.step(action)
          print("obs: ", obs,".... reward: ",reward)
          episode_reward += reward
      env.close()
      print(f"Episode reward: {episode_reward}")

    elif optim == 'GD':
      print("GD is running")
      initial_state = env.reset()
      optimal_actions = gradient_opt(env)
      env.close()
      print("Optimal Actions: ", optimal_actions)

    elif optim == 'BO':
      print("BO is running")
      initial_state = env.reset()
      best_params = bayes_optimizer(env)
      print(f"Optimal parameters: {best_params}")
      env.close()

    elif optim == 'MIN':
      print("MIN is running")
      initial_state = env.reset()
      taken_actions = minimize(env)
      print("Taken Actions: ", taken_actions)
      env.close()

    elif optim == 'MAX':
      print("MAX is running")
      initial_state = env.reset()
      taken_actions = maximize(env)
      print("Taken Actions: ", taken_actions)
      env.close()

if __name__ == '__main__':

  start_signal = multiprocessing.Value('i', False)
  response_signal = multiprocessing.Value('i', False)
  finish_signal = multiprocessing.Value('i', False)
  server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server_socket.bind(('10.140.82.45', 12345))
  server_socket.listen(1)
  process = multiprocessing.Process(target=handle_client, args=(server_socket, start_signal, response_signal,finish_signal))
  process.start()
  optimizers = ['ppo','sb3','MAX','MIN','GD','BO']
  # optimizers = ['ppo','sb3']

  for i in range(0,20):
    for optimizer in optimizers:
      print(f"Iteration: {i}, {optimizer} running")
      print("Server is running and sending signal to the client to start iperf3")
      start_signal.value = True
      print("Waiting for client to complete the task")
      main(optimizer)
      while not response_signal.value:
          pass
      print("Client completed the task")
      time.sleep(2)
      response_signal.value = False

  finish_signal.value = True
  time.sleep(5)
  process.terminate()
  process.join()
