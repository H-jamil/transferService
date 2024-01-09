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

from transferService import transferService
from optimizer_gd import *
from transferClass import *
import time
from pathlib import Path
import pickle



'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=300000, help='which model to load')

parser.add_argument('--seed', type=int, default=209, help='random seed')
parser.add_argument('--T_horizon', type=int, default=256, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=10e3, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e3, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=1e3, help='Model evaluating interval, in steps.')
parser.add_argument('--max_e_steps', type=int, default=1000, help='Max steps in one episode')
parsere.add_argument('--max_episodes', type=int, default=1000, help='Max episodes in one training')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=128, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

# [-0.0214955   0.00838333  0.01865386 -0.0489112 ] <class 'numpy.ndarray'>

def main():
    BriefEnvName = ['CP-v1-normalized','LLd-v2']
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
    opt.max_e_steps = 1000
    # obs_mean, obs_std,reward_mean,reward_std = compute_observation_stats(env)
    # # env.close()
    # # # Save to disk
    # with open('obs_stats.pkl', 'wb') as f:
    #   pickle.dump({'mean': obs_mean, 'std': obs_std}, f)
    # np.savez('obs_stats.npz', mean=obs_mean, std=obs_std)
    # np.savez('reward_stats.npz', mean=reward_mean, std=reward_std)
    # # Load from disk
    data = np.load('obs_stats.npz')
    obs_mean = data['mean']
    obs_std = data['std']

    print("obs_mean after loading", obs_mean)
    print("obs_std after loading", obs_std)

    data = np.load('reward_stats.npz')
    reward_mean = data['mean']
    reward_std = data['std']

    print("reward_mean after loading", reward_mean)
    print("reward_std after loading", reward_std)

    env = NormalizeObservationAndRewardWrapper(env, obs_mean, obs_std, reward_scale=1.0)
    # # env = Monitor(env)  # Wrap with Monitor for advanced logging
    # # env = DummyVecEnv([lambda: env])  # Optional: Use a vectorized environment
    # # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # eval_env = gym.make(EnvName[opt.EnvIdex],render_mode = "human" if opt.render else None)
    # eval_env = NormalizeObservationAndRewardWrapper(eval_env, obs_mean, obs_std, reward_scale=1.0)
    # # eval_env = Monitor(eval_env)  # Wrap with Monitor for advanced logging
    # # eval_env = DummyVecEnv([lambda: eval_env])  # Optional: Use a vectorized environment
    # # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
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
        while True:
            ep_r = evaluate_policy(env, agent, turns=1)
            print(f'Env:{env_string}, Episode Reward:{ep_r}')
    else:
        traj_lenth, total_steps,total_episodes = 0, 0 , 0
        while total_steps < opt.Max_train_steps:
            '''Reset Env'''
            # s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            s = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
            # print("From main", s, type(s))
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

                '''Store the current transition'''
                # agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_lenth)
                agent.put_data(s, a, r, s_next, logprob_a, done, done, idx = traj_lenth)
                s = s_next
                traj_lenth += 1
                total_steps += 1
                '''Update if its time'''
                if traj_lenth % opt.T_horizon == 0:
                    agent.train()
                    traj_lenth = 0

                '''Record & log'''
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(env, agent, turns=1) # evaluate the policy for 3 times, and get averaged result
                    if opt.write: writer.add_scalar('ep_r', score, global_step=total_steps)
                    print('EnvName:',env_string,'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)

                '''Save model'''
                if total_steps % opt.save_interval==0:
                    agent.save(total_steps)

        env.close()
        # eval_env.close()

if __name__ == '__main__':
    main()
