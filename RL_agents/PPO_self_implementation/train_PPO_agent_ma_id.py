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
from torch.utils.tensorboard import SummaryWriter

class EnergyMonitorCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(EnergyMonitorCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.last_time_checked = time.time()
        self.energy_old = read_energy()

    def _on_step(self) -> bool:
        current_time = time.time()
        # Check if at least one second has passed
        if current_time - self.last_time_checked >= 1.0:
            energy_now = read_energy()
            if energy_now is not None:
                # Calculate the energy used in the last period
                energy_used = int((energy_now - self.energy_old) / 1000000)
                # Log the energy used value to TensorBoard
                self.writer.add_scalar('energy/usage', energy_used, self.num_timesteps)
                # Update the old energy value and the last time checked
                self.energy_old = energy_now
                self.last_time_checked = current_time
        return True



def main(option):
  if option == "test":
    print("test is running")
    env = get_env()
    total_scores=0
    s = env.reset()
    action_list=[]
    reward_list=[]
    done = False
    while not done:
        a=env.action_space.sample()
        s_next, r, done, info = env.step(a)
        action_list.append(a)
        reward_list.append(r)
        total_scores += r
        s = s_next
    env.close()
    print(f"Total Reward: {total_scores}")
    print(f"actions {action_list},   {len(action_list)}")
    print(f"rewards {reward_list},  {len(reward_list)}")

  elif option == "train":
    print("train is running")
    env = get_env()
    policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[{'pi': [128, 128], 'vf': [128, 128]}])
    string_='_MA_ID_Train'
    tensorboard_log_dir = f"./ppo_tensorboard_{string_}/"
    model_log_dir = os.path.join(tensorboard_log_dir, "model")
    energy_log_dir = os.path.join(tensorboard_log_dir, "energy")
    model = PPO("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1,tensorboard_log=model_log_dir,ent_coef=0.01)
    eval_callback = EvalCallback(env, best_model_save_path=f'./ppo_{string_}/ppo_best_model/',
                                  log_path=f'./ppo_{string_}/ppo_logs/', eval_freq=10,
                                  deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=10, save_path=f'./ppo_{string_}/ppo_checkpoints/',
                                              name_prefix='ppo_model')
    energy_callback = EnergyMonitorCallback(log_dir=energy_log_dir)
    # Combine both callbacks
    callback = CallbackList([checkpoint_callback, eval_callback, energy_callback])
    model.learn(total_timesteps=100, callback=callback)

  elif option == "generate":
    print("generate log is running")
    sorted_tuples_list=[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8)]
    # sorted_tuples_list=[(8, 4), (8, 5), (8, 6)]
    env = get_env_DG(optimizer='logGenerator')
    s=env.reset()
    count=0
    start_time = time.time()
    for tuple1 in range (31,len(sorted_tuples_list)):
      for tuple2 in range (len(sorted_tuples_list)):
        if tuple1==tuple2:
          continue
        else:
            count+=1
            #
            s=env.reset()
            # initial_action=(1,1)
            # print("Taking action ", initial_action)
            # s_next, r, done, info=env.step(np.array(initial_action))
            print("Taking action ", sorted_tuples_list[tuple1])
            s_next, r, done, info=env.step(np.array(sorted_tuples_list[tuple1]))
            print("Taking action ", sorted_tuples_list[tuple2])
            s_next, r, done, info=env.step(np.array(sorted_tuples_list[tuple2]))
            env.close()
            time.sleep(3)
            print("Current count number: ",count)

    print("Total time in seconds: ",time.time()-start_time)
    print("Total count: ",count)
if __name__ == "__main__":
  main(sys.argv[1])
