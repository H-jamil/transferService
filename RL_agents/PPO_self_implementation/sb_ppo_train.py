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
import time
import os
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


with open('dataframes.pickle', 'rb') as handle:
    loaded_dfs = pickle.load(handle)
with open('initial_dataframes.pickle', 'rb') as handle:
    loaded_initial_dfs = pickle.load(handle)

env=transferClass(loaded_dfs,loaded_initial_dfs,'random')
evaluation_env=transferClass(loaded_dfs,loaded_initial_dfs,'random')
tensorboard_log_dir = "./ppo_tensorboard/PPO_1/"
model_log_dir = os.path.join(tensorboard_log_dir, "model")
energy_log_dir = os.path.join(tensorboard_log_dir, "energy")

policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[{'pi': [128, 128], 'vf': [128, 128]}])
model = PPO("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1,tensorboard_log=model_log_dir,ent_coef=0.01)
eval_callback = EvalCallback(evaluation_env, best_model_save_path='./ppo/ppo_best_model/',
                               log_path='./ppo/ppo_logs/', eval_freq=1000,
                               deterministic=True, render=False)
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./ppo/ppo_checkpoints/',
                                           name_prefix='ppo_model')
energy_callback = EnergyMonitorCallback(log_dir=energy_log_dir)

# Combine both callbacks
callback = CallbackList([checkpoint_callback, eval_callback, energy_callback])
model.learn(total_timesteps=1000000, callback=callback)
