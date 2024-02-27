import os
import re
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
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
import gym
from gym import spaces
import numpy as np

def process_log_file(full_path):
    if os.stat(full_path).st_size == 0:
        print(f"Skipping empty file: {full_path}")
        return pd.DataFrame()

    with open(full_path, 'r') as file:
        data = []
        for line in file:
            match = re.search(r'(\d+\.\d+).*Throughput @(\d+\.\d+)s:\s+(\d+\.\d+)Gbps, lossRate: (\d+\.\d+|\d+)\s+parallelism:(\d+)\s+concurrency:(\d+)\s+score:(-?\d+\.\d+)\s+rtt:(\d+\.\d+) ms energy:(\d+\.\d+) Jules s-plr:([\deE.-]+)', line)
            if match:
                time = datetime.fromtimestamp(float(match.group(1)))
                throughput = float(match.group(3))
                loss_rate = float(match.group(4))
                parallelism = int(match.group(5))
                concurrency = int(match.group(6))
                score = float(match.group(7))
                rtt = float(match.group(8))
                energy = float(match.group(9))
                sender_lr = float(match.group(10))

                data.append([time, throughput, loss_rate, parallelism, concurrency, score, rtt, energy, sender_lr])

        if data:
            df = pd.DataFrame(data, columns=['Time', 'Throughput', 'LossRate', 'Parallelism', 'Concurrency', 'Score', 'RTT', 'Energy', 'SenderLR'])
        else:
            df = pd.DataFrame()
            print(f"No valid data in file: {full_path}")

    return df

def process_directory(directory_path):
    all_dfs = []  # List to hold all DataFrames from each log file

    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            # Construct the full path to the file
            full_path = os.path.join(dirpath, filename)
            # Check if the file has a log extension (assuming you're processing .log files)
            if filename.endswith(".log"):  # Modify the extension as needed
                print(f"Processing file: {full_path}")
                df = process_log_file(full_path)
                if not df.empty:
                    all_dfs.append(df)
                else:
                    print(f"Empty or invalid data in file: {filename}")

    # Concatenate all DataFrames if not empty
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data was found

def print_empty_dataframes(dfs):
    total_rows=0
    for key, df in dfs.items():
        if df.empty:
            print(f"The DataFrame for key '{key}' is empty.")
        else:
            num_rows = len(df)
            total_rows+=num_rows
            print(f"The DataFrame for key '{key}' has {num_rows} rows")
    print(f"\n\n\nThere are toatl {total_rows} rows in all the transitions dictionaries")



def find_transitions_tuple_based(df, col1, col2, old_tuple, target_tuple):
    # Create a new empty DataFrame with the same columns
    new_df = pd.DataFrame(columns=df.columns)
    # Flag to mark if we are currently in a transition block
    in_transition = False
    # Iterate through the DataFrame
    lst=[]
    for index, row in df.iterrows():
        # Check for transition start
        if row[col1] == old_tuple[0] and row[col2] == old_tuple[1] and not in_transition:
            in_transition = True
        # Check for transition end
        elif row[col1] == target_tuple[0] and row[col2] == target_tuple[1] and in_transition:
            new_df = pd.concat([new_df,pd.DataFrame([row])],ignore_index=True)
            # print(type(new_df))
            # lst.append(row)
            # new_df = new_df.append(row, ignore_index=True)
        # If the value is neither start nor target, reset the flag
        elif row[col1] != old_tuple[0] or row[col2] != old_tuple[1]:
            in_transition = False

    # new_df= pd.concat(lst)
    return new_df.reset_index(drop=True)



def filter_df_by_value_tuple(df_original, column_name_1, column_name_2, value_tuple):
    """
    Filters the DataFrame based on specific values in two given columns and creates an independent copy.

    :param df_original: The original DataFrame.
    :param column_name_1: The name of the first column to filter on.
    :param column_name_2: The name of the second column to filter on.
    :param value_tuple: A tuple containing the values to filter by in the specified columns.
    :return: A new DataFrame containing rows where each column matches its corresponding value in value_tuple.
    """
    filtered_df = df_original[(df_original[column_name_1] == value_tuple[0]) & (df_original[column_name_2] == value_tuple[1])].copy()
    return filtered_df



def sample_row_and_neighbors(df, column_name):
    """
    Samples a row based on the distribution of a specific column and returns it with its 4 neighboring rows.
    If fewer than 5 rows are returned, the function duplicates rows to ensure 5 rows are returned.

    :param df: The DataFrame to sample from.
    :param column_name: The column whose distribution to use for sampling.
    :return: A DataFrame containing the sampled row and its 4 neighbors, ensuring 5 rows in total.
    """
    # Ensure the DataFrame's index is reset
    df = df.reset_index(drop=True)
    # Calculate the frequency distribution of the column
    probabilities = df[column_name].value_counts(normalize=True)
    # Map these probabilities back to the DataFrame's index
    probabilities = df[column_name].map(probabilities)
    # Sample one row using these probabilities
    sampled_row = df.sample(n=1, weights=probabilities)
    sampled_index = sampled_row.index[0]

    # Ensure the sampled index is within the DataFrame's range
    if sampled_index >= len(df):
        raise ValueError("Sampled index is out of DataFrame's range.")

    # Attempt to select 2 neighbors from each side
    start_index = max(sampled_index - 2, 0)
    end_index = min(sampled_index + 2, len(df) - 1)
    selected_rows = df.iloc[start_index:end_index + 1]

    # If fewer than 5 rows are selected, duplicate rows to reach 5
    while len(selected_rows) < 5:
        if start_index > 0:
            start_index -= 1
        elif end_index < len(df) - 1:
            end_index += 1
        else:
            # If start and end indices cover the whole DataFrame, duplicate the last row
            selected_rows = pd.concat([selected_rows, selected_rows.iloc[[-1]]], ignore_index=True)
        # Update selected_rows if indices were adjusted
        if len(selected_rows) < 5:
            selected_rows = df.iloc[start_index:end_index + 1]

    return selected_rows


def normalize_and_flatten(df, min_values, max_values):
    # Drop the specified columns
    df = df.drop(columns=['Time'])
    score_array = df['Score'].values
    energy_array=df['Energy'].values
    throughput_array=df['Throughput'].values
    # Normalize each column
    normalized_df = (df - min_values) / (max_values - min_values)

    # Flatten the DataFrame to a single NumPy array
    flattened_array = normalized_df.values.flatten()

    return flattened_array,score_array,energy_array,throughput_array


def get_initial_cluster_dictionary():
    with open('initial_dataframes_multi_action.pickle', 'rb') as handle:
        initial_value_cluster_dictionary = pickle.load(handle)
        return initial_value_cluster_dictionary

def get_cluster_transaction_dictionary():
    with open('dataframes_multi_action.pickle', 'rb') as handle:
        cluster_dictionary = pickle.load(handle)
        return cluster_dictionary


def get_initial_cluster_dictionary_retraining():
    with open('initial_dataframes_multi_action_retraining.pickle', 'rb') as handle:
        initial_value_cluster_dictionary = pickle.load(handle)
        return initial_value_cluster_dictionary

def get_cluster_transaction_dictionary_retraining():
    with open('dataframes_multi_action_retraining.pickle', 'rb') as handle:
        cluster_dictionary = pickle.load(handle)
        return cluster_dictionary


# min_values : [0.0 0.0 1 1 -10.0 0.0 0.0 0.0]
# max_values : [17.6 0.0 8 8 7.0 52.7 120.0 0.9705882352941176]
# min_values=[0.32, 0.0, 1, 1, -3081.0, 0.0, 44.0, 0.0],max_values = [19.2, 2.0, 8, 8, 16.0, 70.1, 120.0, 74.166]

class transferClass_multi_action_increase_decrease_score_SLA(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,transaction_dfs,initial_dfs,backup_initial_dfs,optimizer,total_steps=20,min_values=[0.00, 0.0, 1, 1, -3081.0, 0.0, 0.0, 0.0],max_values = [19.2, 2.0, 8, 8, 16.0, 70.1, 120.0, 74.166],total_file_size=256):
        super().__init__()
        self.min_action=1
        self.max_action=8
        self.action_i_d_array=[1,4,0,-1,-4]

        self.transaction_dfs = transaction_dfs
        self.initial_dfs= initial_dfs
        self.backup_initial_dfs=backup_initial_dfs

        self.action_space = spaces.MultiDiscrete([5, 5])  # example action space
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(40,), dtype=np.float32) # example observation space
        self.current_observation = np.zeros(40,) # initialize current observation

        self.optimizer=optimizer
        self.old_action=None
        self.step_number=0
        self.total_steps=total_steps
        self.sampling_metric='Score'
        self.min_values=np.array(min_values)
        self.max_values=np.array(max_values)
        self.previous_reward=0
        self.obs_df=[]
        self.total_file_size= total_file_size
        self.current_download_size=0

    def reset(self):
        self.current_observation = np.zeros(40,) # initialize current observation
        self.old_action=None
        self.step_number=0
        self.previous_reward=0
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
        self.obs_df=[]
        self.current_download_size=0
        return self.current_observation

    def step(self, action):
        # perform action using transfer_service
        action_1,action_2=action

        self.current_action_parallelism_value+=self.action_i_d_array[action_1]
        if self.current_action_parallelism_value<self.min_action:
            self.current_action_parallelism_value=self.min_action
        elif self.current_action_parallelism_value>self.max_action:
            self.current_action_parallelism_value=self.max_action

        self.current_action_concurrency_value+=self.action_i_d_array[action_2]
        if self.current_action_concurrency_value<self.min_action:
            self.current_action_concurrency_value=self.min_action
        elif self.current_action_concurrency_value>self.max_action:
            self.current_action_concurrency_value=self.max_action


        action_t=(self.current_action_parallelism_value,self.current_action_concurrency_value)

        if self.old_action==None:
            done=False
            try:
                work_df=self.initial_dfs[action_t]
                if work_df.empty:
                    observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            reward_=np.sum(result_array)
            self.old_action=action_t
            reward=reward_-self.previous_reward
            self.previous_reward=reward_

        elif self.old_action==action_t:
            done=False
            try:
                work_df=self.initial_dfs[action_t]
                if work_df.empty:
                    observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            reward_=np.sum(result_array)
            self.old_action=action_t
            reward=reward_-self.previous_reward
            self.previous_reward=reward_

        else:
            done=False
            key_name=(self.old_action, action_t)
            try:
                work_df=self.transaction_dfs[key_name]
                if work_df.empty:
                    try:
                        work_df_=self.initial_dfs[action_t]
                        if work_df_.empty:
                            observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                        else:
                            observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
                    except:
                        observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.transaction_dfs[key_name],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            reward_=np.sum(result_array)
            self.old_action=action_t
            reward=reward_-self.previous_reward
            self.previous_reward=reward_

        self.step_number+=1

        if self.current_download_size>=self.total_file_size:
            done=True
        observation = np.asarray(observation).astype(np.float32)
        self.current_observation=observation
        return self.current_observation, reward, done, {}

    def bayes_step(self,action):
        params = [1 if x<1 else int(np.round(x)) for x in action]
        self.current_action_array_index+=self.action_i_d_array[params[0]]
        if self.current_action_array_index<self.min_action:
            self.current_action_array_index=self.min_action
        elif self.current_action_array_index>self.max_action:
            self.current_action_array_index=self.max_action
        action_t=self.action_array[self.current_action_array_index]

        print("Bayes Step: ",action_t)
        obs,score_b,done_b,__=self.step(action_t)
        print("Bayes Step Score: ", score_b)
        return np.round(score_b * (-1))

    def render(self, mode="human"):
        pass

    def close(self):
        self.reset()

class transferClass_multi_action_increase_decrease_energy_SLA(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,transaction_dfs,initial_dfs,backup_initial_dfs,optimizer,total_steps=20,min_values=[0.00, 0.0, 1, 1, -3081.0, 0.0, 0.0, 0.0],max_values = [19.2, 2.0, 8, 8, 16.0, 70.1, 120.0, 74.166],total_file_size=256,en_sla=75):
        super().__init__()
        self.min_action=1
        self.max_action=8
        self.action_i_d_array=[1,4,0,-1,-4]

        self.transaction_dfs = transaction_dfs
        self.initial_dfs= initial_dfs
        self.backup_initial_dfs=backup_initial_dfs

        self.action_space = spaces.MultiDiscrete([5, 5])  # example action space
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(40,), dtype=np.float32) # example observation space
        self.current_observation = np.zeros(40,) # initialize current observation

        self.energy_sla=en_sla
        self.optimizer=optimizer
        self.old_action=None
        self.step_number=0
        self.total_steps=total_steps
        self.sampling_metric='Score'
        self.min_values=np.array(min_values)
        self.max_values=np.array(max_values)
        self.previous_reward=0
        self.obs_df=[]
        self.total_file_size= total_file_size
        self.current_download_size=0

    def reset(self):
        self.current_observation = np.zeros(40,) # initialize current observation
        self.old_action=None
        self.step_number=0
        self.previous_reward=0
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
        self.obs_df=[]
        self.current_download_size=0
        return self.current_observation

    def step(self, action):
        # perform action using transfer_service
        action_1,action_2=action

        self.current_action_parallelism_value+=self.action_i_d_array[action_1]
        if self.current_action_parallelism_value<self.min_action:
            self.current_action_parallelism_value=self.min_action
        elif self.current_action_parallelism_value>self.max_action:
            self.current_action_parallelism_value=self.max_action

        self.current_action_concurrency_value+=self.action_i_d_array[action_2]
        if self.current_action_concurrency_value<self.min_action:
            self.current_action_concurrency_value=self.min_action
        elif self.current_action_concurrency_value>self.max_action:
            self.current_action_concurrency_value=self.max_action


        action_t=(self.current_action_parallelism_value,self.current_action_concurrency_value)

        if self.old_action==None:
            done=False
            try:
                work_df=self.initial_dfs[action_t]
                if work_df.empty:
                    observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            energy_part_reward=np.mean(e_array)
            if energy_part_reward > self.energy_sla:
                energy_penalty=energy_part_reward-self.energy_sla
            else:
                energy_penalty=0
            reward_=np.mean(t_array)-energy_penalty
            self.old_action=action_t
            reward=reward_- self.previous_reward
            self.previous_reward=reward_

        elif self.old_action==action_t:
            done=False
            try:
                work_df=self.initial_dfs[action_t]
                if work_df.empty:
                    observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            energy_part_reward=np.mean(e_array)
            if energy_part_reward > self.energy_sla:
                energy_penalty=energy_part_reward-self.energy_sla
            else:
                energy_penalty=0
            reward_=np.mean(t_array)-energy_penalty
            self.old_action=action_t
            reward=reward_-self.previous_reward
            self.previous_reward=reward_

        else:
            done=False
            key_name=(self.old_action, action_t)
            try:
                work_df=self.transaction_dfs[key_name]
                if work_df.empty:
                    try:
                        work_df_=self.initial_dfs[action_t]
                        if work_df_.empty:
                            observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                        else:
                            observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
                    except:
                        observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.transaction_dfs[key_name],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            energy_part_reward=np.mean(e_array)
            if energy_part_reward > self.energy_sla:
                energy_penalty=energy_part_reward-self.energy_sla
            else:
                energy_penalty=0
            reward_=np.mean(t_array)-energy_penalty
            self.old_action=action_t
            reward=reward_-self.previous_reward
            self.previous_reward=reward_


        self.step_number+=1
        if self.current_download_size>=self.total_file_size:
            done=True
        observation = np.asarray(observation).astype(np.float32)
        self.current_observation=observation
        return self.current_observation, reward, done, {}

    def bayes_step(self,action):
        params = [1 if x<1 else int(np.round(x)) for x in action]
        self.current_action_array_index+=self.action_i_d_array[params[0]]
        if self.current_action_array_index<self.min_action:
            self.current_action_array_index=self.min_action
        elif self.current_action_array_index>self.max_action:
            self.current_action_array_index=self.max_action
        action_t=self.action_array[self.current_action_array_index]

        print("Bayes Step: ",action_t)
        obs,score_b,done_b,__=self.step(action_t)
        print("Bayes Step Score: ", score_b)
        return np.round(score_b * (-1))

    def render(self, mode="human"):
        pass

    def close(self):
        self.reset()


class transferClass_multi_action_increase_decrease_throughput_SLA(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,transaction_dfs,initial_dfs,backup_initial_dfs,optimizer,total_steps=20,min_values=[0.00, 0.0, 1, 1, -3081.0, 0.0, 0.0, 0.0],max_values = [19.2, 2.0, 8, 8, 16.0, 70.1, 120.0, 74.166],total_file_size=256,throughput_sla=8):
        super().__init__()
        self.min_action=1
        self.max_action=8
        self.action_i_d_array=[1,4,0,-1,-4]

        self.transaction_dfs = transaction_dfs
        self.initial_dfs= initial_dfs
        self.backup_initial_dfs=backup_initial_dfs

        self.action_space = spaces.MultiDiscrete([5, 5])  # example action space
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(40,), dtype=np.float32) # example observation space
        self.current_observation = np.zeros(40,) # initialize current observation
        self.throughput_sla=throughput_sla
        self.optimizer=optimizer
        self.old_action=None
        self.step_number=0
        self.total_steps=total_steps
        self.sampling_metric='Score'
        self.min_values=np.array(min_values)
        self.max_values=np.array(max_values)
        self.previous_reward=0
        self.obs_df=[]
        self.total_file_size= total_file_size
        self.current_download_size=0

    def reset(self):
        self.current_observation = np.zeros(40,) # initialize current observation
        self.old_action=None
        self.step_number=0
        self.previous_reward=0
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
        self.obs_df=[]
        self.current_download_size=0
        return self.current_observation

    def step(self, action):
        # perform action using transfer_service
        action_1,action_2=action

        self.current_action_parallelism_value+=self.action_i_d_array[action_1]
        if self.current_action_parallelism_value<self.min_action:
            self.current_action_parallelism_value=self.min_action
        elif self.current_action_parallelism_value>self.max_action:
            self.current_action_parallelism_value=self.max_action

        self.current_action_concurrency_value+=self.action_i_d_array[action_2]
        if self.current_action_concurrency_value<self.min_action:
            self.current_action_concurrency_value=self.min_action
        elif self.current_action_concurrency_value>self.max_action:
            self.current_action_concurrency_value=self.max_action


        action_t=(self.current_action_parallelism_value,self.current_action_concurrency_value)

        if self.old_action==None:
            done=False
            try:
                work_df=self.initial_dfs[action_t]
                if work_df.empty:
                    observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            throughput_part_reward=np.mean(t_array)
            if throughput_part_reward < self.throughput_sla:
                throughput_penalty=(self.throughput_sla-throughput_part_reward)*5
            else:
                throughput_penalty=0
            reward_=throughput_part_reward-throughput_penalty
            self.old_action=action_t
            reward=reward_- self.previous_reward
            self.previous_reward=reward_

        elif self.old_action==action_t:
            done=False
            try:
                work_df=self.initial_dfs[action_t]
                if work_df.empty:
                    observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            energy_part_reward=np.mean(e_array)
            throughput_part_reward=np.mean(t_array)
            if throughput_part_reward < self.throughput_sla:
                throughput_penalty=(self.throughput_sla-throughput_part_reward)*5
            else:
                throughput_penalty=0
            reward_=throughput_part_reward-throughput_penalty
            self.old_action=action_t
            reward=reward_- self.previous_reward
            self.previous_reward=reward_

        else:
            done=False
            key_name=(self.old_action, action_t)
            try:
                work_df=self.transaction_dfs[key_name]
                if work_df.empty:
                    try:
                        work_df_=self.initial_dfs[action_t]
                        if work_df_.empty:
                            observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                        else:
                            observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
                    except:
                        observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.transaction_dfs[key_name],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            throughput_part_reward=np.mean(t_array)
            if throughput_part_reward < self.throughput_sla:
                throughput_penalty=(self.throughput_sla-throughput_part_reward)*5
            else:
                throughput_penalty=0
            reward_=throughput_part_reward-throughput_penalty
            self.old_action=action_t
            reward=reward_- self.previous_reward
            self.previous_reward=reward_

        self.step_number+=1
        if self.current_download_size>=self.total_file_size:
            done=True
        observation = np.asarray(observation).astype(np.float32)
        self.current_observation=observation
        return self.current_observation, reward, done, {}

    def bayes_step(self,action):
        params = [1 if x<1 else int(np.round(x)) for x in action]
        self.current_action_array_index+=self.action_i_d_array[params[0]]
        if self.current_action_array_index<self.min_action:
            self.current_action_array_index=self.min_action
        elif self.current_action_array_index>self.max_action:
            self.current_action_array_index=self.max_action
        action_t=self.action_array[self.current_action_array_index]

        print("Bayes Step: ",action_t)
        obs,score_b,done_b,__=self.step(action_t)
        print("Bayes Step Score: ", score_b)
        return np.round(score_b * (-1))

    def render(self, mode="human"):
        pass

    def close(self):
        self.reset()


class transferClass_multi_action_increase_decrease_energyEfficiency_SLA(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,transaction_dfs,initial_dfs,backup_initial_dfs,optimizer,total_steps=20,min_values=[0.00, 0.0, 1, 1, -3081.0, 0.0, 0.0, 0.0],max_values = [19.2, 2.0, 8, 8, 16.0, 70.1, 120.0, 74.166],total_file_size=256):
        super().__init__()
        self.min_action=1
        self.max_action=8
        self.action_i_d_array=[1,4,0,-1,-4]

        self.transaction_dfs = transaction_dfs
        self.initial_dfs= initial_dfs
        self.backup_initial_dfs=backup_initial_dfs

        self.action_space = spaces.MultiDiscrete([5, 5])  # example action space
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(40,), dtype=np.float32) # example observation space
        self.current_observation = np.zeros(40,) # initialize current observation
        self.optimizer=optimizer
        self.old_action=None
        self.step_number=0
        self.total_steps=total_steps
        self.sampling_metric='Score'
        self.min_values=np.array(min_values)
        self.max_values=np.array(max_values)
        self.previous_reward=0
        self.obs_df=[]
        self.total_file_size= total_file_size
        self.current_download_size=0

    def reset(self):
        self.current_observation = np.zeros(40,) # initialize current observation
        self.old_action=None
        self.step_number=0
        self.previous_reward=0
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
        self.obs_df=[]
        self.current_download_size=0
        return self.current_observation

    def step(self, action):
        # perform action using transfer_service
        action_1,action_2=action

        self.current_action_parallelism_value+=self.action_i_d_array[action_1]
        if self.current_action_parallelism_value<self.min_action:
            self.current_action_parallelism_value=self.min_action
        elif self.current_action_parallelism_value>self.max_action:
            self.current_action_parallelism_value=self.max_action

        self.current_action_concurrency_value+=self.action_i_d_array[action_2]
        if self.current_action_concurrency_value<self.min_action:
            self.current_action_concurrency_value=self.min_action
        elif self.current_action_concurrency_value>self.max_action:
            self.current_action_concurrency_value=self.max_action


        action_t=(self.current_action_parallelism_value,self.current_action_concurrency_value)

        if self.old_action==None:
            done=False
            try:
                work_df=self.initial_dfs[action_t]
                if work_df.empty:
                    observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            energy_part_reward=np.max(e_array)
            throughput_part_reward=np.mean(t_array)
            reward_=(throughput_part_reward*10)/energy_part_reward
            self.old_action=action_t
            reward=reward_-self.previous_reward
            self.previous_reward=reward_

        elif self.old_action==action_t:
            done=False
            try:
                work_df=self.initial_dfs[action_t]
                if work_df.empty:
                    observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            energy_part_reward=np.max(e_array)
            throughput_part_reward=np.mean(t_array)
            reward_=(throughput_part_reward*10)/energy_part_reward
            self.old_action=action_t
            reward=reward_-self.previous_reward
            self.previous_reward=reward_

        else:
            done=False
            key_name=(self.old_action, action_t)
            try:
                work_df=self.transaction_dfs[key_name]
                if work_df.empty:
                    try:
                        work_df_=self.initial_dfs[action_t]
                        if work_df_.empty:
                            observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                        else:
                            observation_df=sample_row_and_neighbors(self.initial_dfs[action_t],self.sampling_metric)
                    except:
                        observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
                else:
                    observation_df=sample_row_and_neighbors(self.transaction_dfs[key_name],self.sampling_metric)
            except:
                observation_df=sample_row_and_neighbors(self.backup_initial_dfs[action_t],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array,e_array,t_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            self.current_download_size+=np.sum(t_array)
            energy_part_reward=np.max(e_array)
            throughput_part_reward=np.mean(t_array)
            reward_=(throughput_part_reward*10)/energy_part_reward
            self.old_action=action_t
            reward=reward_-self.previous_reward
            self.previous_reward=reward_

        self.step_number+=1
        if self.current_download_size>=self.total_file_size:
            done=True
        observation = np.asarray(observation).astype(np.float32)
        self.current_observation=observation
        return self.current_observation, reward, done, {}

    def bayes_step(self,action):
        params = [1 if x<1 else int(np.round(x)) for x in action]
        self.current_action_array_index+=self.action_i_d_array[params[0]]
        if self.current_action_array_index<self.min_action:
            self.current_action_array_index=self.min_action
        elif self.current_action_array_index>self.max_action:
            self.current_action_array_index=self.max_action
        action_t=self.action_array[self.current_action_array_index]

        print("Bayes Step: ",action_t)
        obs,score_b,done_b,__=self.step(action_t)
        print("Bayes Step Score: ", score_b)
        return np.round(score_b * (-1))

    def render(self, mode="human"):
        pass

    def close(self):
        self.reset()
