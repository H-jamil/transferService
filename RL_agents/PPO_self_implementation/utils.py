import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import gym as gym_old
# import gym
# from gym import Env
from gym import spaces
import numpy as np
import gymnasium as gym
import os
import re
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
from transferService import transferService
import copy
import random
import logging as log
import time


# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, net_width):
#         super(Actor, self).__init__()
#         # Initialize layers
#         self.l1 = nn.Linear(state_dim, net_width)
#         self.l2 = nn.Linear(net_width, net_width)
#         self.l3 = nn.Linear(net_width, net_width)  # Additional hidden layer
#         self.l4 = nn.Linear(net_width, action_dim)  # Output layer

#     def forward(self, state):
#         # Apply layers with tanh activation function for hidden layers
#         n = torch.tanh(self.l1(state))
#         n = torch.tanh(self.l2(n))
#         n = torch.tanh(self.l3(n))  # Activation for additional hidden layer
#         return n

#     def pi(self, state, softmax_dim=0):
#         # Get the policy from the state
#         n = self.forward(state)
#         prob = F.softmax(self.l4(n), dim=softmax_dim)  # Apply softmax to the output layer
#         return prob

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob


# class Critic(nn.Module):
#     def __init__(self, state_dim, net_width):
#         super(Critic, self).__init__()

#         # Initialize layers
#         self.C1 = nn.Linear(state_dim, net_width)
#         self.C2 = nn.Linear(net_width, net_width)
#         self.C3 = nn.Linear(net_width, net_width)  # Additional hidden layer
#         self.C4 = nn.Linear(net_width, 1)  # Output layer

#     def forward(self, state):
#         # Apply layers with relu activation function for hidden layers
#         v = torch.relu(self.C1(state))
#         v = torch.relu(self.C2(v))
#         v = torch.relu(self.C3(v))  # Activation for additional hidden layer
#         v = self.C4(v)  # No activation function before the output
#         return v

class Critic(nn.Module):
    def __init__(self, state_dim,net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)
        return v

def evaluate_policy(env, agent, turns = 3):
    total_scores = 0
    for j in range(turns):
        # s, info = env.reset()
        s = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a, logprob_a = agent.select_action(s, deterministic=True)
            # s_next, r, dw, tr, info = env.step(a)
            s_next, r, done, info = env.step(a)
            # done = (dw or tr)

            total_scores += r
            s = s_next
        env.close()
    return int(total_scores/turns)


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise

class NormalizeObservationAndRewardWrapper(gym.Wrapper):
    def __init__(self, env, min_values=[0.0, 0.0, -75.0, 0.0, 0.0, 0.0, 1, 1],max_values = [19.52, 2.0, 18.0, 89.9, 110.0, 2.0, 8, 8], reward_scale=1.0):
        super().__init__(env)
        self.min_values=np.array(min_values)
        self.max_values=np.array(max_values)
        self.reward_scale = reward_scale
        self.old_score = 0
        self.score_difference_positive_threshold = 2
        self.score_difference_negative_threshold = -2

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        # return self.normalize_observation(observation)
        return observation.astype(np.float32)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        normalized_obs,result_array = self.normalize_observation(observation)

        if reward != 1000000:
            # score_difference = reward - self.old_score
            # reward_=np.mean(result_array)
            score_difference = reward - self.old_score
            self.old_score = reward
            return normalized_obs, round(score_difference,3), done, info
        else:
            return normalized_obs, 0 , done, info

    def normalize_observation(self, observation):
        # print(f"observation: {observation}")
        observation_df=np_array_to_df_with_reordered_columns(observation)
        normalized_observation,result_array=normalize_and_flatten_real(observation_df,self.min_values,self.max_values)
        return normalized_observation.astype(np.float32),result_array

# Function to convert np array to a dataframe with specified columns and shape, with reordered columns
def np_array_to_df_with_reordered_columns(arr, num_rows=5, num_cols=8, col_names=None):
    if col_names is None:
        col_names = ['Throughput', 'concurrency', 'parallelism', 'receiver_lr',
                     'Score', 'RTT', 'Energy', 'sender_lr']

    # Ensure that the array has the correct number of elements
    if len(arr) != num_rows * num_cols:
        raise ValueError("The numpy array does not have the correct number of elements")

    # Reshape the array and create the dataframe
    reshaped_array = arr.reshape((num_rows, num_cols))
    df = pd.DataFrame(reshaped_array, columns=col_names)

    # Reorder the columns
    reordered_col_names = ['Throughput', 'receiver_lr', 'Score', 'RTT', 'Energy', 'sender_lr', 'concurrency', 'parallelism']
    df = df[reordered_col_names]

    return df


def compute_observation_stats(env, num_samples=50):
    """
    Compute the mean and standard deviation of observations from the environment.

    :param env: The gym environment.
    :param num_samples: Number of samples to collect.
    :return: mean, std of observations
    """
    observations = []
    rewards=[]
    for _ in range(num_samples):
        obs = env.reset()
        observations.append(obs)
        done = False
        while not done:
            action = env.action_space.sample()  # replace with your action sampling strategy if needed
            print('action', action, type(action))
            obs, reward, done, *_ = env.step(action)
            observations.append(obs)
            rewards.append(reward)
        env.close()
    observations = np.array(observations)
    print('observations.shape', observations.shape)
    print('observations', observations)
    # Calculate min and max for each observation dimension
    min_obs = np.min(observations, axis=0)
    max_obs = np.max(observations, axis=0)

    # Optional: handling rewards, similar to previous function
    filtered_rewards = [r for r in rewards if r != 1000000]
    mean_reward = np.mean(filtered_rewards)
    std_reward = np.std(filtered_rewards)

    return min_obs, max_obs, mean_reward, std_reward

def get_concurrency_parallelism(cc_value):
    value = int(sqrt(cc_value))
    return value, value

def process_log_file(full_path):
    if os.stat(full_path).st_size == 0:
        print(f"Skipping empty file: {filename}")
        return 0
    with open(full_path, 'r') as file:
        data = []
        last_non_zero_throughput = None
        for line in file:
            match = re.search(r'(\d+\.\d+).*Throughput @(\d+\.\d+)s:\s+(\d+\.\d+)Gbps, lossRate: (\d+\.\d+|\d+) CC:(\d+)\s+score:(-?\d+\.\d+)\s+rtt:(\d+\.\d+) ms energy:(\d+\.\d+) Jules s-plr:([\deE.-]+)', line)
            if match:
                time = datetime.fromtimestamp(float(match.group(1)))
                throughput = "{:.6f}".format(float(match.group(3)))
                loss_rate = "{:.6f}".format(float(match.group(4)))
                cc = int(match.group(5))
                score = "{:.6f}".format(float(match.group(6)))
                rtt = "{:.6f}".format(float(match.group(7)))
                energy = "{:.6f}".format(float(match.group(8)))
                sender_lr = "{:.6f}".format(float(match.group(9)))

                # If you need them as floats and not strings
                throughput = float(throughput)
                loss_rate = float(loss_rate)
                score = float(score)
                rtt = float(rtt)
                energy = float(energy)
                sender_lr = float(sender_lr)
                concurrency, parallelism = get_concurrency_parallelism(cc)
                data.append([time, throughput, loss_rate, cc, score, rtt, energy, sender_lr, concurrency, parallelism])

        if data:
            df = pd.DataFrame(data, columns=['Time', 'Throughput', 'receiver_lr', 'CC', 'Score', 'RTT', 'Energy', 'sender_lr', 'concurrency', 'parallelism'])

        else:
            df=pd.DataFrame()
            print(f"No valid data in file: {filename}")

    return df

def find_transitions(df, column_name, start_value, target_value):
    # Create a new empty DataFrame with the same columns
    new_df = pd.DataFrame(columns=df.columns)
    # Flag to mark if we are currently in a transition block
    in_transition = False
    # Iterate through the DataFrame
    for index, row in df.iterrows():
        # Check for transition start
        if row[column_name] == start_value and not in_transition:
            in_transition = True
        # Check for transition end
        elif row[column_name] == target_value and in_transition:
            new_df = new_df.append(row)
        # If the value is neither start nor target, reset the flag
        elif row[column_name] != start_value:
            in_transition = False

    return new_df

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

def sample_row_and_neighbors(df, column_name):
    """
    Samples a row based on the distribution of a specific column and returns it with its 4 neighboring rows.

    :param df: The DataFrame to sample from.
    :param column_name: The column whose distribution to use for sampling.
    :return: A DataFrame containing the sampled row and its 4 neighbors.
    """
    # Calculate the frequency distribution of the column
    df=df.reset_index(drop=True)
    probabilities = df[column_name].value_counts(normalize=True)

    # Map these probabilities back to the DataFrame's index
    probabilities = df[column_name].map(probabilities)
    # Sample one row using these probabilities
    sampled_row = df.sample(n=1, weights=probabilities)
    sampled_index = sampled_row.index[0]
    # Ensure the sampled index is within the DataFrame's range
    if sampled_index >= len(df):
        raise ValueError("Sampled index is out of DataFrame's range.")
    # Determine the range of indices to return
    if sampled_index < 5:
        start_index = sampled_index
        end_index = min(sampled_index + 4, len(df) - 1)
    else:
        start_index = sampled_index - 4
        end_index = sampled_index
    # Select the range from start_index to end_index
    return df.iloc[start_index:end_index + 1]

def normalize_and_flatten_real(df, min_values, max_values):
    # Drop the specified columns
    score_array = df['Score'].values
    # Normalize each column
    normalized_df = (df - min_values) / (max_values - min_values)
    # Flatten the DataFrame to a single NumPy array
    flattened_array = normalized_df.values.flatten()

    return flattened_array,score_array

def normalize_and_flatten(df, min_values, max_values):
    # Drop the specified columns
    df = df.drop(columns=['Time', 'CC'])
    score_array = df['Score'].values
    # Normalize each column
    normalized_df = (df - min_values) / (max_values - min_values)
    # Flatten the DataFrame to a single NumPy array
    flattened_array = normalized_df.values.flatten()

    return flattened_array,score_array

# class transferClass(gym_old.Env):
#     metadata = {"render_modes": ["human"], "render_fps": 30}
#     def __init__(self,transaction_dfs,initial_dfs,optimizer,total_steps=20,min_values=[0.0, 0.0, -75.0, 0.0, 0.0, 0.0, 1, 1],max_values = [19.52, 2.0, 18.0, 89.9, 110.0, 2.0, 8, 8]):
#         super().__init__()
#         self.action_array= [(1,1),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8)]
#         self.transaction_dfs = transaction_dfs
#         self.initial_dfs= initial_dfs
#         self.action_space = spaces.Discrete(9) # example action space
#         self.observation_space = spaces.Box(low=0, high=np.inf, shape=(40,), dtype=np.float32) # example observation space
#         self.current_observation = np.zeros(40,) # initialize current observation
#         self.optimizer=optimizer
#         self.old_action=None
#         self.step_number=0
#         self.total_steps=total_steps
#         self.sampling_metric='Score'
#         self.min_values=np.array(min_values)
#         self.max_values=np.array(max_values)
#         self.previous_reward=0
#         self.reward_threshold=0.5
#         self.obs_df=[]


#     def reset(self):
#         self.current_observation = np.zeros(40,) # initialize current observation
#         self.old_action=None
#         self.step_number=0
#         self.previous_reward=0
#         self.obs_df=[]
#         return self.current_observation

#     def step(self, action):
#         # perform action using transfer_service
#         if action==0:
#             action=1
#         if self.old_action==None:
#             done=False
#             key_name=f'concurrency_{action}'
#             observation_df=sample_row_and_neighbors(self.initial_dfs[key_name],self.sampling_metric)
#             self.obs_df.append(observation_df)
#             observation,result_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
#             # reward=np.max(result_array)
#             # reward=np.min(result_array)
#             reward_=np.mean(result_array)
#             self.old_action=action
#             # if reward_- self.previous_reward >=self.reward_threshold:
#             #     reward=2
#             # elif reward_- self.previous_reward <= -self.reward_threshold:
#             #     reward= -1
#             # else:
#             #     reward=0
#             # reward = reward_ - self.previous_reward
#             reward = round(reward_ - self.previous_reward, 3)
#             self.previous_reward=reward_

#         elif self.old_action==action:
#             done=False
#             key_name=f'concurrency_{action}'
#             observation_df=sample_row_and_neighbors(self.initial_dfs[key_name],self.sampling_metric)
#             self.obs_df.append(observation_df)
#             observation,result_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
#             # reward=np.max(result_array)
#             # reward=np.min(result_array)
#             reward_=np.mean(result_array)
#             self.old_action=action
#             # if reward_- self.previous_reward >=self.reward_threshold:
#             #     reward=2
#             # elif reward_- self.previous_reward <= -self.reward_threshold:
#             #     reward= -1
#             # else:
#             #     reward=0
#             # reward = reward_ - self.previous_reward
#             reward = round(reward_ - self.previous_reward, 3)
#             self.previous_reward=reward_

#         else:
#             done=False
#             key_name=f'concurrency_{self.old_action}_{action}'
#             observation_df=sample_row_and_neighbors(self.transaction_dfs[key_name],self.sampling_metric)
#             self.obs_df.append(observation_df)
#             observation,result_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
#             # reward=np.max(result_array)
#             # reward=np.min(result_array)
#             reward_=np.mean(result_array)
#             self.old_action=action
#             # if reward_- self.previous_reward >=self.reward_threshold:
#             #     reward=2
#             # elif reward_- self.previous_reward <= -self.reward_threshold:
#             #     reward= -1
#             # else:
#             #     reward=0
#             # reward = reward_ - self.previous_reward
#             reward = round(reward_ - self.previous_reward, 3)
#             self.previous_reward=reward_

#         self.step_number+=1

#         if self.step_number>=self.total_steps:
#             done=True
#             # self.reset()
#         observation=observation.astype(np.float32)
#         self.current_observation=observation
#         return self.current_observation, reward, done, {}

#     def bayes_step(self,action):
#         params = [1 if x<1 else int(np.round(x)) for x in action]
#         # print("Bayes Step: ",params)
#         if params[0] > 8:
#             params[0] = 8
#         obs,score_b,done_b,__=self.step(params[0])
#         # print("Bayes Step Score: ", score_b)
#         if done_b == False:
#             return np.round(score_b * (-1))
#         else:
#             return -1000000


#     def render(self, mode="human"):
#         pass

#     def close(self):
#         self.reset()

def custom_sort_key(item):
    # Sort single integers first, then tuples
    return (0, item) if isinstance(item, int) else (1, item)


# class transferClass_real(gym.Env):
#   metadata = {"render_modes": ["human"], "render_fps": 30}
#   def __init__(self,transferServiceObject,optimizer):
#     super().__init__()
#     self.action_array=[(1,1),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8)]
#     self.transfer_service = transferServiceObject
#     self.action_space =spaces.Discrete(9) # example action space
#     self.observation_space = spaces.Box(low=0, high=np.inf, shape=(40,), dtype=np.float32) # example observation space
#     self.current_observation = np.zeros(40,) # initialize current observation
#     self.optimizer=optimizer

#   def reset(self):
#     self.current_observation = self.transfer_service.reset() # get initial observation
#     # self.current_observation = np.zeros(40,) # initialize current observation
#     return self.current_observation

#   def step(self, action):
#     new_observation,reward=self.transfer_service.step(self.action_array[action][0],self.action_array[action][1])
#     if reward==1000000:
#       done=True
#     else:
#       done=False
#     new_observation = new_observation.astype(np.float32)
#     self.current_observation=new_observation
#     return self.current_observation, reward, done, {}

#   def bayes_step(self,action):
#     params = [1 if x<1 else int(np.round(x)) for x in action]
#     print("Bayes Step: ",params)
#     if params[0] > 8:
#       params[0] = 8
#     obs,score_b,done_b,__=self.step(params[0])
#     print("Bayes Step Score: ", score_b)
#     return np.round(score_b * (-1),3)

#   def render(self, mode="human"):
#     pass

#   def close(self):
#     return self.transfer_service.cleanup() # close transfer_service


def energy_monitor():
    current_energy=0
    try:
        energy_old = read_energy()
    except Exception as e:
        print(f"Error reading initial energy: {e}")
        energy_old = 0
    time.sleep(1)  # Sleep for a bit before re-checking the condition
    try:
        energy_now = read_energy()
    except Exception as e:
        print(f"Error reading current energy: {e}")
        energy_now = energy_old  # Use the old value if there's an error

    current_energy = int((energy_now - energy_old) / 1000000)

    return current_energy

def read_energy():
    try:
        with open("/sys/class/powercap/intel-rapl:0/energy_uj", "r") as file:
            energy = float(file.read())
        return energy
    except IOError as e:
        print(f"Failed to open energy_uj file: {e}")
        return 0


class transferClassReal_MA_ID(gym_old.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,transferServiceObject,optimizer,min_values=[0.32, 0.0, 1, 1, -268.0, 0.0, 43.0, 0.0],max_values = [17.6, 6.0, 8, 8, 16.0, 103.0, 102.0, 6.6]):
        super().__init__()
        self.transfer_service = transferServiceObject
        self.min_action=1
        self.max_action=8
        self.action_i_d_array=[1,4,0,-1,-4]
        self.action_space = spaces.MultiDiscrete([5, 5])  # example action space
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(40,), dtype=np.float32) # example observation space
        self.current_observation = np.zeros(40,) # initialize current observation
        self.optimizer=optimizer
        self.min_values=np.array(min_values)
        self.max_values=np.array(max_values)
        self.previous_reward=0

    def reset(self):
        self.current_observation = self.transfer_service.reset() # get initial observation
        # self.current_observation = np.zeros(40,) # initialize current observation
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

        new_observation,reward_=self.transfer_service.step(self.current_action_parallelism_value,self.current_action_concurrency_value)
        if reward_==1000000:
             done=True
             reward=0
        else:
            done=False
            reward=reward_
        new_observation = new_observation.astype(np.float32)
        self.current_observation=new_observation
        return self.current_observation, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        return self.transfer_service.cleanup() # close transfer_service


class transferClassReal_MA_DG(gym_old.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,transferServiceObject,optimizer,min_values=[0.32, 0.0, 1, 1, -268.0, 0.0, 43.0, 0.0],max_values = [17.6, 6.0, 8, 8, 16.0, 103.0, 102.0, 6.6]):
        super().__init__()
        self.transfer_service = transferServiceObject
        self.min_action=1
        self.max_action=8
        # self.action_i_d_array=[1,4,0,-1,-4]
        self.action_space = spaces.MultiDiscrete([9, 9])  # example action space
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(40,), dtype=np.float32) # example observation space
        self.current_observation = np.zeros(40,) # initialize current observation
        self.optimizer=optimizer
        self.min_values=np.array(min_values)
        self.max_values=np.array(max_values)
        self.previous_reward=0

    def reset(self):
        self.current_observation = self.transfer_service.reset() # get initial observation
        # self.current_observation = np.zeros(40,) # initialize current observation
        return self.current_observation

    def step(self, action):
        # perform action using transfer_service
        action_1,action_2=action
        new_observation,reward_=self.transfer_service.step(action_1,action_2)
        if reward_==1000000:
             done=True
             reward=0
        else:
            done=False
            reward=reward_
        new_observation = new_observation.astype(np.float32)
        self.current_observation=new_observation
        return self.current_observation, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        return self.transfer_service.cleanup() # close transfer_service



def get_env(env_string='transferService', optimizer='multiA_Inc_Dec_trained', REMOTE_IP = "192.5.86.213", REMOTE_PORT = "80", INTERVAL = 1,INTERFACE = "eno1",SERVER_IP = '127.0.0.1',SERVER_PORT = 8080):
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
    transfer_service = transferService(REMOTE_IP, REMOTE_PORT, INTERVAL, INTERFACE, SERVER_IP, SERVER_PORT, optimizer, log)
    env = transferClassReal_MA_ID(transfer_service,optimizer)
    return env

def get_env_DG(env_string='transferService', optimizer='multiA_Inc_Dec_trained', REMOTE_IP = "192.5.86.213", REMOTE_PORT = "80", INTERVAL = 1,INTERFACE = "eno1",SERVER_IP = '127.0.0.1',SERVER_PORT = 8080):
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
    transfer_service = transferService(REMOTE_IP, REMOTE_PORT, INTERVAL, INTERFACE, SERVER_IP, SERVER_PORT, optimizer, log)
    env = transferClassReal_MA_DG(transfer_service,optimizer)
    return env
