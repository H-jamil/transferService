import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import gym as gym_old
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
    def __init__(self, env, obs_min, obs_max, reward_scale=1.0):
        super().__init__(env)
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.reward_scale = reward_scale
        self.old_score = 0
        self.score_difference_positive_threshold = 2
        self.score_difference_negative_threshold = -2

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.normalize_observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        normalized_obs = self.normalize_observation(observation)

        if reward != 1000000:
            score_difference = reward - self.old_score
            self.old_score = reward
            if score_difference > self.score_difference_positive_threshold:
                # normalized_reward = 1
                normalized_reward = max(-10, min(10, score_difference))
            elif score_difference < self.score_difference_negative_threshold:
                # normalized_reward = -1
                normalized_reward = max(-10, min(10, score_difference))
            else:
                normalized_reward = 0
        else:
            normalized_reward = 20

        return normalized_obs, normalized_reward, done, info

    def normalize_observation(self, observation):
        EPSILON = 1e-10  # Small constant to prevent division by zero
        normalized_observation = (observation - self.obs_min) / (self.obs_max - self.obs_min + EPSILON)
        return normalized_observation

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

def normalize_and_flatten(df, min_values, max_values):
    # Drop the specified columns
    df = df.drop(columns=['Time', 'CC'])
    score_array = df['Score'].values
    # Normalize each column
    normalized_df = (df - min_values) / (max_values - min_values)
    # Flatten the DataFrame to a single NumPy array
    flattened_array = normalized_df.values.flatten()

    return flattened_array,score_array

class transferClass(gym_old.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,transaction_dfs,initial_dfs,optimizer,total_steps=20,min_values=[0.0, 0.0, -75.0, 0.0, 0.0, 0.0, 1, 1],max_values = [19.52, 2.0, 18.0, 89.9, 110.0, 2.0, 8, 8]):
        super().__init__()
        self.action_array= [(1,1),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8)]
        self.transaction_dfs = transaction_dfs
        self.initial_dfs= initial_dfs
        self.action_space = spaces.Discrete(9) # example action space
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
        self.reward_threshold=0.5
        self.obs_df=[]


    def reset(self):
        self.current_observation = np.zeros(40,) # initialize current observation
        self.old_action=None
        self.step_number=0
        self.previous_reward=0
        self.obs_df=[]
        return self.current_observation

    def step(self, action):
        # perform action using transfer_service
        if action==0:
            action=1
        if self.old_action==None:
            done=False
            key_name=f'concurrency_{action}'
            observation_df=sample_row_and_neighbors(self.initial_dfs[key_name],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            # reward=np.max(result_array)
            # reward=np.min(result_array)
            reward_=np.mean(result_array)
            self.old_action=action
            # if reward_- self.previous_reward >=self.reward_threshold:
            #     reward=2
            # elif reward_- self.previous_reward <= -self.reward_threshold:
            #     reward= -1
            # else:
            #     reward=0
            # reward = reward_ - self.previous_reward
            reward = round(reward_ - self.previous_reward, 3)
            self.previous_reward=reward_

        elif self.old_action==action:
            done=False
            key_name=f'concurrency_{action}'
            observation_df=sample_row_and_neighbors(self.initial_dfs[key_name],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            # reward=np.max(result_array)
            # reward=np.min(result_array)
            reward_=np.mean(result_array)
            self.old_action=action
            # if reward_- self.previous_reward >=self.reward_threshold:
            #     reward=2
            # elif reward_- self.previous_reward <= -self.reward_threshold:
            #     reward= -1
            # else:
            #     reward=0
            # reward = reward_ - self.previous_reward
            reward = round(reward_ - self.previous_reward, 3)
            self.previous_reward=reward_

        else:
            done=False
            key_name=f'concurrency_{self.old_action}_{action}'
            observation_df=sample_row_and_neighbors(self.transaction_dfs[key_name],self.sampling_metric)
            self.obs_df.append(observation_df)
            observation,result_array=normalize_and_flatten(observation_df,self.min_values,self.max_values)
            # reward=np.max(result_array)
            # reward=np.min(result_array)
            reward_=np.mean(result_array)
            self.old_action=action
            # if reward_- self.previous_reward >=self.reward_threshold:
            #     reward=2
            # elif reward_- self.previous_reward <= -self.reward_threshold:
            #     reward= -1
            # else:
            #     reward=0
            # reward = reward_ - self.previous_reward
            reward = round(reward_ - self.previous_reward, 3)
            self.previous_reward=reward_

        self.step_number+=1

        if self.step_number>=self.total_steps:
            done=True
            # self.reset()
        observation=observation.astype(np.float32)
        self.current_observation=observation
        return self.current_observation, reward, done, {}

    def bayes_step(self,action):
        params = [1 if x<1 else int(np.round(x)) for x in action]
        print("Bayes Step: ",params)
        if params[0] > 8:
            params[0] = 8
        obs,score_b,done_b,__=self.step(params[0])
        print("Bayes Step Score: ", score_b)
        return np.round(score_b * (-1))

    def render(self, mode="human"):
        pass

    def close(self):
        self.reset()

def custom_sort_key(item):
    # Sort single integers first, then tuples
    return (0, item) if isinstance(item, int) else (1, item)
