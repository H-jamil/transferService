import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import gymnasium as gym

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
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a, logprob_a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
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
    def __init__(self, env, obs_mean, obs_std, reward_scale=1.0):
        super().__init__(env)
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.reward_scale = reward_scale

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        # observation, reward, done, _, info = self.env.step(action)
        observation, reward, done, info = self.env.step(action)
        normalized_obs = self.normalize_observation(observation)
        if reward != 1000000:
          normalized_reward = reward * self.reward_scale
        else:
          normalized_reward = 100
        # return normalized_obs, normalized_reward, done, _ ,info
        return normalized_obs, normalized_reward, done, info

    def normalize_observation(self, observation):
      # try:
      #   return (observation - self.obs_mean) / self.obs_std
      # except:
      #   return np.zeros(40,)
      EPSILON = 1e-10  # Small constant to prevent division by zero
      normalized_observation = (observation - self.obs_mean) / (self.obs_std + EPSILON)
      return normalized_observation

def compute_observation_stats(env, num_samples=10):
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
    print('rewards from normalization function', rewards)
    filtered_rewards = [r for r in rewards if r != 1000000]

    # Calculate mean and std of filtered rewards
    mean_reward = np.mean(filtered_rewards)
    std_reward = np.std(filtered_rewards)

    return np.mean(observations, axis=0), np.std(observations, axis=0), mean_reward, std_reward
