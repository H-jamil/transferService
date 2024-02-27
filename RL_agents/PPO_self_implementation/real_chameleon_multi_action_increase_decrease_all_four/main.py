from src_real_transfer import *

def main(optimer):
  if optimer == 'test':
    print("Testing the real transfer with random actions")
    env_ = get_env(optimizer='test_MA_ID')
    env=NormalizeObservationAndRewardWrapper(env_,sla_type='score')
    total_scores=0
    s = env.reset()
    action_list=[]
    reward_list=[]
    done = False
    while not done:
        # print(f"s: {s}")
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

  elif optimer == 'ppo_score_sla':
    print("Real Transfer with score sla")
    env_ = get_env(optimizer='ppo_score_sla_MA_ID')
    env=NormalizeObservationAndRewardWrapper(env_,sla_type='score')
    model = PPO.load("/home/cc/transferService/RL_agents/PPO_self_implementation/simulator_design_multi_action_increase_decrease_all_four_final/ppo_transferClass_multi_action_increase_decrease_score_SLA/ppo_best_model/best_model.zip")
    action_list=[]
    reward_list=[]
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_list.append(action)
        obs, reward, done, info = env.step(action)
        reward_list.append(reward)
        episode_reward += reward
    env.close()
    print(f"Total Reward: {episode_reward}")
    print(f"actions {action_list},   {len(action_list)}")
    print(f"rewards {reward_list},  {len(reward_list)}")

  elif optimer == 'ppo_energy_sla':
    print("Real Transfer with energy sla")
    env_ = get_env(optimizer='ppo_energy_sla_MA_ID')
    env=NormalizeObservationAndRewardWrapper(env_,sla_type='energy')
    model = PPO.load("/home/cc/transferService/RL_agents/PPO_self_implementation/simulator_design_multi_action_increase_decrease_all_four_final/ppo_transferClass_multi_action_increase_decrease_energy_SLA/ppo_best_model/best_model.zip")
    action_list=[]
    reward_list=[]
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_list.append(action)
        obs, reward, done, info = env.step(action)
        reward_list.append(reward)
        episode_reward += reward
    env.close()
    print(f"Total Reward: {episode_reward}")
    print(f"actions {action_list},   {len(action_list)}")
    print(f"rewards {reward_list},  {len(reward_list)}")

  elif optimer == 'ppo_throughput_sla':
    print("Real Transfer with throughput sla")
    env_ = get_env(optimizer='ppo_throughput_sla_MA_ID')
    env=NormalizeObservationAndRewardWrapper(env_,sla_type='throughput')
    model = PPO.load("/home/cc/transferService/RL_agents/PPO_self_implementation/simulator_design_multi_action_increase_decrease_all_four_final/ppo_transferClass_multi_action_increase_decrease_throughput_SLA/ppo_best_model/best_model.zip")
    action_list=[]
    reward_list=[]
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_list.append(action)
        obs, reward, done, info = env.step(action)
        reward_list.append(reward)
        episode_reward += reward
    env.close()
    print(f"Total Reward: {episode_reward}")
    print(f"actions {action_list},   {len(action_list)}")
    print(f"rewards {reward_list},  {len(reward_list)}")

  elif optimer == 'ppo_energyEfficiency_sla':
    print("Real Transfer with energyEfficiency sla")
    env_ = get_env(optimizer='ppo_energyEfficiency_sla_MA_ID')
    env=NormalizeObservationAndRewardWrapper(env_,sla_type='energyEfficiency')
    model = PPO.load("/home/cc/transferService/RL_agents/PPO_self_implementation/simulator_design_multi_action_increase_decrease_all_four_final/ppo_transferClass_multi_action_increase_decrease_energyEfficiency_SLA/ppo_best_model/best_model.zip")
    action_list=[]
    reward_list=[]
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_list.append(action)
        obs, reward, done, info = env.step(action)
        reward_list.append(reward)
        episode_reward += reward
    env.close()
    print(f"Total Reward: {episode_reward}")
    print(f"actions {action_list},   {len(action_list)}")
    print(f"rewards {reward_list},  {len(reward_list)}")



if __name__ == "__main__":
  total_run_number=20
  optimizer_list=['ppo_score_sla','ppo_energy_sla','ppo_throughput_sla','ppo_energyEfficiency_sla']
  for run in range(0,total_run_number):
    for optimizer in optimizer_list:
      print(f"Run: {run}, Optimizer: {optimizer}")
      main(optimizer)
      print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n")
