from src import *
import argparse

def create_environment(env_class, cluster_dict, initial_value_cluster_dict, sla_type):
    return env_class(cluster_dict, initial_value_cluster_dict, sla_type)

def main(env_class):
    initial_value_cluster_dictionary = get_initial_cluster_dictionary()
    cluster_dictionary = get_cluster_transaction_dictionary()

    env = create_environment(env_class, cluster_dictionary, initial_value_cluster_dictionary, env_class.__name__)
    evaluation_env = create_environment(env_class, cluster_dictionary, initial_value_cluster_dictionary, env_class.__name__)
    print("Training for environment", env_class.__name__)
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[{'pi': [128, 128], 'vf': [128, 128]}])
    run_string = env_class.__name__
    ###########################
    # total_scores=0
    # s = env.reset()
    # action_list=[]
    # reward_list=[]
    # done = False
    # while not done:
    #     a=env.action_space.sample()
    #     s_next, r, done, info = env.step(a)
    #     # print( len(s_next))
    #     action_list.append(a)
    #     reward_list.append(r)
    #     total_scores += r
    #     s = s_next
    # accumulator_df = pd.concat(env.obs_df)  # Add more DataFrames in the list if needed
    # env.close()
    # print(f"Total Reward: {total_scores}")
    # print(f"actions {action_list},   {len(action_list)}")
    # print(f"rewards {reward_list},  {len(reward_list)}")
    # print(accumulator_df)
    # print(f"Total downloaded {accumulator_df['Throughput'].sum()} Gb")
    # print(f"Download speed {accumulator_df['Throughput'].mean()} Gbps")
    #####################################
    model = PPO("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1,
                tensorboard_log=f"./ppo_tensorboard_{run_string}/", ent_coef=0.01)

    eval_callback = EvalCallback(evaluation_env, best_model_save_path=f'./ppo_{run_string}/ppo_best_model/',
                                 log_path=f'./ppo_{run_string}/ppo_logs/', eval_freq=1000,
                                 deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=f'./ppo_{run_string}/ppo_checkpoints/',
                                             name_prefix=f'ppo_model_{run_string}')
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(total_timesteps=2000000, callback=callback)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env_class_type', help='Type of environment class to use for training',
                        choices=['energySLA', 'throughputSLA', 'scoreSLA', 'energyEfficiencySLA'])

    args = parser.parse_args()

    # Map the argument to the actual class
    env_classes = {
        'energySLA': transferClass_multi_action_increase_decrease_energy_SLA,
        'throughputSLA': transferClass_multi_action_increase_decrease_throughput_SLA,
        'scoreSLA': transferClass_multi_action_increase_decrease_score_SLA,
        'energyEfficiencySLA': transferClass_multi_action_increase_decrease_energyEfficiency_SLA
    }

    # Get the corresponding class from the argument
    env_class = env_classes.get(args.env_class_type)

    # Run main function with the specified environment class
    main(env_class)
