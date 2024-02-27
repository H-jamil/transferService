from utils import *
import argparse

def create_environment(env_class, cluster_dict, initial_value_cluster_dict,initial_value_cluster_dict_bckup, sla_type):
    return env_class(cluster_dict, initial_value_cluster_dict,initial_value_cluster_dict_bckup,sla_type)

def main(env_class,pretrained_model_dir=None):
    initial_value_cluster_dictionary=get_initial_cluster_dictionary_retraining()
    initial_value_cluster_dictionary_bckup = get_initial_cluster_dictionary()
    cluster_dictionary = get_cluster_transaction_dictionary_retraining()

    env = create_environment(env_class, cluster_dictionary, initial_value_cluster_dictionary,initial_value_cluster_dictionary_bckup, env_class.__name__)
    evaluation_env = create_environment(env_class, cluster_dictionary, initial_value_cluster_dictionary, initial_value_cluster_dictionary_bckup, env_class.__name__)
    print("Training for environment", env_class.__name__)
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[{'pi': [128, 128], 'vf': [128, 128]}])
    run_string = env_class.__name__
    if pretrained_model_dir:
        model = PPO.load(pretrained_model_dir, env=env)
        tensorboard_log_dir = f"./ppo_tensorboard_{run_string}_retrain/"  # Separate directory for retraining logs
    else:
        model = PPO("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1,
                    tensorboard_log=f"./ppo_tensorboard_{run_string}/", ent_coef=0.01)
        tensorboard_log_dir = f"./ppo_tensorboard_{run_string}/"  # Original directory for training logs
    model.tensorboard_log = tensorboard_log_dir
    eval_callback = EvalCallback(evaluation_env, best_model_save_path=f'./ppo_{run_string}/ppo_best_model/',
                                 log_path=f'./ppo_{run_string}/ppo_logs/', eval_freq=1000,
                                 deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=f'./ppo_{run_string}/ppo_checkpoints/',
                                             name_prefix=f'ppo_model_{run_string}')
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(total_timesteps=1000000, callback=callback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env_class_type', help='Type of environment class to use for training',
                        choices=['energySLA', 'throughputSLA', 'scoreSLA', 'energyEfficiencySLA'])
    parser.add_argument('pretrained_model_directory', help=' Pretrained model directory',
                        choices=['/home/cc/transferService/RL_agents/PPO_self_implementation/simulator_design_multi_action_increase_decrease_all_four_final/ppo_transferClass_multi_action_increase_decrease_score_SLA/ppo_best_model', '/home/cc/transferService/RL_agents/PPO_self_implementation/simulator_design_multi_action_increase_decrease_all_four_final/ppo_transferClass_multi_action_increase_decrease_energy_SLA/ppo_best_model', '/home/cc/transferService/RL_agents/PPO_self_implementation/simulator_design_multi_action_increase_decrease_all_four_final/ppo_transferClass_multi_action_increase_decrease_throughput_SLA/ppo_best_model', '/home/cc/transferService/RL_agents/PPO_self_implementation/simulator_design_multi_action_increase_decrease_all_four_final/ppo_transferClass_multi_action_increase_decrease_energyEfficiency_SLA/ppo_best_model'])
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
    main(env_class,args.pretrained_model_directory)
