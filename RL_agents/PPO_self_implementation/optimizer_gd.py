import numpy as np
import random
import time
# from bayes_opt import BayesianOptimization
from skopt.space import Integer
from skopt import Optimizer, dummy_minimize
from scipy.optimize import minimize


def gradient_opt(transferEnvironment):
    max_action, count = transferEnvironment.action_space.n-1, 0
    soft_lim,least_cost = max_action,0
    values = []
    ccs = [2]
    theta = 0
    update_rate = 0.2

    while True:
        state, score, done, _ = transferEnvironment.step(ccs[-1]-1)
        # values.append(np.round(score * (-1)))
        values.append(np.round(score * (1)))
        # print(f"Action: {ccs[-1]-1}, Score: {score}")
        if done:
            print("GD Optimizer Exits ...")
            break

        if values[-1] < least_cost:
            least_cost = values[-1]
            soft_limit = min(ccs[-1]+4, max_action)

        next_action = min((ccs[-1] + 1),max_action)
        state, score, done, _ = transferEnvironment.step(next_action)
        # values.append(np.round(score * (-1)))
        values.append(np.round(score * (1)))
        # print(f"Action: {next_action}, Score: {score}")

        if done:
            print("GD Optimizer Exits ...")
            break

        if values[-1] < least_cost:
            least_cost = values[-1]
            soft_limit = min(ccs[-1]+4, max_action)

        count += 2

        gradient = (values[-1] - values[-2])/2
        update_cc = np.ceil(update_rate * gradient)
        next_cc = min(max(ccs[-1] + update_cc, 2), max_action)
        # print("Gradient: {0}, Previous Action: {1}, Chosen Action: {2}".format(gradient, ccs[-1], next_cc))
        ccs.append(int(next_cc))
        print("Gradient: {0}, Previous Action: {1}, Chosen Action: {2}".format(gradient, ccs[-2], ccs[-1]))
    return ccs

def bayes_optimizer(transferEnvironment):
  limit_obs, count = 25, 0
  max_thread = transferEnvironment.action_space.n
  iterations = -1
  search_space  = [
            Integer(1, max_thread),
        ]
  params = []
  optimizer = Optimizer(
      dimensions=search_space,
      base_estimator="GP", #[GP, RF, ET, GBRT],
      acq_func="gp_hedge", # [LCB, EI, PI, gp_hedge]
      acq_optimizer="auto", #[sampling, lbfgs, auto]
      n_random_starts=3,
      model_queue_size= limit_obs,
  )
  while True:
      count +=1
      if len(optimizer.yi) > limit_obs:
          optimizer.yi = optimizer.yi[-limit_obs:]
          optimizer.Xi = optimizer.Xi[-limit_obs:]


      print("Iteration {0} Starts ...".format(count))

      t1 = time.time()
      res = optimizer.run(func=transferEnvironment.bayes_step, n_iter=1)
      t2 = time.time()


      print("Iteration {0} Ends, Took {3} Seconds. Best Params: {1} and Score: {2}.".format(
              count, res.x, res.fun, np.round(t2-t1, 2)))

      last_value = optimizer.yi[-1]
      if last_value == -1000000:
          print("Bayseian Optimizer Exits ...")
          break
    #   print("Last Value: ", last_value)
    #   print("Last Action: ", optimizer.Xi[-1][0])
      cc = optimizer.Xi[-1][0]
      if iterations < 1:
          reset = False
          if (last_value > 0) and (cc < max_thread):
              max_thread = max(cc, 2)
              reset = True

          if (last_value < 0) and (cc == max_thread):
              max_thread = min(cc+1,max_thread)
              reset = True

          if reset:
              search_space[0] = Integer(1, max_thread)
              optimizer = Optimizer(
                  dimensions=search_space,
                  n_initial_points=3,
                  acq_optimizer="lbfgs",
                  model_queue_size= limit_obs
              )

      if iterations == count:
          print("Best parameters: {0} and score: {1}".format(res.x, res.fun))
          params = res.x
          break

  return params

def maximize(transferEnvironment):
    max_action, count = transferEnvironment.action_space.n, 0
    print(f"Max Action: {max_action}")
    params = []
    while True:
        state, score, done, _ = transferEnvironment.step(max_action-1)
        params.append(max_action-1)
        if done:
            print("Maximizer Exits ...")
            break
    return params

def minimize(transferEnvironment):
    max_action, count = transferEnvironment.action_space.n, 0
    print(f"Maximum Action: {max_action}")
    params = []
    while True:
        state, score, done, _ = transferEnvironment.step(1)
        params.append(1)
        if done:
            print("Minimizer Exits ...")
            break
    return params

def static(transferEnvironment, action):
    max_action, count = transferEnvironment.action_space.n, 0
    print(f"Static Action: {action}")
    params = []
    while True:
        state, score, done, _ = transferEnvironment.step(action)
        params.append(action)
        if done:
            print("Static Exits ...")
            break
    return params


def gradient_descent_optimizer(env, iterations=100, learning_rate=0.1, exploration_rate=0.1):
    max_action, count = 9, 0
    current_action = env.action_space.sample()  # Start with a random action
    last_reward = None
    last_action = None
    reward_list=[]
    action_list=[]
    print("gradient_descent_optimizer Starting ...")
    while True:
        # Take the current action and observe the reward
        action_list.append(current_action)
        _, reward, done, _ = env.step(int(current_action))
        reward_list.append(reward)
        if done:
            print("Gradient Descent Optimizer Exits ...")
            break   # Exit if the environment says we're done
        if last_action is not None:
            # Calculate the approximate gradient based on the last action and reward
            gradient = reward - last_reward
            # Exploitation: Adjust action based on the observed gradient
            action_change = -np.sign(gradient)
            next_action = (last_action + learning_rate * action_change)
            # Make sure the action is within the range of the action space
            next_action =min(max_action-1, max(0, next_action))
        else:
            # For the first iteration, just choose a random next action
            next_action = env.action_space.sample()

        # Update for the next iteration
        last_reward = reward
        last_action = current_action
        current_action = next_action

        print(f"Current Action: {current_action}, Reward: {reward}")
    return (action_list, reward_list)
