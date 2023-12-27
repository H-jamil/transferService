import numpy as np
import random
import time
# from bayes_opt import BayesianOptimization
from skopt.space import Integer
from skopt import Optimizer, dummy_minimize
from scipy.optimize import minimize


def gradient_opt(transferEnvironment):
    max_action, count = transferEnvironment.action_space.n, 0
    least_cost = float('inf')
    values = []
    ccs = [1]  # starting action is chosen randomly
    theta = 0

    while True:
        state, score, done, _ = transferEnvironment.step(ccs[-1])
        values.append(score)
        if done:
            print("GD Optimizer Exits ...")
            break

        if values[-1] < least_cost:
            least_cost = values[-1]

        next_action = (ccs[-1] + 1) % max_action
        state, score, done, _ = transferEnvironment.step(next_action)
        values.append(score)
        if done:
            print("GD Optimizer Exits ...")
            break

        if values[-1] < least_cost:
            least_cost = values[-1]

        count += 2

        gradient = (values[-1] - values[-2])

        if np.abs(values[-2]) < 1e-9:  # Using a small threshold instead of checking for exact zero
            gradient_change = 0
        else:
            gradient_change = np.abs(gradient/values[-2])

        if gradient > 0:
            if theta <= 0:
                theta -= 1
            else:
                theta = -1
        else:
            if theta >= 0:
                theta += 1
            else:
                theta = 1

        update_cc = int(theta * gradient_change)
        # next_cc = (ccs[-1] + update_cc) % max_action
        offered_load = ccs[-1] + update_cc
        if offered_load < 0:
            offered_load = 1
        next_cc = min(offered_load, max_action-1)
        print("Gradient: {0}, Gradient Change: {1}, Theta: {2}, Previous Action: {3}, Chosen Action: {4}".format(gradient, gradient_change, theta, ccs[-1], next_cc))
        ccs.append(next_cc)
    print("Total Actions: ", count)
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
      if last_value == 1000000:
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
