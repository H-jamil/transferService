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
        next_cc = (ccs[-1] + update_cc) % max_action
        print("Gradient: {0}, Gradient Change: {1}, Theta: {2}, Previous Action: {3}, Chosen Action: {4}".format(gradient, gradient_change, theta, ccs[-1], next_cc))
        ccs.append(next_cc)
    print("Total Actions: ", count)
    return ccs



# def bayes_optimizer(transferEnvironment):
#     limit_obs, count = 25, 0
#     max_thread = transferEnvironment.action_space.n
#     iterations = -1
#     last_score = None

#     # Define objective function for BayesianOptimization
#     def objective_function(x):
#         nonlocal last_score  # Declare last_score as nonlocal to modify its value inside this function
#         result = transferEnvironment.bayes_step([x])
#         last_score = result  # Store the last score
#         return last_score     # We are maximizing in this library by default

#     optimizer = BayesianOptimization(
#         f=objective_function,
#         pbounds={'x': (1, max_thread)},
#         verbose=2,
#         random_state=1,
#     )

#     while True:
#         count += 1

#         print(f"Iteration {count} Starts ...")

#         if count == 1:
#             # Maybe do more random exploration in the beginning
#             optimizer.maximize(init_points=5, n_iter=0)
#         else:
#             optimizer.maximize(init_points=1, n_iter=1)

#         best_params = optimizer.max['params']['x']
#         best_score = -optimizer.max['target']  # remember to negate the result

#         print(f"Iteration {count} Ends. Best Params: {best_params} and Score: {best_score}.")

#         if last_score == 1000000:
#             print("Bayesian Optimizer Exits ...")
#             break

#         cc = best_params
#         if (best_score > 0) and (cc < max_thread):
#             max_thread = max(cc, 2)

#         if iterations == count:
#             print(f"Best parameters: {best_params} and score: {best_score}")
#             break

#     return best_params

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

