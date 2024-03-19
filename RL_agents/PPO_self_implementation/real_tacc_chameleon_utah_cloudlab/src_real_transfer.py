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
import subprocess
import threading
import multiprocessing as mp
from multiprocessing import Value, Manager, Process
import psutil
import socket
import logging as log
from collections import OrderedDict

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

class WatchdogTimer:
    def __init__(self, timeout, user_handler=None):
        self.timeout = timeout
        self.handler = user_handler if user_handler is not None else self.default_handler
        self.timer = threading.Timer(self.timeout, self.handler)

    def reset(self):
        self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self.handler)
        self.timer.start()

    def stop(self):
        self.timer.cancel()

    def default_handler(self):
        raise self

def tcp_stats(REMOTE_IP):
    start = time.time()
    sent, retm = 0, 0
    try:
        data = os.popen("ss -ti").read().split("\n")
        for i in range(1,len(data)):
            if REMOTE_IP in data[i-1]:
                parse_data = data[i].split(" ")
                for entry in parse_data:
                    if "data_segs_out" in entry:
                        sent += int(entry.split(":")[-1])
                    if "bytes_retrans" in entry:
                        pass

                    elif "retrans" in entry:
                        retm += int(entry.split("/")[-1])

    except Exception as e:
        print(e)
    end = time.time()
    # print("Time taken to collect tcp stats: {0}ms".format(np.round((end-start)*1000)))
    return sent, retm

class transferService:
    def __init__(self, REMOTE_IP, REMOTE_PORT, INTERVAL, INTERFACE, SERVER_IP, SERVER_PORT,OPTIMIZER,log):
      self.log=log
      self.REMOTE_IP = REMOTE_IP
      self.REMOTE_PORT = REMOTE_PORT
      self.INTERVAL = INTERVAL
      self.INTERFACE = INTERFACE
      self.SERVER_IP = SERVER_IP
      self.SERVER_PORT = SERVER_PORT
      self.speed_mbps = Value('d', 0.0)
      self.rtt_ms = Value('d', 0.0)
      self.c_parallelism = Value('i', 1)
      self.c_concurrency = Value('i', 1)
      self.r_parallelism = Value('i', 1)
      self.r_concurrency = Value('i', 1)
      self.c_energy = Value('d', 0.0)
      self.manager = Manager()
      self.throughput_logs = self.manager.list()
      self.B = 10
      self.K = 1.02
      self.packet_loss_rate = Value('d', 0.0)
      self.OPTIMIZER=OPTIMIZER
      self.runtime_status=Value('i', 0)
      self.monitor_thread = None
      self.run_program_thread = None
      self.speed_calculator = None
      self.rtt_calculator_process = None
      self.loss_rate_client_process = None

    def calculate_download_speed(self):
      sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      # Bind the socket to the address and port
      server_address = ('127.0.0.1', 8081)
      sock.bind(server_address)
      print(f"Listening for messages from 8081 {server_address}")

      while True:
          # Receive data sent from the server
          try:
              data, address = sock.recvfrom(4096)
              received_message = data.decode('utf-8')  # renamed to received_message
          except Exception as e:  # catching general exception and printing it might be more informative
              print(f"Failed to receive data from server. Error: {e}")
              continue

          # Extract throughput value
          try:
            parts = received_message.split()
            # Extract and convert the values
            throughput = float(parts[1])
            energy = int(parts[4])
            parallelism = int(parts[6])
            concurrency = int(parts[8])
            throughput_value = received_message.split(":")[1].strip().split(" ")[0]
            self.speed_mbps.value = throughput
            self.c_energy.value = energy
            self.r_parallelism.value=parallelism
            self.r_concurrency.value=concurrency
          except Exception as e:
              print(f"Failed to extract throughput value from received message. Error: {e}")

    def rtt_calculator(self):
      while True:
        time.sleep(1)
        try:
            # Use ping command with 1 packet and extract the rtt
            output = subprocess.check_output(['ping', '-c', '1', self.REMOTE_IP]).decode('utf-8')
            # Extract the RTT from the output
            rtt_line = [line for line in output.split('\n') if 'time=' in line][0]
            rtt = rtt_line.split('time=')[-1].split(' ms')[0]
            self.rtt_ms.value = float(rtt)
        # except subprocess.CalledProcessError:
        except Exception as e:
          print(f"Failed to ping {self.REMOTE_IP}. Error: {e}")
          # print(f"Failed to ping {self.REMOTE_IP}")
          self.rtt_ms.value = 0.0

    def loss_rate_client(self):
      server_address = (self.REMOTE_IP, 9081)
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
          s.connect(server_address)
          while True:
            data = s.recv(1024)
            message = data.decode()
            try:
              lr_str = message.split("lossRate: ")[1].strip()
              lr = float(lr_str)
              # print("Sender Loss Rate:", lr)
              self.packet_loss_rate.value = lr
            except (IndexError, ValueError) as e:
                print("Error extracting loss rate:", e)
                self.packet_loss_rate.value = 0.0
        except Exception as e:
            print("Error connecting to server:", e)
            self.packet_loss_rate.value = 0.0

    def monitor_process(self):
      prev_sc,prev_rc=0,0
      while True:
        t1 = time.time()
        curr_thrpt = self.speed_mbps.value
        network_rtt=self.rtt_ms.value
        energy_consumed=self.c_energy.value
        curr_parallelism=self.r_parallelism.value
        curr_concurrency=self.r_concurrency.value
        loss_rate = self.packet_loss_rate.value
        cc_level=curr_parallelism*curr_concurrency
        record_list=[]
        # curr_sc,curr_rc=tcp_stats(self.REMOTE_IP)
        curr_sc,curr_rc=0,0
        record_list.append(curr_thrpt)
        record_list.append(curr_parallelism)
        record_list.append(curr_concurrency)
        sc, rc = curr_sc - prev_sc, curr_rc - prev_rc
        lr= 0
        if sc != 0:
          lr = rc/sc if sc>rc else 0
        if lr < 0:
          lr=0
        # plr_impact = self.B*lr
        plr_impact = self.B*loss_rate
        cc_impact_nl = self.K**cc_level
        score = (curr_thrpt/cc_impact_nl) - (curr_thrpt * plr_impact)
        score_value =np.round(score * (1))
        prev_sc,prev_rc=curr_sc,curr_rc
        record_list.append(lr)
        record_list.append(score_value)
        record_list.append(network_rtt)
        record_list.append(energy_consumed)
        record_list.append(loss_rate)
        record_list.append(datetime.now())
        self.throughput_logs.append(record_list)
        self.log.info("Throughput @{0}s:   {1}Gbps, lossRate: {2} parallelism:{3}  concurrency:{4} score:{5}  rtt:{6} ms energy:{7} Jules s-plr:{8} ".format(
            time.time(), curr_thrpt,lr,curr_parallelism,curr_concurrency,score_value,network_rtt,energy_consumed,loss_rate))
        t2 = time.time()
        time.sleep(max(0, 1 - (t2-t1)))

    def run_transfer_parameter_client(self):
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      while True:
        try:
          s.connect((self.SERVER_IP, self.SERVER_PORT))
          break

        except socket.error as e:
          print(f"Failed to connect: {e}")
          time.sleep(1)

      while True:
          print(f"Sending values: {self.c_parallelism.value} {self.c_concurrency.value}")

          try:
            s.send(f"{self.c_parallelism.value} {self.c_concurrency.value}".encode('utf-8'))
            raw_response = s.recv(1024)
            response = raw_response.decode().strip()
            if response.startswith("TERMINATE"):
              print("Received termination signal. Exiting...")
              break
            raw_response = s.recv(1024)
            response = raw_response.decode().strip()
            print(f"Received response: {response}")
            throughput_string = response.split(",")[0].split(":")[1].strip()
            energy_used_string = response.split(",")[1].split(":")[1].strip()
            throughput, energy_used = map(float, [throughput_string, energy_used_string])
          except Exception as e:
            print(f"Failed to decode response: {e}")
            break

      s.close()

    def run_programs(self):
      current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
      file_name = f"./{self.OPTIMIZER}_time_{current_time}.log"
      log_file_path = f"./logFileDir/{self.OPTIMIZER}_time_{current_time}.log"
      # Watchdog handler to kill the process if it hangs
      def kill_process(p):
        try:
          p.kill()
          print(f"Process {p.pid} was killed due to timeout!")
          self.runtime_status.value=1
        except ProcessLookupError:
            # Process might have already finished
          pass
      with open(file_name, 'a') as file:
        process = subprocess.Popen(["./parallel_concurrent", self.REMOTE_IP, log_file_path],
                                    stdout=file, stderr=file)
        # Start the watchdog timer
        watchdog = WatchdogTimer(500, lambda: kill_process(process))
        watchdog.reset()
        client_process = Process(target=self.run_transfer_parameter_client)
        client_process.start()
        client_process.join()
        # Stop the watchdog timer
        watchdog.stop()
      # time.sleep(3)
      os.system("pkill -f parallel_concurrent")
      self.runtime_status.value=1
      files_to_remove = [f for f in os.listdir() if f.startswith("FILE")]
      for f in files_to_remove:
          os.remove(f)

    def reset(self):
      if self.speed_calculator and self.speed_calculator.is_alive():
          self.speed_calculator.terminate()
          self.speed_calculator.join()
      if self.rtt_calculator_process and self.rtt_calculator_process.is_alive():
          self.rtt_calculator_process.terminate()
          self.rtt_calculator_process.join()
      if self.loss_rate_client_process and self.loss_rate_client_process.is_alive():
          self.loss_rate_client_process.terminate()
          self.loss_rate_client_process.join()
      if self.monitor_thread and self.monitor_thread.is_alive():
          self.monitor_thread.terminate()
          self.monitor_thread.join()
      if self.run_program_thread and self.run_program_thread.is_alive():
          self.run_program_thread.terminate()
          self.run_program_thread.join()
      os.system("pkill -f parallel_concurrent")
      files_to_remove = [f for f in os.listdir() if f.startswith("FILE")]
      for f in files_to_remove:
          os.remove(f)
      self.throughput_logs[:] = []
      self.speed_mbps.value = 0.0
      self.rtt_ms.value = 0.0
      self.c_parallelism.value = 1
      self.c_concurrency.value = 1
      self.c_energy.value = 0.0
      self.runtime_status.value = 0
      self.packet_loss_rate.value = 0.0
      self.speed_calculator = Process(target=self.calculate_download_speed)
      self.speed_calculator.start()
      self.rtt_calculator_process = Process(target=self.rtt_calculator)
      self.rtt_calculator_process.start()
      self.loss_rate_client_process = Process(target=self.loss_rate_client)
      self.loss_rate_client_process.start()
      self.monitor_thread = Process(target=self.monitor_process)
      self.run_program_thread = Process(target=self.run_programs)
      self.monitor_thread.start()
      self.run_program_thread.start()
      return np.zeros(40,)

    def step(self, parallelism,concurrency):
      if self.runtime_status.value==0:
        self.c_parallelism.value = parallelism
        self.c_concurrency.value = concurrency
        timer6s=time.time()
        while timer6s + 10 > time.time():
          pass
        last_ten = copy.deepcopy(self.throughput_logs[-10:])  # Get the last 10 elements for total_score
        last_five = last_ten[-5:]  # Get the last 5 elements from the last_ten list for concatenated_values
        concatenated_values = []
        score_list = []
        # total_score = 0
        # Calculate total score from last 10 elements
        for entry in last_ten:
            score_list.append(entry[4])  #  index 4 is for score
        total_score = np.sum(score_list)
        for entry in last_five:
            concatenated_values.extend(entry[:-1])  # Exclude the last value (time)
        # Create a numpy array from the concatenated values
        result_array = np.array(concatenated_values)
        # print("From transferService.step() Total Score: ", total_score)
        # Return result array and total score
        return result_array, total_score
      else:
        return np.zeros(40,),1000000

    def cleanup(self):
      os.system("pkill -f parallel_concurrent")
      if self.speed_calculator and self.speed_calculator.is_alive():
          self.speed_calculator.terminate()
          self.speed_calculator.join()

      if self.rtt_calculator_process and self.rtt_calculator_process.is_alive():
          self.rtt_calculator_process.terminate()
          self.rtt_calculator_process.join()

      if self.loss_rate_client_process and self.loss_rate_client_process.is_alive():
          self.loss_rate_client_process.terminate()
          self.loss_rate_client_process.join()
      if self.monitor_thread and self.monitor_thread.is_alive():
          self.monitor_thread.terminate()
          self.monitor_thread.join()

      if self.run_program_thread and self.run_program_thread.is_alive():
          self.run_program_thread.terminate()
          self.run_program_thread.join()

      self.throughput_logs[:] = []

      # Reset shared variables to default
      self.speed_mbps.value = 0.0
      self.rtt_ms.value = 0.0
      self.c_parallelism.value = 1
      self.c_concurrency.value = 1
      self.c_energy.value = 0.0
      files_to_remove = [f for f in os.listdir() if f.startswith("FILE")]
      for f in files_to_remove:
          os.remove(f)
      self.runtime_status.value=1
      # Print a message indicating cleanup is done
      print("Cleanup completed!")
      return 0

class transferClassReal_MA_ID(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,transferServiceObject,optimizer,min_values=[0.00, 0.0, 1, 1, -3081.0, 0.0, 0.0, 0.0],max_values = [19.2, 2.0, 8, 8, 16.0, 70.1, 120.0, 74.166]):
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
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
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
             reward=reward_
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

def get_env(env_string='transferService', optimizer='multiA_Inc_Dec_trained', REMOTE_IP = "128.110.219.183", REMOTE_PORT = "80", INTERVAL = 1,INTERFACE = "eno1",SERVER_IP = '127.0.0.1',SERVER_PORT = 8080):
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
    reordered_col_names = ['Throughput', 'receiver_lr', 'concurrency', 'parallelism','Score', 'RTT', 'Energy', 'sender_lr']
    df = df[reordered_col_names]

    return df


def normalize_and_flatten_real(df, min_values, max_values):
    # Drop the specified columns
    score_array = df['Score'].values
    energy_array=df['Energy'].values
    throughput_array=df['Throughput'].values
    # Normalize each column
    normalized_df = (df - min_values) / (max_values - min_values)
    # Flatten the DataFrame to a single NumPy array
    flattened_array = normalized_df.values.flatten()

    return flattened_array,score_array,energy_array,throughput_array


class NormalizeObservationAndRewardWrapper(gym.Wrapper):
    def __init__(self, env, sla_type='score',min_values=[0.00, 0.0, 1, 1, -3081.0, 0.0, 0.0, 0.0],max_values = [19.2, 2.0, 8, 8, 16.0, 70.1, 120.0, 74.166], reward_scale=1.0):
        super().__init__(env)
        self.min_values=np.array(min_values)
        self.max_values=np.array(max_values)
        self.reward_scale = reward_scale
        self.old_score = 0
        self.sla_type = sla_type
        self.energy_sla=75
        self.throughput_sla=8

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return observation.astype(np.float32)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if reward != 1000000:
            # print(f"Observation (before normalization): {observation}, Reward: {reward}, Done: {done}, Info: {info}")
            normalized_obs, result_array ,e_array ,t_array = self.normalize_observation(observation)

            if self.sla_type == 'score':
                reward_ = np.sum(result_array)
                reward = reward_ - self.old_score
                self.old_score = reward_

            elif self.sla_type == 'energy':
              energy_part_reward=np.mean(e_array)
              if energy_part_reward > self.energy_sla:
                energy_penalty=energy_part_reward-self.energy_sla
              else:
                energy_penalty=0
              reward_=np.mean(t_array)- energy_penalty
              reward=reward_- self.old_score
              self.old_score=reward_

            elif self.sla_type == 'throughput':
              throughput_part_reward=np.mean(t_array)
              if throughput_part_reward < self.throughput_sla:
                throughput_penalty=(self.throughput_sla-throughput_part_reward)*5
              else:
                throughput_penalty=0
              reward_=throughput_part_reward-throughput_penalty
              reward=reward_- self.old_score
              self.old_score=reward_

            elif self.sla_type == 'energyEfficiency':
              energy_part_reward=np.max(e_array)
              throughput_part_reward=np.mean(t_array)
              reward_=(throughput_part_reward*10)/energy_part_reward
              reward=reward_-self.old_score
              self.old_score=reward_

            # print(f"Observation (after normalization): {normalized_obs}, Reward: {reward}, Done: {done}, Info: {info}")
            return normalized_obs, round(reward,3), done, info
        else:
            normalized_obs, result_array ,e_array ,t_array = self.normalize_observation(observation)
            return normalized_obs, 0 , done, info

    def normalize_observation(self, observation):
        observation_df=np_array_to_df_with_reordered_columns(observation)
        normalized_observation,result_array,e_array,t_array=normalize_and_flatten_real(observation_df,self.min_values,self.max_values)
        return normalized_observation.astype(np.float32),result_array,e_array,t_array

    def close(self):
        self.env.close()


class transferClassReal_GD(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,transferServiceObject,optimizer,min_values=[0.00, 0.0, 1, 1, -3081.0, 0.0, 0.0, 0.0],max_values = [19.2, 2.0, 8, 8, 16.0, 70.1, 120.0, 74.166]):
        super().__init__()
        self.transfer_service = transferServiceObject
        self.min_action=1
        self.max_action=8
        self.action_space = spaces.MultiDiscrete([8, 8])  # example action space
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
        self.current_action_parallelism_value=1
        self.current_action_concurrency_value=1
        return self.current_observation

    def step(self, action):
        action=np.array(action)+1
        # perform action using transfer_service
        action_1,action_2=action

        self.current_action_parallelism_value=action_1
        if self.current_action_parallelism_value<self.min_action:
            self.current_action_parallelism_value=self.min_action
        elif self.current_action_parallelism_value>self.max_action:
            self.current_action_parallelism_value=self.max_action

        self.current_action_concurrency_value=action_2
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

def get_env_gd(env_string='transferService', optimizer='multiA_Inc_Dec_trained', REMOTE_IP = "128.110.219.183", REMOTE_PORT = "80", INTERVAL = 1,INTERFACE = "eno1",SERVER_IP = '127.0.0.1',SERVER_PORT = 8080):
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
    env = transferClassReal_GD(transfer_service,optimizer)
    return env


def gradient_multivariate(max_io_cc, max_net_cc,env):
    count = 0
    cache_net = OrderedDict()
    cache_io = OrderedDict()
    io_opt = True
    values = []
    ccs = [[1,1]]
    cc_change_limit=5
    env.reset()
    while True:
        count += 1
        soft_limit_net = max_net_cc
        soft_limit_io = max_io_cc
        observation, reward, done, info = env.step(np.array(ccs[-1]))
        values.append(reward)
        cache_net[abs(values[-1])] = ccs[-1][0]
        cache_io[abs(values[-1])] = ccs[-1][1]
        if count % 5 == 0:
            soft_limit_net = min(cache_net[max(cache_net.keys())], max_net_cc)
            soft_limit_io = min(cache_io[max(cache_io.keys())], max_io_cc)

        if len(cache_net)>20 or len(cache_io)>20:
            cache_net.popitem(last=True)
            cache_io.popitem(last=True)

        if done == True:
            print("GD Optimizer Exits ...")
            break
        if len(ccs) == 1:
            ccs.append([2,2])
        else:
            # Network
            difference = ccs[-1][0] - ccs[-2][0]
            prev, curr = values[-2], values[-1]
            if difference != 0 and prev !=0:
                gradient = (curr - prev)/(difference*prev)
            else:
                gradient = (curr - prev)/prev if prev != 0 else 1

            update_cc_net = ccs[-1][0] * gradient
            # print(f"Gradient {gradient} at step {count}")
            if update_cc_net>0:
                update_cc_net = min(max(1, int(np.round(update_cc_net))), cc_change_limit)
            else:
                update_cc_net = max(min(-1, int(np.round(update_cc_net))), -cc_change_limit)

            next_cc_net = min(max(ccs[-1][0] + update_cc_net, 1), soft_limit_net)

            # I/O
            next_cc_io = max(1, ccs[-1][1] // 2)
            if io_opt:
                difference = ccs[-1][1] - ccs[-2][1]
                prev, curr = values[-2], values[-1]
                print((prev, curr))
                if curr != 0:
                    if difference != 0 and prev !=0:
                        gradient = (curr - prev)/(difference*prev)
                    else:
                        gradient = (curr - prev)/prev if prev !=0 else 1

                    update_cc_io = ccs[-1][1] * gradient
                    if update_cc_io>0:
                        update_cc_io = min(max(1, int(np.round(update_cc_io))), cc_change_limit)
                    else:
                        update_cc_io = max(min(-1, int(np.round(update_cc_io))), -cc_change_limit)

                    next_cc_io = min(max(ccs[-1][1] + update_cc_io, 1), soft_limit_io)
                    print((update_cc_io, next_cc_io, soft_limit_io))
            else:
                next_cc_io = 0

            ccs.append([next_cc_net, next_cc_io])
            print(f"Gradient: {gradient}")
            print(f"Previous CC: {ccs[-2]}, Choosen CC: {ccs[-1]}")
    env.close()
    return ccs, values
