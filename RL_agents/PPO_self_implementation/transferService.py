import subprocess
import threading
import time
import os
import numpy as np
# import pandas as pd
import multiprocessing as mp
from multiprocessing import Value, Manager, Process
import psutil
import socket
import copy
from datetime import datetime
import logging as log
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



# class transferService:
#     def __init__(self, REMOTE_IP, REMOTE_PORT, INTERVAL, INTERFACE, SERVER_IP, SERVER_PORT,OPTIMIZER,log):
#       self.log=log
#       self.REMOTE_IP = REMOTE_IP
#       self.REMOTE_PORT = REMOTE_PORT
#       self.INTERVAL = INTERVAL
#       self.INTERFACE = INTERFACE
#       self.SERVER_IP = SERVER_IP
#       self.SERVER_PORT = SERVER_PORT
#       self.speed_mbps = Value('d', 0.0)
#       self.rtt_ms = Value('d', 0.0)
#       self.c_parallelism = Value('i', 1)
#       self.c_concurrency = Value('i', 1)
#       self.c_energy = Value('d', 0.0)
#       self.manager = Manager()
#       self.throughput_logs = self.manager.list()
#       self.B = 10
#       self.K = 1.02
#       self.packet_loss_rate = Value('d', 0.0)
#       self.OPTIMIZER=OPTIMIZER
#       self.runtime_status=Value('i', 0)
#       self.monitor_thread = None
#       self.run_program_thread = None
#       self.speed_calculator = None
#       self.rtt_calculator_process = None
#       self.loss_rate_client_process = None

#     def calculate_download_speed(self):
#       sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#       # Bind the socket to the address and port
#       server_address = ('127.0.0.1', 8081)
#       sock.bind(server_address)
#       print(f"Listening for messages from 8081 {server_address}")

#       while True:
#           # Receive data sent from the server
#           try:
#               data, address = sock.recvfrom(4096)
#               received_message = data.decode('utf-8')  # renamed to received_message
#               # print(f"Received {received_message} from 8081 {address}")
#           except Exception as e:  # catching general exception and printing it might be more informative
#               print(f"Failed to receive data from server. Error: {e}")
#               continue

#           # Extract throughput value
#           try:
#               throughput_value = received_message.split(":")[1].strip().split(" ")[0]
#               # print(f"Throughput Value: {throughput_value}")
#               self.speed_mbps.value = float(throughput_value)  # converting to float
#           # except IndexError:
#           except Exception as e:
#               print(f"Failed to extract throughput value from received message. Error: {e}")
#               # print("Couldn't extract the throughput value from the received message.")

#     def rtt_calculator(self):
#       while True:
#         time.sleep(1)
#         try:
#             # Use ping command with 1 packet and extract the rtt
#             output = subprocess.check_output(['ping', '-c', '1', self.REMOTE_IP]).decode('utf-8')
#             # Extract the RTT from the output
#             rtt_line = [line for line in output.split('\n') if 'time=' in line][0]
#             rtt = rtt_line.split('time=')[-1].split(' ms')[0]
#             self.rtt_ms.value = float(rtt)
#         # except subprocess.CalledProcessError:
#         except Exception as e:
#           print(f"Failed to ping {self.REMOTE_IP}. Error: {e}")
#           # print(f"Failed to ping {self.REMOTE_IP}")
#           self.rtt_ms.value = 0.0

#     def loss_rate_client(self):
#       server_address = (self.REMOTE_IP, 9081)
#       with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         try:
#           s.connect(server_address)
#           print(f"Connected to {server_address}")
#           while True:
#             data = s.recv(1024)
#             message = data.decode()

#             try:
#               lr_str = message.split("lossRate: ")[1].strip()
#               lr = float(lr_str)
#               # print("Sender Loss Rate:", lr)
#               self.packet_loss_rate.value = lr
#             except (IndexError, ValueError) as e:
#                 print("Error extracting loss rate:", e)
#                 self.packet_loss_rate.value = 0.0
#         except Exception as e:
#             print("Error connecting to server:", e)
#             self.packet_loss_rate.value = 0.0

#     def monitor_process(self):
#       prev_sc,prev_rc=0,0
#       while True:
#         t1 = time.time()
#         curr_thrpt = self.speed_mbps.value
#         network_rtt=self.rtt_ms.value
#         energy_consumed=self.c_energy.value
#         # print(f"Monitoring Process : Download Speed: {curr_thrpt:.2f} Mbps")
#         curr_parallelism=self.c_parallelism.value
#         curr_concurrency=self.c_concurrency.value
#         loss_rate = self.packet_loss_rate.value
#         # cc_level=curr_parallelism*curr_concurrency
#         cc_level=curr_parallelism
#         record_list=[]
#         curr_sc,curr_rc=tcp_stats(self.REMOTE_IP)
#         record_list.append(curr_thrpt)
#         record_list.append(curr_parallelism)
#         record_list.append(curr_concurrency)
#         sc, rc = curr_sc - prev_sc, curr_rc - prev_rc
#         lr= 0
#         if sc != 0:
#           lr = rc/sc if sc>rc else 0
#         if lr < 0:
#           lr=0
#         # plr_impact = self.B*lr
#         plr_impact = self.B*loss_rate
#         cc_impact_nl = self.K**cc_level
#         score = (curr_thrpt/cc_impact_nl) - (curr_thrpt * plr_impact)
#         # score_value = np.round(score * (-1))
#         score_value =np.round(score * (1))
#         prev_sc,prev_rc=curr_sc,curr_rc
#         record_list.append(lr)
#         record_list.append(score_value)
#         record_list.append(network_rtt)
#         record_list.append(energy_consumed)
#         record_list.append(loss_rate)
#         record_list.append(datetime.now())
#         self.throughput_logs.append(record_list)
#         self.log.info("Throughput @{0}s:   {1}Gbps, lossRate: {2} CC:{3}  score:{4}  rtt:{5} ms energy:{6} Jules s-plr:{7} ".format(
#             time.time(), curr_thrpt,lr,cc_level*cc_level,score_value,network_rtt,energy_consumed,loss_rate))
#         t2 = time.time()
#         time.sleep(max(0, 1 - (t2-t1)))

#     def run_transfer_parameter_client(self):
#       s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#       while True:
#         try:
#           # print (SERVER_IP, SERVER_PORT)
#           s.connect((self.SERVER_IP, self.SERVER_PORT))
#           break

#         except socket.error as e:
#           print(f"Failed to connect: {e}")
#           time.sleep(1)

#       while True:
#           print(f"Sending values: {self.c_parallelism.value} {self.c_concurrency.value}")

#           try:
#             s.send(f"{self.c_parallelism.value} {self.c_concurrency.value}".encode('utf-8'))
#             raw_response = s.recv(1024)
#             response = raw_response.decode().strip()
#             if response.startswith("TERMINATE"):
#               print("Received termination signal. Exiting...")
#               break
#             raw_response = s.recv(1024)
#             response = raw_response.decode().strip()
#             print(f"Received response: {response}")
#             throughput_string = response.split(",")[0].split(":")[1].strip()
#             energy_used_string = response.split(",")[1].split(":")[1].strip()
#             throughput, energy_used = map(float, [throughput_string, energy_used_string])
#             self.c_energy.value=energy_used
#           except Exception as e:
#             print(f"Failed to decode response: {e}")
#             break

#       s.close()

#     def run_programs(self):
#       current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#       file_name = f"./{self.OPTIMIZER}_time_{current_time}.log"
#       log_file_path = f"./logFileDir/{self.OPTIMIZER}_time_{current_time}.log"
#       # Watchdog handler to kill the process if it hangs
#       def kill_process(p):
#         try:
#           p.kill()
#           print(f"Process {p.pid} was killed due to timeout!")
#           self.runtime_status.value=1
#         except ProcessLookupError:
#             # Process might have already finished
#           pass
#       with open(file_name, 'a') as file:
#         process = subprocess.Popen(["./parallel_concurrent", self.REMOTE_IP, log_file_path],
#                                     stdout=file, stderr=file)
#         # Start the watchdog timer
#         watchdog = WatchdogTimer(500, lambda: kill_process(process))
#         watchdog.reset()
#         client_process = Process(target=self.run_transfer_parameter_client)
#         client_process.start()
#         client_process.join()
#         # Stop the watchdog timer
#         watchdog.stop()
#       # time.sleep(3)
#       os.system("pkill -f parallel_concurrent")
#       self.runtime_status.value=1
#       files_to_remove = [f for f in os.listdir() if f.startswith("FILE")]
#       for f in files_to_remove:
#           os.remove(f)

#     def reset(self):
#       if self.speed_calculator and self.speed_calculator.is_alive():
#           self.speed_calculator.terminate()
#           self.speed_calculator.join()
#       if self.rtt_calculator_process and self.rtt_calculator_process.is_alive():
#           self.rtt_calculator_process.terminate()
#           self.rtt_calculator_process.join()
#       if self.loss_rate_client_process and self.loss_rate_client_process.is_alive():
#           self.loss_rate_client_process.terminate()
#           self.loss_rate_client_process.join()
#       # if self.energy_process and self.energy_process.is_alive():
#       #     self.energy_process.terminate()
#       #     self.energy_process.join()
#       if self.monitor_thread and self.monitor_thread.is_alive():
#           self.monitor_thread.terminate()
#           self.monitor_thread.join()
#       if self.run_program_thread and self.run_program_thread.is_alive():
#           self.run_program_thread.terminate()
#           self.run_program_thread.join()
#       # Kill the "parallel_concurrent" process externally
#       os.system("pkill -f parallel_concurrent")
#       # Remove files that start with "FILE"
#       files_to_remove = [f for f in os.listdir() if f.startswith("FILE")]
#       for f in files_to_remove:
#           os.remove(f)
#       # Clear the manager list (logs)
#       self.throughput_logs[:] = []
#       # Reset shared variables to default
#       self.speed_mbps.value = 0.0
#       self.rtt_ms.value = 0.0
#       self.c_parallelism.value = 1
#       self.c_concurrency.value = 1
#       self.c_energy.value = 0.0
#       self.runtime_status.value = 0
#       self.packet_loss_rate.value = 0.0
#       self.speed_calculator = Process(target=self.calculate_download_speed)
#       self.speed_calculator.start()
#       self.rtt_calculator_process = Process(target=self.rtt_calculator)
#       self.rtt_calculator_process.start()
#       self.loss_rate_client_process = Process(target=self.loss_rate_client)
#       self.loss_rate_client_process.start()

#       # self.energy_process = Process(target=self.get_energy_consumption)
#       # self.energy_process.start()
#       self.monitor_thread = Process(target=self.monitor_process)
#       self.run_program_thread = Process(target=self.run_programs)
#       self.monitor_thread.start()
#       self.run_program_thread.start()
#       return np.zeros(40,)
#       # return 0

#     def step(self, parallelism,concurrency):
#       if self.runtime_status.value==0:
#         self.c_parallelism.value = parallelism
#         self.c_concurrency.value = concurrency
#         timer6s=time.time()
#         while timer6s + 10 > time.time():
#           pass
#         last_ten = copy.deepcopy(self.throughput_logs[-10:])  # Get the last 10 elements for total_score
#         last_five = last_ten[-5:]  # Get the last 5 elements from the last_ten list for concatenated_values
#         concatenated_values = []
#         score_list = []
#         total_score = 0
#         # Calculate total score from last 10 elements
#         for entry in last_ten:
#             score_list.append(entry[4])  # Assuming index 4 is for score
#         # total_score = np.sum(score_list)
#         # total_score = np.max(score_list)
#         total_score = np.mean(score_list)
#         # total_score = np.min(score_list)
#         # Concatenate values from last 5 elements
#         for entry in last_five:
#             concatenated_values.extend(entry[:-1])  # Exclude the last value (time)
#         # Create a numpy array from the concatenated values
#         result_array = np.array(concatenated_values)
#         print("From transferService.step() Total Score: ", total_score)
#         # Return result array and total score
#         return result_array, total_score
#       else:
#         return np.zeros(40,),1000000

#     def cleanup(self):
#       os.system("pkill -f parallel_concurrent")
#       if self.speed_calculator and self.speed_calculator.is_alive():
#           self.speed_calculator.terminate()
#           self.speed_calculator.join()

#       if self.rtt_calculator_process and self.rtt_calculator_process.is_alive():
#           self.rtt_calculator_process.terminate()
#           self.rtt_calculator_process.join()

#       if self.loss_rate_client_process and self.loss_rate_client_process.is_alive():
#           self.loss_rate_client_process.terminate()
#           self.loss_rate_client_process.join()

#       # if self.energy_process and self.energy_process.is_alive():
#       #     self.energy_process.terminate()
#       #     self.energy_process.join()

#       if self.monitor_thread and self.monitor_thread.is_alive():
#           self.monitor_thread.terminate()
#           self.monitor_thread.join()

#       if self.run_program_thread and self.run_program_thread.is_alive():
#           self.run_program_thread.terminate()
#           self.run_program_thread.join()

#       # Clear the manager list (logs)
#       self.throughput_logs[:] = []

#       # Reset shared variables to default
#       self.speed_mbps.value = 0.0
#       self.rtt_ms.value = 0.0
#       self.c_parallelism.value = 1
#       self.c_concurrency.value = 1
#       self.c_energy.value = 0.0
#       files_to_remove = [f for f in os.listdir() if f.startswith("FILE")]
#       for f in files_to_remove:
#           os.remove(f)
#       self.runtime_status.value=1
#       # Print a message indicating cleanup is done
#       print("Cleanup completed!")
#       return 0


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
              # print(f"Received {received_message} from 8081 {address}")
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
            # print(f"Throughput Value: {throughput_value}")
            # self.speed_mbps.value = float(throughput_value)  # converting to float
            self.speed_mbps.value = throughput
            self.c_energy.value = energy
            self.r_parallelism.value=parallelism
            self.r_concurrency.value=concurrency
          # except IndexError:
          except Exception as e:
              print(f"Failed to extract throughput value from received message. Error: {e}")
              # print("Couldn't extract the throughput value from the received message.")

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
        # print(f"Monitoring Process : Download Speed: {curr_thrpt:.2f} Mbps")
        curr_parallelism=self.r_parallelism.value
        curr_concurrency=self.r_concurrency.value
        loss_rate = self.packet_loss_rate.value
        cc_level=curr_parallelism*curr_concurrency
        # cc_level=curr_parallelism
        record_list=[]
        curr_sc,curr_rc=tcp_stats(self.REMOTE_IP)
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
        # score_value = np.round(score * (-1))
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
          # print (SERVER_IP, SERVER_PORT)
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
            # self.c_energy.value=energy_used
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
      # if self.energy_process and self.energy_process.is_alive():
      #     self.energy_process.terminate()
      #     self.energy_process.join()
      if self.monitor_thread and self.monitor_thread.is_alive():
          self.monitor_thread.terminate()
          self.monitor_thread.join()
      if self.run_program_thread and self.run_program_thread.is_alive():
          self.run_program_thread.terminate()
          self.run_program_thread.join()
      # Kill the "parallel_concurrent" process externally
      os.system("pkill -f parallel_concurrent")
      # Remove files that start with "FILE"
      files_to_remove = [f for f in os.listdir() if f.startswith("FILE")]
      for f in files_to_remove:
          os.remove(f)
      # Clear the manager list (logs)
      self.throughput_logs[:] = []
      # Reset shared variables to default
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

      # self.energy_process = Process(target=self.get_energy_consumption)
      # self.energy_process.start()
      self.monitor_thread = Process(target=self.monitor_process)
      self.run_program_thread = Process(target=self.run_programs)
      self.monitor_thread.start()
      self.run_program_thread.start()
      return np.zeros(40,)
      # return 0

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
        total_score = 0
        # Calculate total score from last 10 elements
        for entry in last_ten:
            score_list.append(entry[4])  # Assuming index 4 is for score
        # total_score = np.sum(score_list)
        # total_score = np.max(score_list)
        total_score = np.mean(score_list)
        # total_score = np.min(score_list)
        # Concatenate values from last 5 elements
        for entry in last_five:
            concatenated_values.extend(entry[:-1])  # Exclude the last value (time)
        # Create a numpy array from the concatenated values
        result_array = np.array(concatenated_values)
        print("From transferService.step() Total Score: ", total_score)
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

      # if self.energy_process and self.energy_process.is_alive():
      #     self.energy_process.terminate()
      #     self.energy_process.join()

      if self.monitor_thread and self.monitor_thread.is_alive():
          self.monitor_thread.terminate()
          self.monitor_thread.join()

      if self.run_program_thread and self.run_program_thread.is_alive():
          self.run_program_thread.terminate()
          self.run_program_thread.join()

      # Clear the manager list (logs)
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


class transferService_total_score:
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
              # print(f"Received {received_message} from 8081 {address}")
          except Exception as e:  # catching general exception and printing it might be more informative
              print(f"Failed to receive data from server. Error: {e}")
              continue

          # Extract throughput value
          try:
              throughput_value = received_message.split(":")[1].strip().split(" ")[0]
              # print(f"Throughput Value: {throughput_value}")
              self.speed_mbps.value = float(throughput_value)  # converting to float
          # except IndexError:
          except Exception as e:
              print(f"Failed to extract throughput value from received message. Error: {e}")
              # print("Couldn't extract the throughput value from the received message.")

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
        # print(f"Monitoring Process : Download Speed: {curr_thrpt:.2f} Mbps")
        curr_parallelism=self.c_parallelism.value
        curr_concurrency=self.c_concurrency.value
        loss_rate = self.packet_loss_rate.value
        # cc_level=curr_parallelism*curr_concurrency
        cc_level=curr_parallelism
        record_list=[]
        curr_sc,curr_rc=tcp_stats(self.REMOTE_IP)
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
        # score_value = np.round(score * (-1))
        score_value =np.round(score * (1))
        prev_sc,prev_rc=curr_sc,curr_rc
        record_list.append(lr)
        record_list.append(score_value)
        record_list.append(network_rtt)
        record_list.append(energy_consumed)
        record_list.append(loss_rate)
        record_list.append(datetime.now())
        self.throughput_logs.append(record_list)
        self.log.info("Throughput @{0}s:   {1}Gbps, lossRate: {2} CC:{3}  score:{4}  rtt:{5} ms energy:{6} Jules s-plr:{7} ".format(
            time.time(), curr_thrpt,lr,cc_level*cc_level,score_value,network_rtt,energy_consumed,loss_rate))
        t2 = time.time()
        time.sleep(max(0, 1 - (t2-t1)))

    def run_transfer_parameter_client(self):
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      while True:
        try:
          # print (SERVER_IP, SERVER_PORT)
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
            self.c_energy.value=energy_used
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
      # if self.energy_process and self.energy_process.is_alive():
      #     self.energy_process.terminate()
      #     self.energy_process.join()
      if self.monitor_thread and self.monitor_thread.is_alive():
          self.monitor_thread.terminate()
          self.monitor_thread.join()
      if self.run_program_thread and self.run_program_thread.is_alive():
          self.run_program_thread.terminate()
          self.run_program_thread.join()
      # Kill the "parallel_concurrent" process externally
      os.system("pkill -f parallel_concurrent")
      # Remove files that start with "FILE"
      files_to_remove = [f for f in os.listdir() if f.startswith("FILE")]
      for f in files_to_remove:
          os.remove(f)
      # Clear the manager list (logs)
      self.throughput_logs[:] = []
      # Reset shared variables to default
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

      # self.energy_process = Process(target=self.get_energy_consumption)
      # self.energy_process.start()
      self.monitor_thread = Process(target=self.monitor_process)
      self.run_program_thread = Process(target=self.run_programs)
      self.monitor_thread.start()
      self.run_program_thread.start()
      return np.zeros(40,)
      # return 0

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
        total_score = 0
        # Calculate total score from last 10 elements
        for entry in last_ten:
            score_list.append(entry[4])  # Assuming index 4 is for score
        total_score = np.sum(score_list)
        # total_score = np.max(score_list)
        # Concatenate values from last 5 elements
        for entry in last_five:
            concatenated_values.extend(entry[:-1])  # Exclude the last value (time)
        # Create a numpy array from the concatenated values
        result_array = np.array(concatenated_values)
        print("From transferService.step() Total Score: ", total_score)
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

      # if self.energy_process and self.energy_process.is_alive():
      #     self.energy_process.terminate()
      #     self.energy_process.join()

      if self.monitor_thread and self.monitor_thread.is_alive():
          self.monitor_thread.terminate()
          self.monitor_thread.join()

      if self.run_program_thread and self.run_program_thread.is_alive():
          self.run_program_thread.terminate()
          self.run_program_thread.join()

      # Clear the manager list (logs)
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




if __name__ == "__main__":
  log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
  log_file = "logFileDir/" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
  log.basicConfig(
      format=log_FORMAT,
      datefmt='%m/%d/%Y %I:%M:%S %p',
      level=log.INFO,
      handlers=[
          log.FileHandler(log_file),
          # log.StreamHandler()
      ]
  )

  # REMOTE_IP = "192.5.87.228"
  REMOTE_IP = "128.205.222.176"
  REMOTE_PORT = "80"
  INTERVAL = 1
  # INTERFACE="eno1np0"
  INTERFACE="enp3s0"
  # SERVER_IP = '10.52.1.91'
  SERVER_IP = '127.0.0.1'
  SERVER_PORT = 8080
  OPTIMIZER="random_test"
  transfer_service=transferService(REMOTE_IP,REMOTE_PORT,INTERVAL,INTERFACE,SERVER_IP,SERVER_PORT,OPTIMIZER,log)
  transfer_service.reset()
  transfer_s=0
  while transfer_s != 1000000:
    _,transfer_s=transfer_service.step(1,1)
    _,transfer_s=transfer_service.step(3,3)
    _,transfer_s=transfer_service.step(8,8)

  transfer_service.cleanup()

