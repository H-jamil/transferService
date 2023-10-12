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
      self.c_energy = Value('d', 0.0)
      self.manager = Manager()
      self.throughput_logs = self.manager.list()
      self.B = 10
      self.K = 1.02
      self.OPTIMIZER=OPTIMIZER
      self.runtime_status=Value('i', 0)
      self.monitor_thread = None
      self.run_program_thread = None
      self.speed_calculator = None
      self.rtt_calculator_process = None
      self.energy_process = None



    def calculate_download_speed(self):
      prev_rx = psutil.net_io_counters(pernic=True)[self.INTERFACE].bytes_recv
      while True:
          time.sleep(1)
          current_rx = psutil.net_io_counters(pernic=True)[self.INTERFACE].bytes_recv
          speed_bps = current_rx - prev_rx
          self.speed_mbps.value = (speed_bps * 8) / 1_000_000  # 8 bits per byte and 1e6 bits in a Mbps
          # print(f"Download Speed: {speed_mbps:.2f} Mbps")
          prev_rx = current_rx

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
        except subprocess.CalledProcessError:
            print(f"Failed to ping {self.REMOTE_IP}")
            self.rtt_ms.value = 0.0

    def get_energy_consumption(self):
      while True:
        try:
          with open("/sys/class/powercap/intel-rapl:0/energy_uj", "r") as file:
            energy_old = float(file.readline().strip())
        except IOError as e:
          print(f"Failed to open energy_uj file: {e}")
          return
        time.sleep(1)
        try:
          with open("/sys/class/powercap/intel-rapl:0/energy_uj", "r") as file:
            energy_now = float(file.readline().strip())
        except IOError as e:
          print(f"Failed to open energy_uj file: {e}")
          return
        energy_consumed = (energy_now - energy_old) / 1e6
        self.c_energy.value = energy_consumed

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
        cc_level=curr_parallelism*curr_concurrency
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
        plr_impact = self.B*lr
        cc_impact_nl = self.K**cc_level
        score = (curr_thrpt/cc_impact_nl) - (curr_thrpt * plr_impact)
        score_value = np.round(score * (-1))
        # score_value =np.round(score * (1))
        prev_sc,prev_rc=curr_sc,curr_rc
        record_list.append(lr)
        record_list.append(score_value)
        record_list.append(network_rtt)
        record_list.append(energy_consumed)
        record_list.append(datetime.now())
        self.throughput_logs.append(record_list)
        self.log.info("Throughput @{0}s:   {1}Mbps, lossRate: {2} CC:{3}  score:{4}  rtt:{5} ms energy:{6} Jules".format(
            time.time(), curr_thrpt,lr,cc_level,score_value,network_rtt,energy_consumed))
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
          except socket.error as e:
            print(f"Failed to communicate with server: {e}")
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
        watchdog = WatchdogTimer(1000, lambda: kill_process(process))
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
      if self.energy_process and self.energy_process.is_alive():
          self.energy_process.terminate()
          self.energy_process.join()
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
      self.speed_calculator = Process(target=self.calculate_download_speed)
      self.speed_calculator.start()
      self.rtt_calculator_process = Process(target=self.rtt_calculator)
      self.rtt_calculator_process.start()
      self.energy_process = Process(target=self.get_energy_consumption)
      self.energy_process.start()
      self.monitor_thread = Process(target=self.monitor_process)
      self.run_program_thread = Process(target=self.run_programs)
      self.monitor_thread.start()
      self.run_program_thread.start()
      return np.zeros(35,)
      # return 0

    def step(self, parallelism,concurrency):
      if self.runtime_status.value==0:
        self.c_parallelism.value = parallelism
        self.c_concurrency.value = concurrency
        timer6s=time.time()
        while timer6s + 6 > time.time():
          pass
        last_five = copy.deepcopy(self.throughput_logs[-5:])
        concatenated_values = []
        total_score = 0
        for entry in last_five:
            # Exclude the last value (time)
            concatenated_values.extend(entry[:-1])
            total_score += entry[4] # Index 4 for score_value
        # Step 4: Create a numpy array from the concatenated values
        result_array = np.array(concatenated_values)
        # Step 5: Compute the average of the 5 score values
        avg_score = total_score / 5.0
        return result_array, avg_score
      else:
        return np.zeros(35,),1000000

    def cleanup(self):
      os.system("pkill -f parallel_concurrent")
      if self.speed_calculator and self.speed_calculator.is_alive():
          self.speed_calculator.terminate()
          self.speed_calculator.join()

      if self.rtt_calculator_process and self.rtt_calculator_process.is_alive():
          self.rtt_calculator_process.terminate()
          self.rtt_calculator_process.join()

      if self.energy_process and self.energy_process.is_alive():
          self.energy_process.terminate()
          self.energy_process.join()

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

if __name__ == "__main__":
  log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
  log_file = "logFileDir/" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
  log.basicConfig(
      format=log_FORMAT,
      datefmt='%m/%d/%Y %I:%M:%S %p',
      level=log.INFO,
      handlers=[
          log.FileHandler(log_file),
          log.StreamHandler()
      ]
  )

  REMOTE_IP = "192.5.87.228"
  REMOTE_PORT = "80"
  INTERVAL = 1
  INTERFACE="eno1np0"
  SERVER_IP = '10.52.1.91'
  SERVER_PORT = 8080
  OPTIMIZER="random"
  transfer_service=transferService(REMOTE_IP,REMOTE_PORT,INTERVAL,INTERFACE,SERVER_IP,SERVER_PORT,OPTIMIZER,log)
  transfer_s=transfer_service.reset()

  while transfer_s is not None:
    transfer_s=transfer_service.step(1,1)
    transfer_s=transfer_service.step(3,3)
    transfer_s=transfer_service.step(16,16)

  transfer_service.cleanup()

