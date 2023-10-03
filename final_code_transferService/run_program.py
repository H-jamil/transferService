import subprocess
import threading
import time
import os
from datetime import datetime
from multiprocessing import Process,Value
import multiprocessing as mp
import psutil
import socket
import random
import sys
import numpy as np
import re
import pandas as pd

REMOTE_IP = "192.5.87.228"
REMOTE_PORT = "80"
INTERVAL = 1
INTERFACE="eno1np0"
SERVER_IP = '10.52.1.91'
SERVER_PORT = 8080

speed_mbps = Value('d', 0.0)  # 'd' indicates a double
rtt_ms = Value('d', 0.0)  # 'd' indicates a double
c_parallelism = Value('i', 1)
c_concurrency = Value('i', 1)
c_energy = Value('d', 0.0)
manager=mp.Manager()
throughput_logs=manager.list()


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
    print("Time taken to collect tcp stats: {0}ms".format(np.round((end-start)*1000)))
    return sent, retm

def calculate_download_speed(interface):
    prev_rx = psutil.net_io_counters(pernic=True)[interface].bytes_recv
    while True:
        time.sleep(1)
        current_rx = psutil.net_io_counters(pernic=True)[interface].bytes_recv
        speed_bps = current_rx - prev_rx
        speed_mbps.value = (speed_bps * 8) / 1_000_000  # 8 bits per byte and 1e6 bits in a Mbps
        # print(f"Download Speed: {speed_mbps:.2f} Mbps")
        prev_rx = current_rx

def rtt_calculator(REMOTE_IP):
    while True:
        time.sleep(1)
        try:
            # Use ping command with 1 packet and extract the rtt
            output = subprocess.check_output(['ping', '-c', '1', REMOTE_IP]).decode('utf-8')
            # Extract the RTT from the output
            rtt_line = [line for line in output.split('\n') if 'time=' in line][0]
            rtt = rtt_line.split('time=')[-1].split(' ms')[0]
            rtt_ms.value = float(rtt)
        except subprocess.CalledProcessError:
            print(f"Failed to ping {remote_ip}")
            rtt_ms.value = 0.0

def get_energy_consumption():
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
        c_energy.value = energy_consumed

def monitor_process(REMOTE_IP,B,K):
    global throughput_logs
    prev_sc,prev_rc=0,0
    while True:
        t1 = time.time()
        curr_thrpt = speed_mbps.value
        network_rtt=rtt_ms.value
        energy_consumed=c_energy.value
        # print(f"Monitoring Process : Download Speed: {curr_thrpt:.2f} Mbps")
        curr_parallelism=c_parallelism.value
        curr_concurrency=c_concurrency.value
        cc_level=curr_parallelism*curr_concurrency
        record_list=[]
        curr_sc,curr_rc=tcp_stats(REMOTE_IP)
        record_list.append(curr_thrpt)
        record_list.append(curr_parallelism)
        record_list.append(curr_concurrency)
        sc, rc = curr_sc - prev_sc, curr_rc - prev_rc
        lr= 0
        if sc != 0:
          lr = rc/sc if sc>rc else 0
        if lr < 0:
          lr=0
        plr_impact = B*lr
        cc_impact_nl = K**cc_level
        score = (curr_thrpt/cc_impact_nl) - (curr_thrpt * plr_impact)
        score_value = np.round(score * (-1))
        prev_sc,prev_rc=curr_sc,curr_rc
        record_list.append(lr)
        record_list.append(score_value)
        record_list.append(network_rtt)
        record_list.append(energy_consumed)
        record_list.append(datetime.now())
        throughput_logs.append(record_list)
        print("Throughput @{0}s:   {1}Mbps, lossRate: {2} CC:{3}  score:{4}  rtt:{5} ms energy:{6} Jules".format(
            time.time(), curr_thrpt,lr,cc_level,score_value,network_rtt,energy_consumed))
        t2 = time.time()
        time.sleep(max(0, 1 - (t2-t1)))



def run_programs(parallelism, concurrency):
    c_parallelism.value = parallelism
    c_concurrency.value = concurrency
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"./P{parallelism}_C{concurrency}.log"
    log_file_path = f"./logFileDir/{parallelism}_{concurrency}_{current_time}.log"

    # Watchdog handler to kill the process if it hangs
    def kill_process(p):
        try:
            p.kill()
            print(f"Process {p.pid} was killed due to timeout!")
        except ProcessLookupError:
            # Process might have already finished
            pass

    with open(file_name, 'a') as file:
        process = subprocess.Popen(["./parallel_concurrent", REMOTE_IP, log_file_path],
                                   stdout=file, stderr=file)
        # Start the watchdog timer
        watchdog = WatchdogTimer(1000, lambda: kill_process(process))
        watchdog.reset()
        # monitor_process = subprocess.Popen(["./monitor_tcp", str(process.pid),"80"])
        time.sleep(1)
        # subprocess.run(["/home/cc/.pyenv/shims/python", "client.py", str(parallelism), str(concurrency)])
        # # Wait for the process to complete
        # process.communicate()
        client_process = Process(target=run_transfer_parameter_client, args=(SERVER_IP,SERVER_PORT))
        client_process.start()
        client_process.join()
        # monitor_process.terminate()
        # Stop the watchdog timer
        watchdog.stop()
    time.sleep(3)
    os.system("pkill -f parallel_concurrent")
    files_to_remove = [f for f in os.listdir() if f.startswith("FILE")]
    for f in files_to_remove:
        os.remove(f)

def run_transfer_parameter_client(SERVER_IP,SERVER_PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            # print (SERVER_IP, SERVER_PORT)
            s.connect((SERVER_IP, SERVER_PORT))
            break

        except socket.error as e:
            print(f"Failed to connect: {e}")
            time.sleep(1)

    while True:
        print(f"Sending values: {c_parallelism.value} {c_concurrency.value}")

        try:
            s.send(f"{c_parallelism.value} {c_concurrency.value}".encode('utf-8'))
            raw_response = s.recv(1024)
            response = raw_response.decode().strip()
            if response.startswith("TERMINATE"):
              print("Received termination signal. Exiting...")
              break
        except socket.error as e:
            print(f"Failed to communicate with server: {e}")
            break

    s.close()



if __name__ == "__main__":
    run_no = list(range(0, 20))
    parallelism_values = [1, 2, 4, 8, 16]
    concurrency_values = [1, 2, 4, 8, 16]
    # global throughput_logs
    # run_no = list(range(1, 2))
    # parallelism_values = [ 8, 16]
    # concurrency_values = [ 8, 16]
    B=10
    K=1.02
    speed_calculator = Process(target=calculate_download_speed, args=(INTERFACE,))
    speed_calculator.start()
    rtt_calculator_process = Process(target=rtt_calculator, args=(REMOTE_IP,))
    rtt_calculator_process.start()
    energy_process = Process(target=get_energy_consumption)
    energy_process.start()
    monitor_process = Process(target=monitor_process, args=(REMOTE_IP,B,K))
    monitor_process.start()
    for run in run_no:
        for parallelism in parallelism_values:
            for concurrency in concurrency_values:
                file_name = f"./throughputLogs/throughput_data_P{parallelism}_C{concurrency}.csv"
                print(f"Running: Run Number={run}, Parallelism={parallelism}, Concurrency={concurrency}")
                run_programs(parallelism, concurrency)
                columns = ["Throughput", "Parallelism", "Concurrency", "LossRate", "Score", "rtt", "energy", "Time"]
                df = pd.DataFrame(list(throughput_logs), columns=columns)
                write_header = not os.path.exists(file_name)
                df.to_csv(file_name, mode='a', header=write_header, index=False)
                throughput_logs[:] = []
                time.sleep(2)

    speed_calculator.terminate()
    speed_calculator.join()
    rtt_calculator_process.terminate()
    rtt_calculator_process.join()
    energy_process.terminate()
    energy_process.join()
    monitor_process.terminate()
    monitor_process.join()
