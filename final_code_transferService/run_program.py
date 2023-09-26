import subprocess
import threading
import time
import os
from datetime import datetime
from multiprocessing import Process,Value
import psutil
import socket
import random
import sys

REMOTE_IP = "192.5.87.228"
REMOTE_PORT = "80"
INTERVAL = 1
INTERFACE="eno1np0"
SERVER_IP = '10.52.1.91'
SERVER_PORT = 8080

speed_mbps = Value('d', 0.0)  # 'd' indicates a double
c_parallelism = Value('i', 1)
c_concurrency = Value('i', 1)

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


def calculate_download_speed(interface):
    prev_rx = psutil.net_io_counters(pernic=True)[interface].bytes_recv
    while True:
        time.sleep(1)
        current_rx = psutil.net_io_counters(pernic=True)[interface].bytes_recv
        speed_bps = current_rx - prev_rx
        speed_mbps.value = (speed_bps * 8) / 1_000_000  # 8 bits per byte and 1e6 bits in a Mbps
        # print(f"Download Speed: {speed_mbps:.2f} Mbps")
        prev_rx = current_rx

def monitor_process():
    while True:
        time.sleep(1)
        print(f"Monitoring Process : Download Speed: {speed_mbps.value:.2f} Mbps")

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
        watchdog = WatchdogTimer(500, lambda: kill_process(process))
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
    # SERVER_IP = '10.52.1.91'
    # SERVER_PORT = 8080
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
    # run_no = list(range(1, 21))
    # parallelism_values = [1, 2, 4, 8, 16]
    # concurrency_values = [1, 2, 4, 8, 16]
    run_no = list(range(1, 2))
    parallelism_values = [ 8, 16]
    concurrency_values = [ 8, 16]
    speed_calculator = Process(target=calculate_download_speed, args=(INTERFACE,))
    speed_calculator.start()
    monitor_process = Process(target=monitor_process)
    monitor_process.start()
    for run in run_no:
        for parallelism in parallelism_values:
            for concurrency in concurrency_values:
                print(f"Running: Run Number={run}, Parallelism={parallelism}, Concurrency={concurrency}")
                run_programs(parallelism, concurrency)
                time.sleep(2)

    speed_calculator.terminate()
    speed_calculator.join()
    monitor_process.terminate()
    monitor_process.join()
