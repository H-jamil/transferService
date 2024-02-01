#pythor server_lr.py 129.114.108.188 10.140.82.12 80


import os
import time
import socket
import sys
import numpy as np
import re

def tcp_stats(REMOTE_IP):
    start = time.time()
    sent, retm = 0, 0
    try:
        data = os.popen("ss -ti").read().split("\n")
        for i in range(1, len(data)):
            if REMOTE_IP in data[i-1]:
                parse_data = data[i].split(" ")
                for entry in parse_data:
                  if "data_segs_out" in entry:
                    # sent += int(re.findall(r'\d+', entry)[-1])  # Find all numbers and take the last one
                    sent += int(entry.split(":")[-1])
                  if "bytes_retrans" in entry:
                    pass
                  elif "retrans" in entry:
                      # retm += int(re.findall(r'\d+', entry)[-1])  # Find all numbers and take the last one
                    retm += int(entry.split("/")[-1])
    except Exception as e:
        print(e)
    end = time.time()
    # print("Time taken to collect tcp stats: {0}ms".format(np.round((end-start)*1000)))
    return sent, retm

if __name__ == "__main__":
  if len(sys.argv) != 4:
    print("Usage: python client.py <REMOTE_IP> <SERVER_IP> <SERVER_PORT>")
    sys.exit(1)
  REMOTE_IP = sys.argv[1]
  SERVER_IP = sys.argv[2]
  SERVER_PORT = int(sys.argv[3])
  while True:
    # t1 = time.time()
    prev_sc, prev_rc = 0, 0
    prev_sc, prev_rc = 0, 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.bind((SERVER_IP, SERVER_PORT))
      s.listen()
      print(f"Server listening on {SERVER_IP}:{SERVER_PORT}")
      conn, addr = s.accept()
      with conn:
        print(f"Connected by {addr}")
        try:
          while True:
            t1 = time.time()
            curr_sc, curr_rc = tcp_stats(REMOTE_IP)
            sc, rc = curr_sc - prev_sc, curr_rc - prev_rc
            lr = 0
            if sc != 0:
                lr = rc/sc if sc > rc else 0
            if lr < 0:
                lr = 0
            prev_sc, prev_rc = curr_sc, curr_rc
            print("Time {0}s: lossRate: {1} ".format(np.round(time.time()), lr))
            message = f"lossRate: {lr}\n"
            try:
              conn.sendall(message.encode())
            # except BrokenPipeError:
            except:
              print("Broken pipe error")
              break
            t2 = time.time()
            time.sleep(max(0, 1 - (t2-t1)))
        except KeyboardInterrupt:
          s.close()
          print("Server stopped manually")


