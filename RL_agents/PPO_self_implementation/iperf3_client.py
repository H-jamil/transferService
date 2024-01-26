import socket
import subprocess
import sys

if len(sys.argv) != 3:
    print("Usage: python client.py [server_address] [iperf3_host]")
    sys.exit(1)

server_address = sys.argv[1]
iperf3_host = sys.argv[2]

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_address, 12345))

while True:
    message = client_socket.recv(1024)
    if message.decode() == 'start':
        subprocess.run(["iperf3", "-c", iperf3_host, "-P8", "-t30"])
        client_socket.sendall(b'done')
    elif message.decode() == 'finish':
        client_socket.close()
        break
