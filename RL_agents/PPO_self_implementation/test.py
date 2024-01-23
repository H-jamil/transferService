import socket
import multiprocessing
from time import sleep

def handle_client(server_socket, start_signal, response_signal,finish_signal):
    print("Server process is waiting for connection")
    connection, _ = server_socket.accept()
    while True:
      if start_signal.value:
          connection.sendall(b'start')
          start_signal.value = False
          if connection.recv(1024).decode() == 'done':
              response_signal.value = True
              start_signal.value = False
      if finish_signal.value:
          print("Server process is closing the connection from finish signal")
          connection.sendall(b'finish')
          connection.close()
          break

def main():
    start_signal = multiprocessing.Value('i', False)
    response_signal = multiprocessing.Value('i', False)
    finish_signal = multiprocessing.Value('i', False)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('10.140.82.45', 12345))
    server_socket.listen(1)

    process = multiprocessing.Process(target=handle_client, args=(server_socket, start_signal, response_signal,finish_signal))
    process.start()

    print("Server is running and sending signal to the client to start iperf3")
    start_signal.value = True

    print("Waiting for client to complete the task")
    while not response_signal.value:
        pass

    print("Client completed the task")
    sleep(2)
    response_signal.value = False
    print("Server is running and sending signal to the client to start iperf3")
    start_signal.value = True

    print("Waiting for client to complete the task")
    while not response_signal.value:
        pass

    print("Client completed the task")

    finish_signal.value = True

    sleep(2)


    process.terminate()
    process.join()

if __name__ == "__main__":
    main()
