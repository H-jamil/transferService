import socket
import random
import time

MAX_PARALLELISM = 4
MAX_CONCURRENCY = 4

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.connect(('127.0.0.1', 8080))
    except socket.error as e:
        print(f"Failed to connect: {e}")
        return

    while True:
        user_parallelism = random.randint(1, MAX_PARALLELISM)
        user_concurrency = random.randint(1, MAX_CONCURRENCY)
        print(f"Sending values: {user_parallelism} {user_concurrency}")

        try:
            s.send(f"{user_parallelism} {user_concurrency}".encode('utf-8'))
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
    main()
