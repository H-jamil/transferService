import socket
import random
import time
import sys


MAX_PARALLELISM = 4
MAX_CONCURRENCY = 4

def main():
    # Check if correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python <script_name> <user_parallelism> <user_concurrency>")
        sys.exit(1)

    # Get values from command line arguments
    user_parallelism = int(sys.argv[1])
    user_concurrency = int(sys.argv[2])

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # s.connect(('127.0.0.1', 8080))
        s.connect(('10.52.1.91', 8080))
    except socket.error as e:
        print(f"Failed to connect: {e}")
        return

    while True:
        # user_parallelism = random.randint(1, MAX_PARALLELISM)
        # user_concurrency = random.randint(1, MAX_CONCURRENCY)
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
