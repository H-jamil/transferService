import threading
import time

class WorkerThread(threading.Thread):
    def __init__(self, id):
        super().__init__()
        self.id = id
        self.active = True
        self.pause_condition = threading.Condition(threading.Lock())  # Use a condition to manage pausing
        self.paused = False

    def run(self):
        while self.active:
            with self.pause_condition:
                while self.paused:
                    self.pause_condition.wait()  # Release lock and wait for notification

            print(f"ID: {self.id}, Time: {time.ctime()}")

            # Busy-wait for about 1 second
            end_time = time.time() + 1
            while time.time() < end_time:
                pass

    def stop(self):
        """ Stop the thread completely """
        self.active = False
        # If paused, resume the thread to ensure it can be stopped
        if self.paused:
            self.resume()

    def pause(self):
        """ Pause the thread """
        with self.pause_condition:
            self.paused = True

    def resume(self):
        """ Resume the thread """
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify()  # Unpause the paused thread


def main():
    # Start 5 worker threads
    threads = [WorkerThread(i) for i in range(5)]
    for t in threads:
        t.start()

    time.sleep(3)  # Let the threads run for 5 seconds

    # Pause the first 2 threads
    print("\nPausing first 2 threads\n")
    for i in range(2):
        threads[i].pause()

    time.sleep(3)  # Let the remaining threads run for 5 more seconds

    # Restart the paused threads
    print("\nRestarting the paused threads\n")
    for i in range(2):
        threads[i].resume()

    # Let them run for another 5 seconds
    time.sleep(3)

    # Stop all threads
    for t in threads:
        t.stop()
        t.join()

    print("All threads have been stopped")


if __name__ == '__main__':
    main()
