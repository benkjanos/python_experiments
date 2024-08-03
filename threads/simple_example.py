import threading
import time

some_int: int = 0

def thread_function(name, some_int=None):
    time.sleep(0)
    for i in range(5):
        some_int += 1
        print(f"Thread {name}: iteration {i} , c {some_int}")
        time.sleep(1e-3)  # Yield execution to other threads
    print(f"Thread {name}: finishing")


# Create a few threads
threads = []
for i in range(8):
    t = threading.Thread(target=thread_function, args=(i,7))
    threads.append(t)
print("main thread is doing something")
for t in threads:
    t.start()
print("main thread is doing something 2")

# Wait for all threads to complete
for t in threads:
    t.join()

print("All threads have finished their execution.")
