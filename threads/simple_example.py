import threading
import time

def thread_function(name):
    for i in range(20):
        print(f"Thread {name}: iteration {i}")
        time.sleep(0)  # Yield execution to other threads
    print(f"Thread {name}: finishing")

# Create a few threads
threads = []
for i in range(3):
    t = threading.Thread(target=thread_function, args=(i,))
    threads.append(t)
    t.start()
print("main thread is doing something")
# Wait for all threads to complete
for t in threads:
    t.join()

print("All threads have finished their execution.")