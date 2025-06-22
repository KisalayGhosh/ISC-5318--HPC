import subprocess
import matplotlib.pyplot as plt
import os

# Define the range of threads to test
thread_counts = [1, 2, 4, 8, 16, 32, 64]
n, m = 1000000000, 1000001000  

execution_times = []

for threads in thread_counts:
    print(f"Running with {threads} threads...")

    # Set OMP_NUM_THREADS 
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    # Run the OpenMP program and pass input
    result = subprocess.run(
        ["./prime_gap"],
        input=f"{n} {m}\n",
        text=True,
        capture_output=True,
        env=env
    )

    
    print("Program Output:\n", result.stdout)

    # Extract execution time from output
    found_time = False
    for line in result.stdout.split("\n"):
        if "Execution Time:" in line:
            time_taken = float(line.split(":")[-1].strip().split()[0])
            execution_times.append(time_taken)
            found_time = True
            break

    if not found_time:
        print(f"Warning: Execution time not found for {threads} threads!")

# Ensure execution_times is correct
if len(execution_times) != len(thread_counts):
    print("Error: Execution times were not captured correctly. Check program output format.")
    exit(1)

# Compute speedup
baseline_time = execution_times[0]  # Time for 1 thread
speedup = [baseline_time / t for t in execution_times]

# Plot Execution Time vs Threads
plt.figure(figsize=(8, 5))
plt.plot(thread_counts, execution_times, marker='o', linestyle='-', label="Execution Time")
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (seconds)")
plt.title("Performance Analysis of OpenMP Prime Gap Program")
plt.grid(True)
plt.legend()
plt.savefig("execution_time_plot.png")
plt.show()

# Plot Speedup vs Threads
plt.figure(figsize=(8, 5))
plt.plot(thread_counts, speedup, marker='o', linestyle='-', label="Speedup")
plt.plot(thread_counts, thread_counts, linestyle="dashed", label="Ideal Speedup (Linear)")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup Analysis of OpenMP Prime Gap Program")
plt.grid(True)
plt.legend()
plt.savefig("speedup_plot.png")
plt.show()
