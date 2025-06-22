import matplotlib.pyplot as plt

# Execution time results (averaged from multiple runs)
processes = [1, 2, 4, 8]
execution_times = [0.02401, 0.01594, 0.008468, 0.01682]  # Averaged values

# Compute Speedup (Baseline: Execution time with 1 process)
baseline_time = execution_times[0]
speedup = [baseline_time / t for t in execution_times]

#  Plot 1: Execution Time vs Number of Processes
plt.figure(figsize=(8, 5))
plt.plot(processes, execution_times, marker='o', linestyle='-', color='blue', label="Execution Time")
plt.xlabel("Number of Processes")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time vs Number of Processes")
plt.grid(True)
plt.legend()
plt.savefig("mpi_kmeans_execution_time.png")  # Save the plot
plt.show()

#  Plot 2: Speedup vs Number of Processes
plt.figure(figsize=(8, 5))
plt.plot(processes, speedup, marker='s', linestyle='-', color='green', label="Speedup")
plt.xlabel("Number of Processes")
plt.ylabel("Speedup")
plt.title("Speedup vs Number of Processes")
plt.grid(True)
plt.legend()
plt.savefig("mpi_kmeans_speedup.png")  # Save the plot
plt.show()
