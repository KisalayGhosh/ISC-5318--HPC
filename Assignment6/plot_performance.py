import matplotlib.pyplot as plt

# Data
num_processes = [1, 2, 4, 8]
execution_times = [0.009568, 0.012749, 0.008736, 0.010668]  # In seconds

# Calculate Speedup: Speedup = T1 / Tp (where T1 is time with 1 process, Tp is time with p processes)
speedup = [execution_times[0] / t for t in execution_times]

plt.figure(figsize=(8, 5))
plt.plot(num_processes, execution_times, marker='o', linestyle='-', markersize=8, label='Execution Time')
plt.xlabel("Number of Processes")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time vs Number of Processes (MPI Grayscale Conversion)")
plt.xticks(num_processes)
plt.grid(True)
plt.legend()
plt.savefig("mpi_grayscale_performance.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(num_processes, speedup, marker='s', linestyle='-', markersize=8, color='orange', label='Speedup')
plt.xlabel("Number of Processes")
plt.ylabel("Speedup (T1 / Tp)")
plt.title("Speedup vs Number of Processes (MPI Grayscale Conversion)")
plt.xticks(num_processes)
plt.grid(True)
plt.legend()
plt.savefig("mpi_grayscale_speedup.png")
plt.show()
