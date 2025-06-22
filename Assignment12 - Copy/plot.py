import matplotlib.pyplot as plt
import numpy as np

# Threads / Processes
threads = np.array([1, 2, 4, 8])

# Execution times in seconds
execution_times = {
    "OpenMP":  np.array([0.003044, 0.000446, 0.000258, 0.000230]),
    "MPI":     np.array([0.00125,  0.00140,  0.00070,  0.00090]),
    "CUDA":    np.array([0.000368] * 4),
    "OpenACC": np.array([0.362456, None, None, None], dtype='float')
}

# Baseline for speedup
baseline = execution_times["OpenMP"][0]

# Calculate speedup for each method
speedups = {
    method: baseline / times for method, times in execution_times.items()
}

# Calculate efficiency = speedup / threads
efficiencies = {
    method: speedups[method] / threads for method in speedups
}

# 1. Execution Time Plot
plt.figure(figsize=(10, 5))
for method, times in execution_times.items():
    plt.plot(threads, times, marker='o', label=method)
plt.xlabel("Threads / Processes")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("execution_time_comparison_all_ext.png")
plt.show()

# 2. Speedup Plot
plt.figure(figsize=(10, 5))
for method, spd in speedups.items():
    plt.plot(threads, spd, marker='s', label=method)
plt.xlabel("Threads / Processes")
plt.ylabel("Speedup (vs OpenMP 1-thread)")
plt.title("Speedup Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_comparison_all_ext.png")
plt.show()

# 3. Efficiency Plot
plt.figure(figsize=(10, 5))
for method, eff in efficiencies.items():
    plt.plot(threads, eff, marker='^', label=method)
plt.xlabel("Threads / Processes")
plt.ylabel("Efficiency (Speedup / Threads)")
plt.title("Efficiency Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("efficiency_comparison_all.png")
plt.show()

# 4. Normalized Execution Time Plot
plt.figure(figsize=(10, 5))
for method, times in execution_times.items():
    norm_time = times / baseline
    plt.plot(threads, norm_time, marker='D', label=method)
plt.xlabel("Threads / Processes")
plt.ylabel("Normalized Time (vs OpenMP 1-thread)")
plt.title("Normalized Execution Time Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("normalized_time_comparison_all.png")
plt.show()
