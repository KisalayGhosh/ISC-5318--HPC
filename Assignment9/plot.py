import matplotlib.pyplot as plt

# Threads/processes
threads = [1, 2, 4, 8]

# Execution times in seconds
openmp_times = [0.003044, 0.000446, 0.000258, 0.000230]
mpi_times = [0.00125, 0.00140, 0.00070, 0.00090]
cuda_time = 0.000368


plt.figure(figsize=(10, 6))
plt.plot(threads, openmp_times, marker='o', label='OpenMP', color='blue')
plt.plot(threads, mpi_times, marker='s', label='MPI', color='green')
plt.axhline(y=cuda_time, color='red', linestyle='--', label='CUDA (0.00037s)')

plt.title('Execution Time Comparison: OpenMP vs MPI vs CUDA')
plt.xlabel('Number of Threads / Processes')
plt.ylabel('Execution Time (seconds)')
plt.xticks(threads)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('execution_time_comparison.png', dpi=300)
plt.show()


openmp_speedup = [openmp_times[0] / t for t in openmp_times]
mpi_speedup = [mpi_times[0] / t for t in mpi_times]
cuda_speedup = [openmp_times[0] / cuda_time] * len(threads)

plt.figure(figsize=(10, 6))
plt.plot(threads, openmp_speedup, marker='o', label='OpenMP Speedup', color='blue')
plt.plot(threads, mpi_speedup, marker='s', label='MPI Speedup', color='green')
plt.plot(threads, cuda_speedup, linestyle='--', color='red', label=f'CUDA Speedup (~{openmp_times[0] / cuda_time:.2f}x)')

plt.title('Speedup Comparison: OpenMP vs MPI vs CUDA')
plt.xlabel('Number of Threads / Processes')
plt.ylabel('Speedup (Relative to OpenMP - 1 Thread)')
plt.xticks(threads)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('speedup_comparison.png', dpi=300)
plt.show()
