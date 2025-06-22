import matplotlib.pyplot as plt

# ================================
#Data from previous assignments of openmp and mpi
# ================================

# CUDA results 
clusters = [3, 5, 7]
cuda_10_iter = [8.2953, 8.2842, 7.1872]   # in ms
cuda_20_iter = [16.1787, 14.9898, 13.0309]

# MPI results 
mpi_procs = [1, 2, 4, 8]
mpi_time = [0.02401, 0.01594, 0.008468, 0.01682]  # in seconds
mpi_time_ms = [t * 1000 for t in mpi_time]
mpi_speedup = [mpi_time[0] / t for t in mpi_time]

# OpenMP results 
omp_threads = [1, 2, 4, 8]
omp_time_ms = [25.3, 15.4, 10.2, 8.1]  
omp_speedup = [omp_time_ms[0] / t for t in omp_time_ms]

# CUDA speedup
cuda_time_ref = 8.2842 
cuda_speedup = [omp_time_ms[0] / cuda_time_ref] * len(omp_threads)

# ================================
# Plot 1: CUDA Execution Time vs Clusters
# ================================
plt.figure(figsize=(8, 5))
plt.plot(clusters, cuda_10_iter, marker='o', label='CUDA - 10 Iterations')
plt.plot(clusters, cuda_20_iter, marker='s', label='CUDA - 20 Iterations')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Execution Time (ms)")
plt.title("CUDA Execution Time vs Clusters")
plt.grid(True)
plt.legend()
plt.savefig("cuda_kmeans_execution_time.png")
plt.show()

# ================================
# Plot 2: MPI Execution Time vs Processes
# ================================
plt.figure(figsize=(8, 5))
plt.plot(mpi_procs, mpi_time_ms, marker='o', color='blue', label="MPI")
plt.xlabel("Number of Processes")
plt.ylabel("Execution Time (ms)")
plt.title("MPI Execution Time vs Number of Processes")
plt.grid(True)
plt.legend()
plt.savefig("mpi_kmeans_execution_time.png")
plt.show()

# ================================
# Plot 3: MPI Speedup vs Processes
# ================================
plt.figure(figsize=(8, 5))
plt.plot(mpi_procs, mpi_speedup, marker='s', color='green', label="MPI Speedup")
plt.xlabel("Number of Processes")
plt.ylabel("Speedup")
plt.title("MPI Speedup vs Number of Processes")
plt.grid(True)
plt.legend()
plt.savefig("mpi_kmeans_speedup.png")
plt.show()

# ================================
# Plot 4: Execution Time Comparison
# ================================
plt.figure(figsize=(8, 5))
plt.plot(omp_threads, omp_time_ms, marker='o', label="OpenMP")
plt.plot(mpi_procs, mpi_time_ms, marker='s', label="MPI")
plt.hlines(cuda_time_ref, xmin=1, xmax=8, colors='red', linestyles='dashed', label="CUDA (k=5, it=10)")
plt.xlabel("Threads / Processes")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time Comparison: OpenMP vs MPI vs CUDA")
plt.grid(True)
plt.legend()
plt.savefig("all_models_execution_time.png")
plt.show()

# ================================
# Plot 5: Speedup Comparison
# ================================
plt.figure(figsize=(8, 5))
plt.plot(omp_threads, omp_speedup, marker='o', label="OpenMP Speedup")
plt.plot(mpi_procs, mpi_speedup, marker='s', label="MPI Speedup")
plt.plot(omp_threads, cuda_speedup, color='red', linestyle='dashed', label=f"CUDA Speedup (~{cuda_speedup[0]:.2f}x)")
plt.xlabel("Threads / Processes")
plt.ylabel("Speedup")
plt.title("Speedup Comparison: OpenMP vs MPI vs CUDA")
plt.grid(True)
plt.legend()
plt.savefig("all_models_speedup.png")
plt.show()
