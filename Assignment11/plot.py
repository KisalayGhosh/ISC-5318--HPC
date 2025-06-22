import matplotlib.pyplot as plt

# === DATA ===
clusters = [3, 5, 7, 128]
basic_exec = [8.2953, 8.2842, 7.1872, 28.6827]
opt_exec = [23.8699, 18.2318, 13.3845, 24.3257]
speedup = [b/o for b, o in zip(basic_exec, opt_exec)]

# === 1. Bar Plot at k=128 ===
plt.figure()
plt.bar(['Basic CUDA', 'Optimized CUDA'], [basic_exec[-1], opt_exec[-1]], color=['red', 'green'])
plt.title("Execution Time Comparison at k=128 (30 Iter)")
plt.ylabel("Execution Time (ms)")
plt.savefig("bar_comparison_k128.png")
plt.show()

# === 2. Combined Execution Time ===
plt.figure()
plt.plot(clusters[:-1], basic_exec[:-1], 'o-', label='Basic CUDA - 10 Iter')
plt.plot(clusters[:-1], opt_exec[:-1], 'o--', label='Optimized CUDA - 10 Iter')
plt.plot(clusters[:-1], [16.1787, 14.9898, 13.0309], 's-', label='Basic CUDA - 20 Iter')
plt.plot(clusters[:-1], [51.3646, 35.7843, 26.9140], 's--', label='Optimized CUDA - 20 Iter')
plt.title("Basic vs Optimized CUDA Execution Time")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Execution Time (ms)")
plt.grid(True)
plt.legend()
plt.savefig("cuda_comparison_execution_time.png")
plt.show()

# === 3. Speedup Comparison ===
plt.figure()
plt.plot(clusters[:-1], [a/b for a, b in zip(basic_exec[:-1], opt_exec[:-1])], 'o-', label='10 Iterations')
plt.plot(clusters[:-1], [a/b for a, b in zip([16.1787, 14.9898, 13.0309], [51.3646, 35.7843, 26.9140])], 's-', label='20 Iterations')
plt.axhline(1.0, linestyle='--', color='gray')
plt.title("Speedup of Basic over Optimized CUDA")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Speedup (Basic / Optimized)")
plt.grid(True)
plt.legend()
plt.savefig("cuda_speedup_comparison.png")
plt.show()

# === 4. Execution Time vs Clusters (Basic vs Optimized CUDA) ===
plt.figure()
plt.plot(clusters, basic_exec, marker='o', label='Basic CUDA')
plt.plot(clusters, opt_exec, marker='s', label='Optimized CUDA')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time vs Clusters (Basic vs Optimized CUDA)")
plt.grid(True)
plt.legend()
plt.savefig("execution_time_vs_clusters.png")
plt.show()

# === 5. Execution Time vs Iterations (k=128) ===
plt.figure()
plt.scatter([30], [basic_exec[-1]], label="Basic CUDA (k=128)")
plt.scatter([30], [opt_exec[-1]], label="Optimized CUDA (k=128)")
plt.xlabel("Iterations")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time vs Iterations (k=128)")
plt.grid(True)
plt.legend()
plt.savefig("execution_time_vs_iterations_k128.png")
plt.show()

# === 6. Optimized CUDA Execution Time vs Clusters ===
opt_exec_10 = [23.8699, 18.2318, 13.3845]
opt_exec_20 = [51.3646, 35.7843, 26.9140]
plt.figure()
plt.plot(clusters[:-1], opt_exec_10, marker='o', label='10 Iterations')
plt.plot(clusters[:-1], opt_exec_20, marker='s', label='20 Iterations')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Execution Time (ms)")
plt.title("Optimized CUDA Execution Time vs Clusters")
plt.grid(True)
plt.legend()
plt.savefig("optimized_cuda_kmeans_execution_time.png")
plt.show()

# === 7. Optimized CUDA Execution Time vs Iterations (Fixed k) ===
iters = [10, 20]
plt.figure()
for i, k in enumerate(clusters[:-1]):
    y = [opt_exec_10[i], opt_exec_20[i]]
    plt.plot(iters, y, marker='o', label=f'k = {k}')
plt.xlabel("Iterations")
plt.ylabel("Execution Time (ms)")
plt.title("Optimized CUDA Execution Time vs Iterations (Fixed k)")
plt.grid(True)
plt.legend()
plt.savefig("optimized_cuda_time_vs_iterations.png")
plt.show()

# === 8. Extended Speedup vs Clusters ===
plt.figure()
plt.plot(clusters, speedup, marker='^', color='purple')
plt.axhline(1.0, linestyle='--', color='gray', label='Baseline Speedup = 1')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Speedup (Basic / Optimized)")
plt.title("Speedup of Basic CUDA over Optimized CUDA")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("speedup_vs_clusters.png")
plt.show()
