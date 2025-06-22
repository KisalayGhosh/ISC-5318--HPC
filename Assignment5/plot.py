import matplotlib.pyplot as plt

# Data for SMALL RANGE (n=10, m=20)
threads = [1, 2, 4, 8, 16]
exec_small = [0.007, 0.004, 0.003, 0.013, 0.005]
speedup_small = [exec_small[0] / t for t in exec_small]

# Data for LARGE RANGE (n=1B, m=1.03B)
exec_large = [192.571, 189.755, 114.465, 118.387, 110.008]
speedup_large = [exec_large[0] / t for t in exec_large]

# Plot Execution Time Comparison
plt.figure(figsize=(9, 5))
plt.plot(threads, exec_small, marker='o', label='Small Range (10–20)')
plt.plot(threads, exec_large, marker='o', label='Large Range (1B–1.03B)')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Threads (BST Approach)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('execution_time_both_ranges.png')
plt.show()

# Plot Speedup Comparison
plt.figure(figsize=(9, 5))
plt.plot(threads, speedup_small, marker='o', label='Small Range (10–20)')
plt.plot(threads, speedup_large, marker='o', label='Large Range (1B–1.03B)')
plt.plot(threads, threads, linestyle='--', color='gray', label='Ideal Speedup')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.title('Speedup vs Threads (BST Approach)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('speedup_both_ranges.png')
plt.show()
