import matplotlib.pyplot as plt

# Data from provided execution times
k_values = [2, 4, 5, 8, 16]
threads = [1, 2, 4, 4, 8]
iterations = [10, 20, 20, 30, 40]
execution_times = [0.012, 0.029, 0.018, 0.038, 0.074]

# Plot 1: Execution Time vs Threads
plt.figure(figsize=(10, 6))
plt.plot(threads, execution_times, marker='o')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Number of Threads with OpenMP')
plt.grid(True)
plt.show()

# Plot 2: Execution Time vs Iterations
plt.figure(figsize=(10, 6))
plt.plot(iterations, execution_times, marker='s')
plt.xlabel('Number of Iterations')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Number of Iterations with OpenMP')
plt.grid(True)
plt.show()

# Plot 3: Execution Time vs Clusters (k) with Threads Annotated
plt.figure(figsize=(10, 6))
for i, txt in enumerate(threads):
    plt.annotate(f"{txt} threads", (k_values[i], execution_times[i]))
plt.plot(k_values, execution_times, marker='^')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Clusters (k) with Threads Annotated')
plt.grid(True)
plt.show()
