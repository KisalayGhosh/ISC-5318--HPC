import numpy as np
import matplotlib.pyplot as plt
import os

def load_all_parts(prefix="output_rank_", max_ranks=16):
    data = []
    for rank in range(max_ranks):
        filename = f"{prefix}{rank}.txt"
        if os.path.exists(filename):
            print(f"Loading: {filename}")
            d = np.loadtxt(filename)
            data.append(d)
        else:
            print(f"Stopping at rank {rank}: {filename} not found.")
            break
    return np.vstack(data) if data else None


full_grid = load_all_parts()

if full_grid is not None:
    plt.imshow(full_grid, cmap='hot', origin='lower')
    plt.colorbar(label='Temperature (°C)')
    plt.title("2D Heat Distribution at T=20")
    plt.xlabel("Y-axis")
    plt.ylabel("X-axis")
    plt.savefig("heatmap.png")  
    plt.show()
else:
    print("No data found. Please check output files.")
