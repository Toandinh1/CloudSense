import matplotlib.pyplot as plt
import numpy as np

# Set style and parameters
plt.style.use('seaborn-whitegrid')
params = {
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"],
    "axes.labelweight": "bold",  # Make axis labels bold
    "axes.linewidth": 2.5,  # Make axes thicker
}
plt.rcParams.update(params)

# Example data (replace with your actual data)
weights = [0.1, 0.25, 0.5, 0.75, 1.0]
accuracy_64 = [36.99, 40.49, 41.75, 40.02, 37.89]
accuracy_128 = [41.83, 42.07, 45.65, 40.19, 42.15]
accuracy_256 = [46.56, 44.47, 41.83, 43.42, 43.24]
accuracy_512 = [44.30, 44.15, 44.74, 44.25, 42.97]
accuracy_1024 = [41.97, 44.45, 40.33, 42.60, 43.94]

# Plotting
plt.figure(figsize=(8, 5))

plt.plot(weights, accuracy_64, 'o-', color='red', label=r'$D = 64$')
plt.plot(weights, accuracy_128, '^-', color='green', label=r'$D = 128$')
plt.plot(weights, accuracy_256, 's-', color='blue', label=r'$D = 256$')
plt.plot(weights, accuracy_512, 'x-', color='cyan', label=r'$D = 512$')
plt.plot(weights, accuracy_1024, '*-', color='orange', label=r'$D = 1024$')

# Labeling
plt.xlabel('Weight $\\lambda$', fontsize=21, fontweight="heavy", color="black")
plt.ylabel(r'PCK$_{20}$ (\%)', fontsize=21, fontweight="heavy", color="black")

# Add legend outside the plot
plt.legend(
    loc='lower center', 
    ncol=3, 
    frameon=True, 
    fontsize=14
)

# Grid and axes styling
plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.5)
plt.ylim(36.5, 47)
plt.xticks(weights,fontsize=20)
plt.yticks(np.arange(36.5, 47, 2), fontsize=20)

# Adjust layout and save
plt.tight_layout()
plt.savefig("/home/jackson-devworks/Desktop/CloudSense/tsne/hyperparameter_sensitivity.pdf", dpi=500, bbox_inches='tight')
plt.show()
