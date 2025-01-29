import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Schwefel function
def schwefel(x, y):
    return 418.9829 * 2 - x * 50 * np.sin(np.sqrt(np.abs(x * 50))) - y * 50 * np.sin(np.sqrt(np.abs(y * 50)))

# Generate a grid of points
x = np.linspace(-10, 10, 1000)  # Domain of the Schwefel function
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)
Z = schwefel(X, Y)

# Create the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Add color bar and labels
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.set_title('Schwefel Function')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Adjust viewing angle
ax.view_init(elev=40, azim=45)

# Show the plot
plt.show()
