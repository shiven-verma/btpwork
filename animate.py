import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# Given data
x_data = np.array([1, 2, 3, 4, 5])  # x-coordinates of circle centers
y_data = np.array([2, 3, 1, 4, 2])  # y-coordinates of circle centers
x_data = np.linspace(1,20,500)
y_data = np.abs(40*np.sin(np.exp(-x_data/5)*np.cos(x_data)))
radius = 0.5  # Radius of the circle

# Create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(min(x_data) - radius, max(x_data) + radius)
ax.set_ylim(min(y_data) - radius, max(y_data) + radius)

# Plot the given data as a line
line, = ax.plot([], [], '-', label='Data Points')

# Initialize the circle patch (empty patch for now)
circle = Circle((0, 0), radius, fill=False, edgecolor='r', label='Moving Circle')
ax.add_patch(circle)

# Animation update function
def update(frame):
    x_center = x_data[frame]
    y_center = y_data[frame]

    # Update the line plot
    line.set_data(x_data[:frame+1], y_data[:frame+1])

    # Update the circle patch
    circle.set_center((x_center, y_center))

    return line, circle

# Set labels and legend
# ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

# Create the animation
animation = FuncAnimation(fig, update, frames=len(x_data), interval=10, blit=True)

# Show the plot
plt.show()
