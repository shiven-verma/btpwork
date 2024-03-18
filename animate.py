import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# Given data

# x_data = np.linspace(1,20,500)
# y_data = np.abs(40*np.sin(np.exp(-x_data/5)*np.cos(x_data)))

file_path = "/home/shivendra/IITM/BtechProject/Python/mpcdata1.csv"
data = np.genfromtxt(file_path, delimiter=',')

x_data = data[1,:]
y_data = data[0,:]
obsxdata = data[3,:]
obsydata = data[2,:] 
pathx = data[4,:]
pathy = data[5,:]

radius = 1.7  # Radius of the circle

# Create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(min(pathy-3) - radius, max(pathy+1) + radius)
ax.set_ylim(min(pathx-3) - radius, max(pathx+1) + radius)

# Plot the given data as a line
line, = ax.plot([], [], '-', label='Agent')
line1, = ax.plot([], [], '-', label='Obs Path')

# Initialize the circle patch (empty patch for now)
circle = Circle((0, 0), radius, fill=False, edgecolor='r', label='Moving Obstacle')
ax.add_patch(circle)

# Animation update function
def update(frame):
    x_center = obsxdata[frame]
    y_center = obsydata[frame]

    # Update the line plot
    line.set_data(x_data[:frame+1], y_data[:frame+1])
    line1.set_data(obsxdata[:frame+1], obsydata[:frame+1])

    # Update the circle patch
    circle.set_center((x_center, y_center))

    return line,line1, circle

# Set labels and legend
# ax = plt.gca()
plt.plot(pathy,pathx,label="path")
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend(fontsize=5.5,loc="lower right")

# Create the animation
animation = FuncAnimation(fig, update, frames=len(x_data), interval=200, blit=True)

# Show the plot
plt.show()
# animation.save('obs_incoming_speed_2.gif', writer = 'pillow', fps = 10)
