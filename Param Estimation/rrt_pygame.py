import sys
import random
import math
import pygame
from pygame.locals import QUIT

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Define screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Define node radius
NODE_RADIUS = 2

# Define step size
STEP_SIZE = 40

# Define goal bias (probability of selecting the goal as the target)
GOAL_BIAS = 0.05

# Define maximum number of iterations
MAX_ITERATIONS = 5000

# Define goal coordinates
GOAL_X = 750
GOAL_Y = 350

# Function to calculate distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to find nearest node
def nearest_node(nodes, target):
    nearest_node = None
    min_dist = float('inf')
    for node in nodes:
        dist = distance(node, target)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node

# Function to steer towards the target
def steer(node, target, step_size):
    if distance(node, target) < step_size:
        return target
    else:
        theta = math.atan2(target[1] - node[1], target[0] - node[0])
        return node[0] + step_size * math.cos(theta), node[1] + step_size * math.sin(theta)
    

# Function to check if there is an obstacle between two points
def collision_free(p1, p2):
    # For simplicity, assume no obstacles
    return True

# Main function to run RRT algorithm
def rrt(start, goal):
    nodes = [start]
    for _ in range(MAX_ITERATIONS):
        if random.random() < GOAL_BIAS:
            target = goal
        else:
            target = (random.uniform(0, SCREEN_WIDTH), random.uniform(0, SCREEN_HEIGHT))
        
        nearest = nearest_node(nodes, target)
        new_node = steer(nearest, target, STEP_SIZE)
        
        if collision_free(nearest, new_node):
            nodes.append(new_node)
            pygame.draw.line(screen, BLACK, nearest, new_node)
            pygame.display.update()
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
        
        if distance(new_node, goal) < STEP_SIZE:
            pygame.draw.line(screen, RED, new_node, goal)
            pygame.display.update()
            clock.tick(FPS)
            return nodes
    return None

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("RRT Algorithm")

clock = pygame.time.Clock()
FPS = 30

# Draw start and goal
screen.fill(WHITE)
pygame.draw.circle(screen, RED, (50, 50), NODE_RADIUS)
pygame.draw.circle(screen, RED, (GOAL_X, GOAL_Y), NODE_RADIUS)

# Run RRT algorithm
start = (50, 50)
goal = (GOAL_X, GOAL_Y)
rrt_path = rrt(start, goal)

# Display result
if rrt_path:
    print("Path found!")
else:
    print("Failed to find path.")

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    



# Quit pygame

