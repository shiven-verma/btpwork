import pygame
import sys

# Initialize Pygame
pygame.init()

# Set the screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Draw Points and Connect")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

clock = pygame.time.Clock()
FPS = 3

# Define the array of points (x, y)
points = [(100, 100), (200, 300), (300, 200), (400, 400), (500, 300)]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            running = False
    
    # Fill the screen with white
    screen.fill(WHITE)
    
    # Draw points and lines
    for i in range(len(points) - 1):
        pygame.draw.circle(screen, BLACK, points[i], 5)  # Draw points as circles
        pygame.draw.line(screen, RED, points[i], points[i + 1], 2)  # Connect points with lines
        pygame.display.update()
        clock.tick(FPS)
    
    # Draw the last point
    pygame.draw.circle(screen, BLACK, points[-1], 5)  # Draw the last point as a circle
    
    # Update the display
    pygame.display.flip()





