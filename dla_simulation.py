import pygame
import numpy as np
import random
import sys
import time

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Diffusion-Limited Aggregation")

# Debug flag
DEBUG = False

# Print debug information
if DEBUG:
    print(f"Window initialized with dimensions: {WIDTH}x{HEIGHT}")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Particle properties
PARTICLE_RADIUS = 2
SEED_RADIUS = 3
WALKER_RADIUS = 2

# DLA parameters
MAX_PARTICLES = 5000
STICKING_PROBABILITY = 1.0  # Probability of sticking when in contact
WALKER_SPEED = 1  # Pixels per step
NUM_WALKERS = 500  # Number of walkers to simulate simultaneously

# Initialize the grid to track occupied positions
grid = np.zeros((WIDTH, HEIGHT), dtype=bool)
particles = []  # List to store particle positions
walkers = []  # List to store all active walkers

# Place the seed particle at the center
seed_x, seed_y = WIDTH // 2, HEIGHT // 2
grid[seed_x, seed_y] = True
particles.append((seed_x, seed_y))
if DEBUG:
    print(f"Seed particle placed at: ({seed_x}, {seed_y})")

# Function to generate a random walker at the edge of the screen
def generate_walker():
    side = random.randint(0, 3)  # 0: top, 1: right, 2: bottom, 3: left
    
    if side == 0:  # Top
        x = random.randint(0, WIDTH - 1)
        y = 0
    elif side == 1:  # Right
        x = WIDTH - 1
        y = random.randint(0, HEIGHT - 1)
    elif side == 2:  # Bottom
        x = random.randint(0, WIDTH - 1)
        y = HEIGHT - 1
    else:  # Left
        x = 0
        y = random.randint(0, HEIGHT - 1)
    
    if DEBUG:
        print(f"Generated new walker on side {side} at position ({x}, {y})")
    return (x, y)

# Function to check if a walker is in contact with any existing particle
def check_contact(x, y):
    # Check in a 3x3 neighborhood around the walker
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and grid[nx, ny]:
                # Add a small random chance to stick based on STICKING_PROBABILITY
                if random.random() < STICKING_PROBABILITY:
                    return True
    return False

# Function to move a walker
def move_walker(x, y):
    # Random direction: up, right, down, left
    dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
    new_x, new_y = x + dx, y + dy
    
    # Check if the new position is within bounds
    if 0 <= new_x < WIDTH and 0 <= new_y < HEIGHT:
        return new_x, new_y
    else:
        # If out of bounds, generate a new walker
        return generate_walker()

# Function to draw the DLA structure
def draw_dla():
    global walkers
    screen.fill(BLACK)
    
    # Draw a grid to help visualize the coordinate system
    for x in range(0, WIDTH, 50):
        pygame.draw.line(screen, (30, 30, 30), (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 50):
        pygame.draw.line(screen, (30, 30, 30), (0, y), (WIDTH, y))
    
    # Draw all particles
    for x, y in particles:
        pygame.draw.circle(screen, WHITE, (int(x), int(y)), PARTICLE_RADIUS)
    
    # Draw the seed particle
    pygame.draw.circle(screen, RED, (int(seed_x), int(seed_y)), SEED_RADIUS)
    
    # Draw all walkers
    for walker in walkers:
        pygame.draw.circle(screen, GREEN, (int(walker[0]), int(walker[1])), WALKER_RADIUS)
    
    # Add debug text to show counts
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Particles: {len(particles)} Walkers: {len(walkers)}", True, GREEN)
    screen.blit(text, (10, 10))
    
    # Draw center crosshair
    pygame.draw.line(screen, (100, 100, 100), (WIDTH//2, 0), (WIDTH//2, HEIGHT), 1)
    pygame.draw.line(screen, (100, 100, 100), (0, HEIGHT//2), (WIDTH, HEIGHT//2), 1)
    
    pygame.display.flip()

def run_simulation():
    global particles, walkers
    
    if DEBUG:
        print("Starting simulation...")
    clock = pygame.time.Clock()
    running = True
    particle_count = 1  # Start with the seed particle
    
    if DEBUG:
        print("Entering main simulation loop...")
    while running and particle_count < MAX_PARTICLES:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                if DEBUG:
                    print("Quit event received")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    if DEBUG:
                        print("Escape key pressed")
        
        # Move all walkers
        new_walkers = []
        for walker in walkers:
            new_pos = move_walker(*walker)
            
            # Check if walker is in contact with any existing particle
            if check_contact(*new_pos):
                # Add the walker as a new particle
                x, y = new_pos
                grid[x, y] = True
                particles.append((x, y))
                particle_count += 1
                if DEBUG:
                    print(f"Particle added at ({x}, {y}). Total particles: {particle_count}")
                
                # Replace this walker with a new one
                new_walkers.append(generate_walker())
            else:
                # Keep the walker moving
                new_walkers.append(new_pos)
        
        # Update walkers list
        walkers = new_walkers
        
        # Draw the current state
        draw_dla()
        
        # Limit the frame rate
        clock.tick(60)
    
    if DEBUG:
        print("Main simulation loop ended")
    # Keep the window open after simulation completes
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        draw_dla()
        clock.tick(30)
    
    if DEBUG:
        print("Simulation ended")

# Initialize walkers
for _ in range(NUM_WALKERS):
    walkers.append(generate_walker())
if DEBUG:
    print(f"Created {NUM_WALKERS} initial walkers")

# Run the simulation
if __name__ == "__main__":
    run_simulation()
    pygame.quit()
    sys.exit() 