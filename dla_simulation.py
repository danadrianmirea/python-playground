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

# Performance settings
DRAW_FREQUENCY = 1  # Only draw every N frames
BATCH_SIZE = 50  # Process walkers in batches

# View settings
ZOOM_FACTOR = 1.2  # How much to zoom in/out per scroll
MIN_ZOOM = 0.1
MAX_ZOOM = 10.0
view_zoom = 1.0
# Initialize view offset to center on the seed particle
view_offset_x = 0
view_offset_y = 0
is_panning = False
last_mouse_pos = None

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
PARTICLE_RADIUS = 1
SEED_RADIUS = 1
WALKER_RADIUS = 1

# DLA parameters
MAX_PARTICLES = 50000
STICKING_PROBABILITY = 1.0  # Probability of sticking when in contact
WALKER_SPEED = 1  # Pixels per step
NUM_WALKERS = 2000  # Number of walkers to simulate simultaneously
ATTRACTION_STRENGTH = 0.0  # Strength of the attraction force (0.0 to 1.0)
EDGE_GENERATION = True  # Whether to generate walkers at the edges of the screen

# Initialize the grid to track occupied positions
grid = np.zeros((WIDTH, HEIGHT), dtype=bool)
particles = []  # List to store particle positions
walkers = np.zeros((NUM_WALKERS, 2), dtype=int)  # NumPy array for walker positions

# Place the seed particle at the center
seed_x, seed_y = WIDTH // 2, HEIGHT // 2
grid[seed_x, seed_y] = True
particles.append((seed_x, seed_y))
if DEBUG:
    print(f"Seed particle placed at: ({seed_x}, {seed_y})")

# Function to generate a random walker at a random position on the screen
def generate_walker():
    if EDGE_GENERATION:
        # Choose a random edge (0: top, 1: right, 2: bottom, 3: left)
        edge = random.randint(0, 3)
        if edge == 0:  # Top edge
            x = random.randint(0, WIDTH - 1)
            y = 0
        elif edge == 1:  # Right edge
            x = WIDTH - 1
            y = random.randint(0, HEIGHT - 1)
        elif edge == 2:  # Bottom edge
            x = random.randint(0, WIDTH - 1)
            y = HEIGHT - 1
        else:  # Left edge
            x = 0
            y = random.randint(0, HEIGHT - 1)
    else:
        # Original random position generation
        x = random.randint(0, WIDTH - 1)
        y = random.randint(0, HEIGHT - 1)
    
    if DEBUG:
        print(f"Generated new walker at position ({x}, {y})")
    return np.array([x, y])

# Function to check if walkers are in contact with any existing particle
def check_contacts(walker_positions):
    # Create a mask for valid positions
    valid_mask = (walker_positions[:, 0] >= 0) & (walker_positions[:, 0] < WIDTH) & \
                 (walker_positions[:, 1] >= 0) & (walker_positions[:, 1] < HEIGHT)
    
    # Initialize result array
    contacts = np.zeros(len(walker_positions), dtype=bool)
    
    # For each valid position, check the 3x3 neighborhood
    for i in range(len(walker_positions)):
        if valid_mask[i]:
            x, y = walker_positions[i]
            # Check surrounding cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and grid[nx, ny]:
                        if random.random() < STICKING_PROBABILITY:
                            contacts[i] = True
                            break
                if contacts[i]:
                    break
    
    return contacts

# Function to move walkers
def move_walkers(walker_positions):
    # Calculate direction to center for each walker
    center = np.array([WIDTH//2, HEIGHT//2])
    to_center = center - walker_positions
    
    # Normalize the direction vectors
    distances = np.sqrt(np.sum(to_center**2, axis=1))
    distances[distances == 0] = 1  # Avoid division by zero
    to_center = to_center / distances[:, np.newaxis]
    
    # Generate random movements
    directions = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)])
    indices = np.random.randint(0, 4, size=len(walker_positions))
    random_movements = directions[indices]
    
    # Combine random movement with attraction force
    movements = random_movements + to_center * ATTRACTION_STRENGTH
    
    # Round to nearest integer for pixel positions
    movements = np.round(movements).astype(int)
    
    new_positions = walker_positions + movements
    
    # Handle out-of-bounds walkers
    out_of_bounds = (new_positions[:, 0] < 0) | (new_positions[:, 0] >= WIDTH) | \
                    (new_positions[:, 1] < 0) | (new_positions[:, 1] >= HEIGHT)
    
    # Replace out-of-bounds walkers with new ones
    for i in np.where(out_of_bounds)[0]:
        new_positions[i] = generate_walker()
    
    return new_positions

# Function to convert world coordinates to screen coordinates
def world_to_screen(x, y):
    # Convert from world coordinates (seed at 0,0) to screen coordinates
    screen_x = x * view_zoom + WIDTH/2 - view_offset_x * view_zoom
    screen_y = y * view_zoom + HEIGHT/2 - view_offset_y * view_zoom
    return int(screen_x), int(screen_y)

# Function to convert screen coordinates to world coordinates
def screen_to_world(screen_x, screen_y):
    # Convert from screen coordinates to world coordinates (seed at 0,0)
    world_x = (screen_x - WIDTH/2 + view_offset_x * view_zoom) / view_zoom
    world_y = (screen_y - HEIGHT/2 + view_offset_y * view_zoom) / view_zoom
    return world_x, world_y

# Function to draw the DLA structure
def draw_dla():
    global walkers, view_offset_x, view_offset_y, view_zoom
    screen.fill(BLACK)
    
    # Draw all particles
    for x, y in particles:
        # Convert from grid coordinates to world coordinates (relative to center)
        world_x = x - WIDTH/2
        world_y = y - HEIGHT/2
        screen_x, screen_y = world_to_screen(world_x, world_y)
        radius = max(1, int(PARTICLE_RADIUS * view_zoom))  # Ensure minimum radius of 1
        pygame.draw.circle(screen, WHITE, (screen_x, screen_y), radius)
    
    # Draw the seed particle
    screen_x, screen_y = world_to_screen(0, 0)  # Seed is at world origin
    radius = max(1, int(SEED_RADIUS * view_zoom))  # Ensure minimum radius of 1
    pygame.draw.circle(screen, RED, (screen_x, screen_y), radius)
    
    # Draw all walkers
    for walker in walkers:
        # Convert from grid coordinates to world coordinates (relative to center)
        world_x = walker[0] - WIDTH/2
        world_y = walker[1] - HEIGHT/2
        screen_x, screen_y = world_to_screen(world_x, world_y)
        radius = max(1, int(WALKER_RADIUS * view_zoom))  # Ensure minimum radius of 1
        pygame.draw.circle(screen, GREEN, (screen_x, screen_y), radius)
    
    # Draw crosshair in the center of the screen
    crosshair_size = 10  # Size of the crosshair in pixels
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    # Horizontal line
    pygame.draw.line(screen, (100, 100, 100), 
                    (center_x - crosshair_size, center_y), 
                    (center_x + crosshair_size, center_y), 1)
    # Vertical line
    pygame.draw.line(screen, (100, 100, 100), 
                    (center_x, center_y - crosshair_size), 
                    (center_x, center_y + crosshair_size), 1)
    
    # Add debug text to show counts and zoom level
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Particles: {len(particles)} Walkers: {len(walkers)} Zoom: {view_zoom:.2f}x", True, GREEN)
    screen.blit(text, (10, 10))
    pygame.display.flip()

def run_simulation():
    global particles, walkers, view_zoom, view_offset_x, view_offset_y, is_panning, last_mouse_pos
    
    if DEBUG:
        print("Starting simulation...")
    clock = pygame.time.Clock()
    running = True
    particle_count = 1  # Start with the seed particle
    frame_count = 0
    
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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    is_panning = True
                    last_mouse_pos = event.pos
                elif event.button == 4:  # Mouse wheel up
                    # Zoom in at crosshair position (center of view)
                    center_x, center_y = WIDTH // 2, HEIGHT // 2
                    center_world_x, center_world_y = screen_to_world(center_x, center_y)
                    old_zoom = view_zoom
                    view_zoom = min(view_zoom * ZOOM_FACTOR, MAX_ZOOM)
                    # Calculate new center position after zoom
                    new_center_world_x, new_center_world_y = screen_to_world(center_x, center_y)
                    # Adjust offset to keep center position fixed
                    view_offset_x += (new_center_world_x - center_world_x)
                    view_offset_y += (new_center_world_y - center_world_y)
                elif event.button == 5:  # Mouse wheel down
                    # Zoom out at crosshair position (center of view)
                    center_x, center_y = WIDTH // 2, HEIGHT // 2
                    center_world_x, center_world_y = screen_to_world(center_x, center_y)
                    old_zoom = view_zoom
                    view_zoom = max(view_zoom / ZOOM_FACTOR, MIN_ZOOM)
                    # Calculate new center position after zoom
                    new_center_world_x, new_center_world_y = screen_to_world(center_x, center_y)
                    # Adjust offset to keep center position fixed
                    view_offset_x += (new_center_world_x - center_world_x)
                    view_offset_y += (new_center_world_y - center_world_y)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click release
                    is_panning = False
            elif event.type == pygame.MOUSEMOTION:
                if is_panning and last_mouse_pos is not None:
                    # Calculate the movement in world coordinates
                    dx = (event.pos[0] - last_mouse_pos[0]) / view_zoom
                    dy = (event.pos[1] - last_mouse_pos[1]) / view_zoom
                    view_offset_x -= dx
                    view_offset_y -= dy
                    last_mouse_pos = event.pos
        
        # Move all walkers
        walkers = move_walkers(walkers)
        
        # Check for contacts in batches
        for i in range(0, len(walkers), BATCH_SIZE):
            batch = walkers[i:i+BATCH_SIZE]
            contacts = check_contacts(batch)
            
            # Handle contacts
            for j, is_contact in enumerate(contacts):
                if is_contact:
                    idx = i + j
                    x, y = walkers[idx]
                    grid[x, y] = True
                    particles.append((x, y))
                    particle_count += 1
                    if DEBUG:
                        print(f"Particle added at ({x}, {y}). Total particles: {particle_count}")
                    
                    # Replace this walker with a new one
                    walkers[idx] = generate_walker()
        
        # Draw only every N frames
        frame_count += 1
        if frame_count % DRAW_FREQUENCY == 0:
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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    is_panning = True
                    last_mouse_pos = event.pos
                elif event.button == 4:  # Mouse wheel up
                    # Zoom in at crosshair position (center of view)
                    center_x, center_y = WIDTH // 2, HEIGHT // 2
                    center_world_x, center_world_y = screen_to_world(center_x, center_y)
                    old_zoom = view_zoom
                    view_zoom = min(view_zoom * ZOOM_FACTOR, MAX_ZOOM)
                    # Calculate new center position after zoom
                    new_center_world_x, new_center_world_y = screen_to_world(center_x, center_y)
                    # Adjust offset to keep center position fixed
                    view_offset_x += (new_center_world_x - center_world_x)
                    view_offset_y += (new_center_world_y - center_world_y)
                elif event.button == 5:  # Mouse wheel down
                    # Zoom out at crosshair position (center of view)
                    center_x, center_y = WIDTH // 2, HEIGHT // 2
                    center_world_x, center_world_y = screen_to_world(center_x, center_y)
                    old_zoom = view_zoom
                    view_zoom = max(view_zoom / ZOOM_FACTOR, MIN_ZOOM)
                    # Calculate new center position after zoom
                    new_center_world_x, new_center_world_y = screen_to_world(center_x, center_y)
                    # Adjust offset to keep center position fixed
                    view_offset_x += (new_center_world_x - center_world_x)
                    view_offset_y += (new_center_world_y - center_world_y)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click release
                    is_panning = False
            elif event.type == pygame.MOUSEMOTION:
                if is_panning and last_mouse_pos is not None:
                    dx = (event.pos[0] - last_mouse_pos[0]) / view_zoom
                    dy = (event.pos[1] - last_mouse_pos[1]) / view_zoom
                    view_offset_x -= dx
                    view_offset_y -= dy
                    last_mouse_pos = event.pos
        
        draw_dla()
        clock.tick(30)
    
    if DEBUG:
        print("Simulation ended")

# Initialize walkers
for i in range(NUM_WALKERS):
    walkers[i] = generate_walker()
if DEBUG:
    print(f"Created {NUM_WALKERS} initial walkers")

# Run the simulation
if __name__ == "__main__":
    run_simulation()
    pygame.quit()
    sys.exit() 