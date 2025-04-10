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
view_offset_x = WIDTH // 2
view_offset_y = HEIGHT // 2
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
PARTICLE_RADIUS = 2
SEED_RADIUS = 3
WALKER_RADIUS = 2

# DLA parameters
MAX_PARTICLES = 5000
STICKING_PROBABILITY = 1.0  # Probability of sticking when in contact
WALKER_SPEED = 2  # Pixels per step
NUM_WALKERS = 1000  # Number of walkers to simulate simultaneously
ATTRACTION_STRENGTH = 0.6  # Strength of the attraction force (0.0 to 1.0)

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
    screen_x = (x - view_offset_x) * view_zoom + WIDTH/2
    screen_y = (y - view_offset_y) * view_zoom + HEIGHT/2
    return int(screen_x), int(screen_y)

# Function to convert screen coordinates to world coordinates
def screen_to_world(screen_x, screen_y):
    world_x = (screen_x - WIDTH/2) / view_zoom + view_offset_x
    world_y = (screen_y - HEIGHT/2) / view_zoom + view_offset_y
    return world_x, world_y

# Function to draw the DLA structure
def draw_dla():
    global walkers, view_offset_x, view_offset_y, view_zoom
    screen.fill(BLACK)
    
    # Draw a grid to help visualize the coordinate system
    grid_spacing = 50 * view_zoom
    start_x = int((-WIDTH/2 - view_offset_x * view_zoom) / grid_spacing) * grid_spacing
    start_y = int((-HEIGHT/2 - view_offset_y * view_zoom) / grid_spacing) * grid_spacing
    end_x = int((WIDTH/2 - view_offset_x * view_zoom) / grid_spacing) * grid_spacing
    end_y = int((HEIGHT/2 - view_offset_y * view_zoom) / grid_spacing) * grid_spacing
    
    # Draw vertical grid lines
    x = start_x
    while x <= end_x:
        screen_x1, screen_y1 = world_to_screen(x/view_zoom, start_y/view_zoom)
        screen_x2, screen_y2 = world_to_screen(x/view_zoom, end_y/view_zoom)
        pygame.draw.line(screen, (30, 30, 30), (screen_x1, screen_y1), (screen_x2, screen_y2))
        x += grid_spacing
    
    # Draw horizontal grid lines
    y = start_y
    while y <= end_y:
        screen_x1, screen_y1 = world_to_screen(start_x/view_zoom, y/view_zoom)
        screen_x2, screen_y2 = world_to_screen(end_x/view_zoom, y/view_zoom)
        pygame.draw.line(screen, (30, 30, 30), (screen_x1, screen_y1), (screen_x2, screen_y2))
        y += grid_spacing
    
    # Draw all particles
    for x, y in particles:
        screen_x, screen_y = world_to_screen(x, y)
        radius = int(PARTICLE_RADIUS * view_zoom)
        pygame.draw.circle(screen, WHITE, (screen_x, screen_y), radius)
    
    # Draw the seed particle
    screen_x, screen_y = world_to_screen(seed_x, seed_y)
    radius = int(SEED_RADIUS * view_zoom)
    pygame.draw.circle(screen, RED, (screen_x, screen_y), radius)
    
    # Draw all walkers
    for walker in walkers:
        screen_x, screen_y = world_to_screen(walker[0], walker[1])
        radius = int(WALKER_RADIUS * view_zoom)
        pygame.draw.circle(screen, GREEN, (screen_x, screen_y), radius)
    
    # Add debug text to show counts and zoom level
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Particles: {len(particles)} Walkers: {len(walkers)} Zoom: {view_zoom:.2f}x", True, GREEN)
    screen.blit(text, (10, 10))
    
    # Draw center crosshair
    screen_x1, screen_y1 = world_to_screen(0, -HEIGHT/2)
    screen_x2, screen_y2 = world_to_screen(0, HEIGHT/2)
    screen_x3, screen_y3 = world_to_screen(-WIDTH/2, 0)
    screen_x4, screen_y4 = world_to_screen(WIDTH/2, 0)
    pygame.draw.line(screen, (100, 100, 100), (screen_x1, screen_y1), (screen_x2, screen_y2), 1)
    pygame.draw.line(screen, (100, 100, 100), (screen_x3, screen_y3), (screen_x4, screen_y4), 1)
    
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
                    # Zoom in at mouse position
                    mouse_x, mouse_y = screen_to_world(*event.pos)
                    view_zoom = min(view_zoom * ZOOM_FACTOR, MAX_ZOOM)
                    new_mouse_x, new_mouse_y = screen_to_world(*event.pos)
                    view_offset_x += (new_mouse_x - mouse_x)
                    view_offset_y += (new_mouse_y - mouse_y)
                elif event.button == 5:  # Mouse wheel down
                    # Zoom out at mouse position
                    mouse_x, mouse_y = screen_to_world(*event.pos)
                    view_zoom = max(view_zoom / ZOOM_FACTOR, MIN_ZOOM)
                    new_mouse_x, new_mouse_y = screen_to_world(*event.pos)
                    view_offset_x += (new_mouse_x - mouse_x)
                    view_offset_y += (new_mouse_y - mouse_y)
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
                    mouse_x, mouse_y = screen_to_world(*event.pos)
                    view_zoom = min(view_zoom * ZOOM_FACTOR, MAX_ZOOM)
                    new_mouse_x, new_mouse_y = screen_to_world(*event.pos)
                    view_offset_x += (new_mouse_x - mouse_x)
                    view_offset_y += (new_mouse_y - mouse_y)
                elif event.button == 5:  # Mouse wheel down
                    mouse_x, mouse_y = screen_to_world(*event.pos)
                    view_zoom = max(view_zoom / ZOOM_FACTOR, MIN_ZOOM)
                    new_mouse_x, new_mouse_y = screen_to_world(*event.pos)
                    view_offset_x += (new_mouse_x - mouse_x)
                    view_offset_y += (new_mouse_y - mouse_y)
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