import pygame
import numpy as np
import random
import sys
import time
import json
import os
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Diffusion-Limited Aggregation")

# Debug flag
DEBUG = True

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
menu_open = False  # Flag to track if the menu is open

# Square size for particles and walkers
SQUARE_SIZE = 2  # Size of squares in pixels

# Print debug information
if DEBUG:
    print(f"Window initialized with dimensions: {WIDTH}x{HEIGHT}")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)  # Added yellow color for walkers

# DLA parameters
MAX_PARTICLES = 50000
STICKING_PROBABILITY = 1.0  # Probability of sticking when in contact
WALKER_SPEED = 4  # Pixels per step
NUM_WALKERS = 2000  # Number of walkers to simulate simultaneously
ATTRACTION_STRENGTH = 0.0  # Strength of the attraction force (0.0 to 1.0)
EDGE_GENERATION = False  # Whether to generate walkers at the edges of the screen

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

    # Apply WALKER_SPEED to the random movements
    random_movements = random_movements * WALKER_SPEED
    
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

# Function to save the current simulation state
def save_simulation():
    # Create a dictionary with the current simulation state
    # Convert all NumPy types to Python native types
    state = {
        "particles": [(int(x), int(y)) for x, y in particles],  # Convert to Python ints
        "walkers": [[int(x), int(y)] for x, y in walkers],  # Convert to Python ints
        "grid": grid.tolist(),  # Convert NumPy array to list
        "view_zoom": float(view_zoom),  # Convert to Python float
        "view_offset_x": float(view_offset_x),  # Convert to Python float
        "view_offset_y": float(view_offset_y),  # Convert to Python float
        "particle_count": len(particles)
    }
    
    # Open a file dialog to save the file
    pygame.display.set_mode((WIDTH, HEIGHT))  # Ensure the window is active
    pygame.display.set_caption("Save Simulation")
    
    # Create a simple file dialog
    font = pygame.font.SysFont(None, 24)
    input_box = pygame.Rect(WIDTH//4, HEIGHT//2, WIDTH//2, 30)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_inactive
    active = False
    text = "simulation.json"
    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if save button is clicked
                save_button = pygame.Rect(WIDTH//2 - 50, HEIGHT//2 + 50, 100, 30)
                if save_button.collidepoint(event.pos):
                    done = True
                elif input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        done = True
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode
                elif event.key == pygame.K_ESCAPE:
                    return False
        
        screen.fill(BLACK)
        txt_surface = font.render(text, True, color)
        width = max(WIDTH//2, txt_surface.get_width()+10)
        input_box.w = width
        screen.blit(txt_surface, (input_box.x+5, input_box.y+5))
        pygame.draw.rect(screen, color, input_box, 2)
        
        # Draw save button
        save_button = pygame.Rect(WIDTH//2 - 50, HEIGHT//2 + 50, 100, 30)
        pygame.draw.rect(screen, GREEN, save_button)
        save_text = font.render("Save", True, BLACK)
        screen.blit(save_text, (save_button.centerx - save_text.get_width()//2, 
                               save_button.centery - save_text.get_height()//2))
        
        pygame.display.flip()
    
    # Save the file
    try:
        with open(text, 'w') as f:
            json.dump(state, f)
        if DEBUG:
            print(f"Simulation saved to {text}")
            print(f"Saved state: {len(particles)} particles, zoom: {view_zoom:.2f}x")
        return True
    except Exception as e:
        print(f"Error saving simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Function to load a simulation state
def load_simulation():
    # Open a file dialog to load the file
    pygame.display.set_mode((WIDTH, HEIGHT))  # Ensure the window is active
    pygame.display.set_caption("Load Simulation")
    
    # Create a simple file dialog
    font = pygame.font.SysFont(None, 24)
    input_box = pygame.Rect(WIDTH//4, HEIGHT//2, WIDTH//2, 30)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_inactive
    active = False
    text = "simulation.json"
    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if load button is clicked
                load_button = pygame.Rect(WIDTH//2 - 50, HEIGHT//2 + 50, 100, 30)
                if load_button.collidepoint(event.pos):
                    done = True
                elif input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        done = True
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode
                elif event.key == pygame.K_ESCAPE:
                    return False
        
        screen.fill(BLACK)
        txt_surface = font.render(text, True, color)
        width = max(WIDTH//2, txt_surface.get_width()+10)
        input_box.w = width
        screen.blit(txt_surface, (input_box.x+5, input_box.y+5))
        pygame.draw.rect(screen, color, input_box, 2)
        
        # Draw load button
        load_button = pygame.Rect(WIDTH//2 - 50, HEIGHT//2 + 50, 100, 30)
        pygame.draw.rect(screen, GREEN, load_button)
        load_text = font.render("Load", True, BLACK)
        screen.blit(load_text, (load_button.centerx - load_text.get_width()//2, 
                               load_button.centery - load_text.get_height()//2))
        
        pygame.display.flip()
    
    # Load the file
    try:
        if not os.path.exists(text):
            print(f"Error: File '{text}' does not exist")
            return False
            
        with open(text, 'r') as f:
            state = json.load(f)
        
        # Validate the loaded state
        required_keys = ["particles", "walkers", "grid", "view_zoom", "view_offset_x", "view_offset_y"]
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            print(f"Error: Loaded file is missing required keys: {missing_keys}")
            return False
        
        # Update the simulation state
        global particles, walkers, grid, view_zoom, view_offset_x, view_offset_y
        particles = state["particles"]
        walkers = np.array(state["walkers"])
        grid = np.array(state["grid"])
        view_zoom = state["view_zoom"]
        view_offset_x = state["view_offset_x"]
        view_offset_y = state["view_offset_y"]
        if DEBUG:
            print(f"Simulation loaded from {text}")
            print(f"Loaded state: {len(particles)} particles, zoom: {view_zoom:.2f}x")
        return True
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file '{text}': {str(e)}")
        return False
    except Exception as e:
        print(f"Error loading simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Function to quit the application
def quit_application():
    """Properly quit the pygame application"""
    global running
    running = False
    pygame.quit()
    sys.exit()

# Menu state
menu_open = False

class MenuItem:
    def __init__(self, text, action):
        self.text = text
        self.action = action
        self.rect = None
        self.hover = False

# Menu items
menu_items = [
    MenuItem("Save", lambda: save_simulation()),
    MenuItem("Load", lambda: load_simulation()),
    MenuItem("Quit", lambda: quit_application())
]

def draw_menu(screen):
    global menu_open
    MENU_HEIGHT = 30  # Fixed menu height as a constant
    
    # Draw menu bar
    pygame.draw.rect(screen, (50, 50, 50), (0, 0, WIDTH, MENU_HEIGHT))
    pygame.draw.rect(screen, (100, 100, 100), (10, 5, 60, 20))
    font = pygame.font.Font(None, 24)
    text = font.render("Menu", True, WHITE)
    screen.blit(text, (15, 8))
    
    if menu_open:
        # Draw menu items
        y = 35
        for item in menu_items:
            # Draw item background
            color = (70, 70, 70) if item.hover else (50, 50, 50)
            item.rect = pygame.draw.rect(screen, color, (10, y, 100, 25))
            # Draw item text
            text = font.render(item.text, True, WHITE)
            screen.blit(text, (15, y + 5))
            y += 30
    
    return MENU_HEIGHT  # Return the constant menu height

def handle_menu_click(pos):
    global menu_open
    # Check if menu button is clicked
    if 10 <= pos[0] <= 70 and 5 <= pos[1] <= 25:
        menu_open = not menu_open
        if DEBUG:
            print(f"Menu {'opened' if menu_open else 'closed'}")
        return True
    
    # If menu is open, check for item clicks
    if menu_open:
        for item in menu_items:
            if item.rect and item.rect.collidepoint(pos):
                if DEBUG:
                    print(f"Menu item '{item.text}' clicked")
                item.action()
                menu_open = False
                return True
        
        # If clicked outside menu items, close menu
        menu_open = False
        if DEBUG:
            print("Menu closed (clicked outside)")
        return True
    
    return False

# Function to draw the DLA structure
def draw_dla():
    global walkers, view_offset_x, view_offset_y, view_zoom, menu_open
    screen.fill(BLACK)
    
    # Draw menu bar and get menu height
    menu_height = draw_menu(screen)
    
    # Calculate the scaled square size based on zoom
    scaled_size = max(1, int(SQUARE_SIZE * view_zoom))
    
    # Draw all particles
    for x, y in particles:
        # Convert from grid coordinates to world coordinates (relative to center)
        world_x = x - WIDTH/2
        world_y = y - HEIGHT/2
        screen_x, screen_y = world_to_screen(world_x, world_y)
        # Draw a square
        pygame.draw.rect(screen, GREEN, (screen_x - scaled_size//2, screen_y - scaled_size//2, scaled_size, scaled_size))
    
    # Draw the seed particle
    screen_x, screen_y = world_to_screen(0, 0)  # Seed is at world origin
    # Draw a square for the seed
    pygame.draw.rect(screen, RED, (screen_x - scaled_size//2, screen_y - scaled_size//2, scaled_size, scaled_size))
    
    # Draw all walkers
    for walker in walkers:
        # Convert from grid coordinates to world coordinates (relative to center)
        world_x = walker[0] - WIDTH/2
        world_y = walker[1] - HEIGHT/2
        screen_x, screen_y = world_to_screen(world_x, world_y)
        # Draw a square for each walker
        pygame.draw.rect(screen, YELLOW, (screen_x - scaled_size//2, screen_y - scaled_size//2, scaled_size, scaled_size))
    
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
    
    # Add debug text to show counts and zoom level - moved to right side
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Particles: {len(particles)} Walkers: {len(walkers)} Zoom: {view_zoom:.2f}x", True, GREEN)
    # Position text on the right side with a small margin
    text_x = WIDTH - text.get_width() - 10
    screen.blit(text, (text_x, menu_height + 10))
    
    # Display message if max particles reached - also moved to right side
    if 'max_particles_reached' in globals() and max_particles_reached:
        max_text = font.render(f"Maximum particles ({MAX_PARTICLES}) reached!", True, RED)
        max_text_x = WIDTH - max_text.get_width() - 10
        screen.blit(max_text, (max_text_x, menu_height + 40))
    
    pygame.display.flip()

def run_simulation():
    global particles, walkers, view_zoom, view_offset_x, view_offset_y, is_panning, last_mouse_pos, menu_open, running
    
    if DEBUG:
        print("Starting simulation...")
    clock = pygame.time.Clock()
    running = True
    particle_count = 1  # Start with the seed particle
    frame_count = 0
    menu_open = False
    max_particles_reached = False
    
    if DEBUG:
        print("Entering main simulation loop...")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_application()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_application()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if handle_menu_click(event.pos):
                        if not running:  # If quit was called, exit the loop
                            return
                        continue
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
        
        # Only move walkers and check for contacts if max particles not reached
        if not max_particles_reached:
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
                        
                        # Replace this walker with a new one
                        walkers[idx] = generate_walker()
                        
                        # Check if max particles reached
                        if particle_count >= MAX_PARTICLES:
                            max_particles_reached = True
                            if DEBUG:
                                print(f"Maximum number of particles ({MAX_PARTICLES}) reached!")
                            break
                
                if max_particles_reached:
                    break
        else:
            # If max particles reached, just move walkers without checking for contacts
            walkers = move_walkers(walkers)
        
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
                quit_application()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_application()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if handle_menu_click(event.pos):
                        continue
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
        
        # If max particles reached, just move walkers without checking for contacts
        if max_particles_reached:
            walkers = move_walkers(walkers)
        
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
    quit_application() 