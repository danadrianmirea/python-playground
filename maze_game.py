import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 30
MAZE_WIDTH = 20
MAZE_HEIGHT = 20
SCREEN_WIDTH = MAZE_WIDTH * CELL_SIZE
SCREEN_HEIGHT = MAZE_HEIGHT * CELL_SIZE

# Colors
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Maze Game")

# Player properties
player_x = CELL_SIZE // 2  # Start in top-left cell
player_y = CELL_SIZE // 2
player_size = 20
player_speed = 5

# Create a simple maze (1 represents walls, 0 represents paths)
maze = [[1 for x in range(MAZE_WIDTH)] for y in range(MAZE_HEIGHT)]

def check_collision(x, y):
    cell_x = int(x // CELL_SIZE)
    cell_y = int(y // CELL_SIZE)
    return (0 <= cell_x < MAZE_WIDTH and 
            0 <= cell_y < MAZE_HEIGHT and 
            maze[cell_y][cell_x] == 0)

# Create a simple path (this is a very basic maze for prototype)
def create_simple_maze():
    # Start with all walls
    # Create a basic path from top left to right side
    current_x = 0
    current_y = 0
    maze[current_y][current_x] = 0  # Start position
    
    # Create random path to right side
    while current_x < MAZE_WIDTH - 1:
        if current_y > 0 and random.random() < 0.5:
            maze[current_y-1][current_x] = 0  # Create up path
            current_y -= 1
        elif current_y < MAZE_HEIGHT - 1 and random.random() < 0.5:
            maze[current_y+1][current_x] = 0  # Create down path
            current_y += 1
        else:
            maze[current_y][current_x+1] = 0  # Create right path
            current_x += 1
    
    # Ensure exit on right side
    maze[current_y][MAZE_WIDTH-1] = 0

create_simple_maze()

# Game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Handle player movement
    keys = pygame.key.get_pressed()
    new_x = player_x
    new_y = player_y
    
    # Calculate movement direction
    dx = 0
    dy = 0
    
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        dx -= 1
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        dx += 1
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        dy -= 1
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        dy += 1
    
    # Normalize diagonal movement
    if dx != 0 and dy != 0:
        # Normalize the diagonal movement to maintain consistent speed
        dx = dx * player_speed / (2 ** 0.5)
        dy = dy * player_speed / (2 ** 0.5)
    else:
        dx = dx * player_speed
        dy = dy * player_speed
    
    new_x += dx
    new_y += dy
    
    # Check multiple points around the player's bounds
    half_size = player_size // 2
    can_move = True
    
    # Check corners and center
    points_to_check = [
        (new_x, new_y),  # center
        (new_x - half_size, new_y - half_size),  # top-left
        (new_x + half_size, new_y - half_size),  # top-right
        (new_x - half_size, new_y + half_size),  # bottom-left
        (new_x + half_size, new_y + half_size),  # bottom-right
    ]
    
    for check_x, check_y in points_to_check:
        if not check_collision(check_x, check_y):
            can_move = False
            break
    
    if can_move:
        player_x = new_x
        player_y = new_y
    
    # Keep player within screen bounds
    player_x = max(player_size//2, min(player_x, SCREEN_WIDTH - player_size//2))
    player_y = max(player_size//2, min(player_y, SCREEN_HEIGHT - player_size//2))
    
    # Drawing
    screen.fill(BLACK)
    
    # Draw maze
    for y in range(MAZE_HEIGHT):
        for x in range(MAZE_WIDTH):
            if maze[y][x] == 1:
                pygame.draw.rect(screen, WHITE, 
                               (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw player
    pygame.draw.rect(screen, RED, 
                    (player_x - player_size//2, player_y - player_size//2, 
                     player_size, player_size))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
