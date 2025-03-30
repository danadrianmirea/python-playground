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

def is_solvable():
    # Use BFS to check if there's a path from start to end
    from collections import deque
    
    # Start from top-left, end at bottom-right
    start = (0, 0)
    end = (MAZE_WIDTH - 1, MAZE_HEIGHT - 1)
    
    if maze[start[1]][start[0]] == 1 or maze[end[1]][end[0]] == 1:
        return False
    
    queue = deque([start])
    visited = {start}
    
    while queue:
        current = queue.popleft()
        if current == end:
            return True
            
        x, y = current
        # Check all four directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            next_pos = (next_x, next_y)
            
            if (0 <= next_x < MAZE_WIDTH and 
                0 <= next_y < MAZE_HEIGHT and 
                maze[next_y][next_x] == 0 and 
                next_pos not in visited):
                queue.append(next_pos)
                visited.add(next_pos)
    
    return False

def generate_maze():
    max_attempts = 10  # Maximum number of attempts to generate a solvable maze
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        # Initialize maze with walls
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                maze[y][x] = 1
        
        # Use a stack for iterative approach
        stack = [(1, 1)]  # Start from top-left corner
        maze[1][1] = 0  # Carve starting point
        
        while stack:
            current = stack[-1]
            x, y = current
            
            # Define possible directions (up, right, down, left)
            directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
            random.shuffle(directions)
            
            # Try each direction
            found_path = False
            for dx, dy in directions:
                next_x, next_y = x + dx, y + dy
                # Check if the next cell is within bounds and is a wall
                if (0 <= next_x < MAZE_WIDTH and 
                    0 <= next_y < MAZE_HEIGHT and 
                    maze[next_y][next_x] == 1):
                    # Carve the cell between current and next
                    maze[y + dy//2][x + dx//2] = 0
                    maze[next_y][next_x] = 0
                    stack.append((next_x, next_y))
                    found_path = True
                    break
            
            # If no path found, backtrack
            if not found_path:
                stack.pop()
        
        # Ensure start and end points are accessible
        maze[0][0] = 0  # Start point
        maze[MAZE_HEIGHT-1][MAZE_WIDTH-1] = 0  # End point
        
        # Create a path from start to end if needed
        current_x, current_y = 0, 0
        while current_x < MAZE_WIDTH - 1 or current_y < MAZE_HEIGHT - 1:
            if current_x < MAZE_WIDTH - 1 and random.random() < 0.5:
                maze[current_y][current_x + 1] = 0
                current_x += 1
            elif current_y < MAZE_HEIGHT - 1:
                maze[current_y + 1][current_x] = 0
                current_y += 1
        
        # If maze is solvable, we're done
        if is_solvable():
            break
    
    # If we couldn't generate a solvable maze after max attempts,
    # create a simple path from start to end
    if attempt >= max_attempts:
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                maze[y][x] = 1
        # Create a simple path from start to end
        for x in range(MAZE_WIDTH):
            maze[0][x] = 0
        for y in range(MAZE_HEIGHT):
            maze[y][MAZE_WIDTH-1] = 0

# Generate the maze
generate_maze()

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
