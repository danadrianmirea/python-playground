import pygame
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FOV = math.pi / 2  # Field of view
HALF_FOV = FOV / 2
MAX_DEPTH = 1000
CELL_SIZE = 64
PLAYER_SPEED = 10
PLAYER_ROT_SPEED = 0.1

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("3D Maze")

# Maze definition (0 = empty, 1 = wall)
MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

noclip = True

class Player:
    def __init__(self):
        self.x = CELL_SIZE * 1.5 
        self.y = CELL_SIZE * 1.5
        self.angle = 0

    def move(self, dx, dy):
        new_x = self.x + dx
        new_y = self.y + dy

        if noclip:
            self.x = new_x
            self.y = new_y
        elif (0 <= new_x < len(MAZE[0]) * CELL_SIZE and 
             0 <= new_y < len(MAZE) * CELL_SIZE and
             MAZE[int(new_y // CELL_SIZE)][int(new_x // CELL_SIZE)] == 0):
             self.x = new_x
             self.y = new_y

    def rotate(self, angle):
        self.angle += angle

def cast_ray(angle, player):
    # Ray starting point
    ray_x = player.x
    ray_y = player.y
    
    # Ray direction
    ray_cos = math.cos(angle)
    ray_sin = math.sin(angle)
    
    # Ray step size
    step_size = 1
    
    # Cast ray until hitting a wall or reaching max depth
    for distance in range(MAX_DEPTH):
        ray_x += ray_cos * step_size
        ray_y += ray_sin * step_size
        
        # Check if ray hit a wall
        map_x = int(ray_x // CELL_SIZE)
        map_y = int(ray_y // CELL_SIZE)
        
        if (0 <= map_x < len(MAZE[0]) and 
            0 <= map_y < len(MAZE) and 
            MAZE[map_y][map_x] == 1):
            return distance
    
    return MAX_DEPTH

def draw_3d_view(player):
    screen.fill(BLACK)
    
    # Draw ceiling
    pygame.draw.rect(screen, BLUE, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT // 2))
    # Draw floor
    pygame.draw.rect(screen, GRAY, (0, SCREEN_HEIGHT // 2, SCREEN_WIDTH, SCREEN_HEIGHT // 2))
    
    # Cast rays
    for x in range(SCREEN_WIDTH):
        # Calculate ray angle
        ray_angle = player.angle - HALF_FOV + (x / SCREEN_WIDTH) * FOV
        
        # Cast ray and get distance
        distance = cast_ray(ray_angle, player)
        
        # Calculate wall height
        wall_height = min(SCREEN_HEIGHT, (CELL_SIZE * SCREEN_HEIGHT) / (distance + 0.0001))
        
        # Calculate wall top and bottom
        wall_top = (SCREEN_HEIGHT - wall_height) // 2
        wall_bottom = wall_top + wall_height
        
        # Draw wall slice
        if distance < MAX_DEPTH:
            # Calculate wall color based on distance with a minimum intensity
            color_intensity = min(255, max(50, 255 - (distance)))  # Reduced multiplier and added minimum
            wall_color = (color_intensity, color_intensity, color_intensity)
            pygame.draw.line(screen, wall_color, (x, wall_top), (x, wall_bottom))

def draw_debug_info(player):
    # Calculate maze cell coordinates
    cell_x = int(player.x // CELL_SIZE)
    cell_y = int(player.y // CELL_SIZE)
    
    # Create debug text
    debug_text = f"Player: ({player.x:.1f}, {player.y:.1f})"
    cell_text = f"Cell: [{cell_x}, {cell_y}]"
    
    # Create font
    font = pygame.font.Font(None, 24)
    
    # Render text
    debug_surface = font.render(debug_text, True, WHITE)
    cell_surface = font.render(cell_text, True, WHITE)
    
    # Draw text
    screen.blit(debug_surface, (10, 10))
    screen.blit(cell_surface, (10, 40))

def main():
    player = Player()
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Handle player movement
        keys = pygame.key.get_pressed()
        
        # Forward/Backward movement
        if keys[pygame.K_w]:
            player.move(math.cos(player.angle) * PLAYER_SPEED, math.sin(player.angle) * PLAYER_SPEED)
        if keys[pygame.K_s]:
            player.move(-math.cos(player.angle) * PLAYER_SPEED, -math.sin(player.angle) * PLAYER_SPEED)
        
        # Strafe movement
        if keys[pygame.K_q]:
            player.move(math.cos(player.angle - math.pi/2) * PLAYER_SPEED, 
                       math.sin(player.angle - math.pi/2) * PLAYER_SPEED)
        if keys[pygame.K_e]:
            player.move(math.cos(player.angle + math.pi/2) * PLAYER_SPEED,
                       math.sin(player.angle + math.pi/2) * PLAYER_SPEED)
        
        # Rotation
        if keys[pygame.K_a]:
            player.rotate(-PLAYER_ROT_SPEED)
        if keys[pygame.K_d]:
            player.rotate(PLAYER_ROT_SPEED)
        
        # Draw the 3D view
        draw_3d_view(player)
        
        # Draw debug info
        draw_debug_info(player)
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main() 