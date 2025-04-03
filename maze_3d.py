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
PLAYER_SPEED = 200
PLAYER_ROT_SPEED = 2.5

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
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

noclip = False

class Player:
    def __init__(self):
        self.x = CELL_SIZE * 1.5 
        self.y = CELL_SIZE * 1.5
        self.angle = 0
        self.width = CELL_SIZE * 0.25  # Player width is 1/4 of a cell

    def move(self, dx, dy, delta_time):
        # Scale movement by delta time
        new_x = self.x + dx * delta_time
        new_y = self.y + dy * delta_time

        if noclip:
            self.x = new_x
            self.y = new_y
        else:
            # Check collisions at multiple points around the player
            # We'll check 8 points in a circle around the player
            collision_points = [
                (new_x, new_y),  # Center
                (new_x + self.width, new_y),  # Right
                (new_x - self.width, new_y),  # Left
                (new_x, new_y + self.width),  # Down
                (new_x, new_y - self.width),  # Up
                (new_x + self.width * 0.707, new_y + self.width * 0.707),  # Bottom right
                (new_x - self.width * 0.707, new_y + self.width * 0.707),  # Bottom left
                (new_x + self.width * 0.707, new_y - self.width * 0.707),  # Top right
                (new_x - self.width * 0.707, new_y - self.width * 0.707),  # Top left
            ]

            # Check if any point would collide with a wall
            can_move = True
            for point_x, point_y in collision_points:
                map_x = int(point_x // CELL_SIZE)
                map_y = int(point_y // CELL_SIZE)
                
                if not (0 <= map_x < len(MAZE[0]) and 
                       0 <= map_y < len(MAZE) and 
                       MAZE[map_y][map_x] == 0):
                    can_move = False
                    break

            if can_move:
                self.x = new_x
                self.y = new_y

    def rotate(self, angle, delta_time):
        self.angle += angle * delta_time

def cast_ray(angle, player):
    # Ray starting point
    ray_x = player.x
    ray_y = player.y
    
    # Ray direction
    ray_cos = math.cos(angle)
    ray_sin = math.sin(angle)
    
    # Use a larger step size for better performance
    step_size = 4
    
    # Cast ray until hitting a wall or reaching max depth
    for distance in range(0, MAX_DEPTH, step_size):
        ray_x += ray_cos * step_size
        ray_y += ray_sin * step_size
        
        # Check if ray hit a wall
        map_x = int(ray_x // CELL_SIZE)
        map_y = int(ray_y // CELL_SIZE)
        
        if (0 <= map_x < len(MAZE[0]) and 
            0 <= map_y < len(MAZE) and 
            MAZE[map_y][map_x] == 1):
            # Fine-tune the distance for more accurate wall placement
            return distance + (step_size // 2)
    
    return MAX_DEPTH

def draw_3d_view(player):
    screen.fill(BLACK)
    
    # Draw ceiling
    pygame.draw.rect(screen, BLUE, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT // 2))
    # Draw floor
    pygame.draw.rect(screen, GRAY, (0, SCREEN_HEIGHT // 2, SCREEN_WIDTH, SCREEN_HEIGHT // 2))
    
    # Cast rays and draw walls
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
            # Calculate wall color based on distance
            color_intensity = min(255, max(50, 255 - (distance * 0.3)))
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
        # Get delta time in seconds
        delta_time = clock.get_rawtime() / 1000.0  # Convert milliseconds to seconds
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Handle player movement
        keys = pygame.key.get_pressed()
        
        # Forward/Backward movement
        if keys[pygame.K_w]:
            player.move(math.cos(player.angle) * PLAYER_SPEED, math.sin(player.angle) * PLAYER_SPEED, delta_time)
        if keys[pygame.K_s]:
            player.move(-math.cos(player.angle) * PLAYER_SPEED, -math.sin(player.angle) * PLAYER_SPEED, delta_time)
        
        # Strafe movement
        if keys[pygame.K_q]:
            player.move(math.cos(player.angle - math.pi/2) * PLAYER_SPEED, 
                       math.sin(player.angle - math.pi/2) * PLAYER_SPEED, delta_time)
        if keys[pygame.K_e]:
            player.move(math.cos(player.angle + math.pi/2) * PLAYER_SPEED,
                       math.sin(player.angle + math.pi/2) * PLAYER_SPEED, delta_time)
        
        # Rotation
        if keys[pygame.K_a]:
            player.rotate(-PLAYER_ROT_SPEED, delta_time)
        if keys[pygame.K_d]:
            player.rotate(PLAYER_ROT_SPEED, delta_time)
        
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