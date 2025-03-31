import pygame
import random
import noise
import numpy as np
from typing import List, Tuple, Set
import math
import os

# Constants
INITIAL_MAZE_WIDTH = 23
INITIAL_MAZE_HEIGHT = 23
MIN_CELL_SIZE = 20  # Minimum cell size to maintain playability
MAX_CELL_SIZE = 40  # Maximum cell size to prevent too large cells
PLAYER_SIZE_RATIO = 0.7  # Player size as a ratio of cell size
NEXT_LEVEL_SIZE_INCREMENT = 6

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game settings
PLAYER_SPEED = 20

class MazeGame:
    def __init__(self):
        pygame.init()
        # Get the screen info to determine the display size
        screen_info = pygame.display.Info()
        self.screen_width = screen_info.current_w
        self.screen_height = screen_info.current_h
        
        # Calculate the maximum maze size that can fit on screen
        max_maze_width = (self.screen_width * 0.8) // MIN_CELL_SIZE  # Use 80% of screen width
        max_maze_height = (self.screen_height * 0.8) // MIN_CELL_SIZE  # Use 80% of screen height
        
        # Set maze dimensions to fit screen while maintaining aspect ratio
        self.maze_width = min(INITIAL_MAZE_WIDTH, int(max_maze_width))
        self.maze_height = min(INITIAL_MAZE_HEIGHT, int(max_maze_height))
        
        # Calculate cell size to fit the maze
        self.cell_size = min(
            int(self.screen_width * 0.8 / self.maze_width),
            int(self.screen_height * 0.8 / self.maze_height),
            MAX_CELL_SIZE
        )
        
        # Calculate window dimensions
        self.window_width = self.maze_width * self.cell_size
        self.window_height = self.maze_height * self.cell_size
        
        # Create the game window with maximized state
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Maze Game")
        
        # Center the window on the screen
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self):
        # Recalculate cell size to fit the new maze dimensions
        self.cell_size = min(
            int(self.screen_width * 0.8 / self.maze_width),
            int(self.screen_height * 0.8 / self.maze_height),
            MAX_CELL_SIZE
        )
        
        # Update window dimensions with new cell size
        self.window_width = self.maze_width * self.cell_size
        self.window_height = self.maze_height * self.cell_size
        
        # Create new window with updated dimensions
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        
        self.maze = self.generate_maze()
        self.player_pos = [1.0, 1.0]  # Starting position as floats
        self.exit_pos = [self.maze_height - 2, self.maze_width - 1]  # Exit position
        self.game_over = False
        
        # Movement tracking
        self.pressed_keys: Set[int] = set()
        self.last_update_time = 0

    def generate_maze(self) -> List[List[int]]:
        # Initialize maze with walls
        maze = [[1 for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        
        def carve_path(x: int, y: int):
            maze[y][x] = 0  # Carve current cell
            
            # Define possible directions (up, right, down, left)
            directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
            random.shuffle(directions)  # Randomize direction order
            
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                
                # Check if the new position is within bounds and is a wall
                if (0 < new_x < self.maze_width - 1 and 
                    0 < new_y < self.maze_height - 1 and 
                    maze[new_y][new_x] == 1):
                    # Carve the path between current and new position
                    maze[y + dy//2][x + dx//2] = 0
                    carve_path(new_x, new_y)
        
        # Start from (1,1) and carve paths
        carve_path(1, 1)
        
        # Ensure borders
        for y in range(self.maze_height):
            maze[y][0] = 1  # Left border
            maze[y][self.maze_width-1] = 1  # Right border
        for x in range(self.maze_width):
            maze[0][x] = 1  # Top border
            maze[self.maze_height-1][x] = 1  # Bottom border
        
        # Create exit
        maze[self.maze_height-2][self.maze_width-1] = 0
        
        # Add some random paths to make it more interesting
        for _ in range(self.maze_width * self.maze_height // 30):  # Add random paths
            x = random.randint(1, self.maze_width - 2)
            y = random.randint(1, self.maze_height - 2)
            if maze[y][x] == 1:  # If it's a wall
                # Check if we can create a path here
                can_create = False
                for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    next_y, next_x = y + dy, x + dx
                    if (0 <= next_y < self.maze_height and 
                        0 <= next_x < self.maze_width and 
                        maze[next_y][next_x] == 0):
                        can_create = True
                        break
                if can_create:
                    maze[y][x] = 0
        
        return maze

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw maze
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, WHITE,
                                   (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        # Draw player (using float positions)
        player_size = int(self.cell_size * PLAYER_SIZE_RATIO)
        player_offset = (self.cell_size - player_size) / 2
        pygame.draw.rect(self.screen, RED,
                        (self.player_pos[1] * self.cell_size + player_offset,
                         self.player_pos[0] * self.cell_size + player_offset,
                         player_size, player_size))

        # Draw exit
        pygame.draw.rect(self.screen, GREEN,
                        (self.exit_pos[1] * self.cell_size, self.exit_pos[0] * self.cell_size,
                         self.cell_size, self.cell_size))

        pygame.display.flip()

    def is_valid_position(self, x: float, y: float) -> bool:
        # Check if the position is within maze bounds
        if not (0 <= x < self.maze_width and 0 <= y < self.maze_height):
            return False
            
        # Get the grid cells the player is overlapping with, considering the smaller player size
        player_size = PLAYER_SIZE_RATIO
        left_cell = int(x)
        right_cell = min(int(x + player_size), self.maze_width - 1)
        top_cell = int(y)
        bottom_cell = min(int(y + player_size), self.maze_height - 1)
        
        # Check if any of the overlapping cells are walls
        return (self.maze[top_cell][left_cell] == 0 and 
                self.maze[top_cell][right_cell] == 0 and
                self.maze[bottom_cell][left_cell] == 0 and 
                self.maze[bottom_cell][right_cell] == 0)

    def move_player(self, dx: float, dy: float):
        # Calculate new position
        new_x = self.player_pos[1] + dx
        new_y = self.player_pos[0] + dy
        
        # Try to move horizontally first
        if self.is_valid_position(new_x, self.player_pos[0]):
            self.player_pos[1] = new_x
        else:
            # Snap to grid horizontally if collision
            self.player_pos[1] = round(self.player_pos[1])
            
        # Then try to move vertically
        if self.is_valid_position(self.player_pos[1], new_y):
            self.player_pos[0] = new_y
        else:
            # Snap to grid vertically if collision
            self.player_pos[0] = round(self.player_pos[0])
            
        # Check if player reached exit (using rounded position for exit check)
        if (round(self.player_pos[0]) == self.exit_pos[0] and 
            round(self.player_pos[1]) == self.exit_pos[1]):
            self.game_over = True

    def handle_movement(self):
        dx = 0
        dy = 0
        
        # Arrow keys
        if pygame.K_LEFT in self.pressed_keys or pygame.K_a in self.pressed_keys:
            dx -= 1
        if pygame.K_RIGHT in self.pressed_keys or pygame.K_d in self.pressed_keys:
            dx += 1
        if pygame.K_UP in self.pressed_keys or pygame.K_w in self.pressed_keys:
            dy -= 1
        if pygame.K_DOWN in self.pressed_keys or pygame.K_s in self.pressed_keys:
            dy += 1
            
        # Normalize diagonal movement
        if dx != 0 and dy != 0:
            magnitude = math.sqrt(dx * dx + dy * dy)
            dx = dx / magnitude
            dy = dy / magnitude
            
        if dx != 0 or dy != 0:
            # Use fixed movement speed instead of delta time for more consistent movement
            movement = PLAYER_SPEED / 60.0  # Divide by FPS for consistent speed
            self.move_player(dx * movement, dy * movement)

    def run(self):
        running = True
        while running:
            current_time = pygame.time.get_ticks()
            
            # Handle movement based on time
            if self.pressed_keys and not self.game_over:
                self.handle_movement()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.game_over:
                        # Reset game with larger maze, increasing both dimensions by 2 to maintain odd numbers
                        self.maze_width += NEXT_LEVEL_SIZE_INCREMENT
                        self.maze_height += NEXT_LEVEL_SIZE_INCREMENT
                        self.reset_game()
                    else:
                        self.pressed_keys.add(event.key)
                        self.handle_movement()
                elif event.type == pygame.KEYUP:
                    self.pressed_keys.discard(event.key)

            self.draw()
            
            if self.game_over:
                # Calculate font size based on window dimensions
                base_font_size = min(self.window_width, self.window_height) // 20
                font = pygame.font.Font(None, base_font_size)
                
                # Split message into two lines
                textColor = WHITE
                text1 = font.render('You Win!', True, textColor)
                text2 = font.render('Press any key to continue', True, textColor)
                
                # Calculate text dimensions and padding
                padding = base_font_size // 2
                text_width = max(text1.get_width(), text2.get_width())
                text_height = text1.get_height() + text2.get_height() + padding
                
                # Create semi-transparent black surface for background
                background = pygame.Surface((text_width + padding * 2, text_height + padding * 2))
                background.fill(BLACK)
                background.set_alpha(255)  # Semi-transparent
                
                # Position background and text in center of screen
                background_rect = background.get_rect(center=(self.window_width/2, self.window_height/2))
                text1_rect = text1.get_rect(center=(self.window_width/2, self.window_height/2 - text_height/4))
                text2_rect = text2.get_rect(center=(self.window_width/2, self.window_height/2 + text_height/4))
                
                # Draw background and text
                self.screen.blit(background, background_rect)
                self.screen.blit(text1, text1_rect)
                self.screen.blit(text2, text2_rect)
                pygame.display.flip()

            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    game = MazeGame()
    game.run() 