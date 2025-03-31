import pygame
import random
import noise
import numpy as np
from typing import List, Tuple, Set

# Constants
INITIAL_MAZE_WIDTH = 15
INITIAL_MAZE_HEIGHT = 15
MIN_CELL_SIZE = 20  # Minimum cell size to maintain playability
MAX_CELL_SIZE = 40  # Maximum cell size to prevent too large cells

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game settings
PLAYER_SPEED = 50

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
        
        # Create the game window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Maze Game")
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
        self.player_pos = [1, 1]  # Starting position
        self.exit_pos = [self.maze_height - 2, self.maze_width - 1]  # Exit position
        self.game_over = False
        
        # Movement tracking
        self.pressed_keys: Set[int] = set()
        self.last_update_time = 0

    def generate_maze(self) -> List[List[int]]:
        # Initialize maze with Perlin noise
        maze = [[0 for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        
        # Generate Perlin noise with random base value
        scale = 10.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        base = random.randint(0, 1000)  # Random base value for different mazes each run
        
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                # Generate noise value between 0 and 1
                noise_val = noise.pnoise2(x/scale, y/scale, octaves=octaves, 
                                        persistence=persistence, lacunarity=lacunarity, 
                                        base=base)
                # Convert to binary (wall or path)
                maze[y][x] = 1 if noise_val > 0 else 0

        # Ensure borders
        for y in range(self.maze_height):
            maze[y][0] = 1  # Left border
            maze[y][self.maze_width-1] = 1  # Right border
        for x in range(self.maze_width):
            maze[0][x] = 1  # Top border
            maze[self.maze_height-1][x] = 1  # Bottom border

        # Create exit
        maze[self.maze_height-2][self.maze_width-1] = 0

        # Ensure starting position is clear
        maze[1][1] = 0

        # Make maze solvable using a simple path-finding algorithm
        self.make_solvable(maze)
        
        return maze

    def make_solvable(self, maze: List[List[int]]):
        # Create a path from start to exit
        current = [1, 1]
        while current[0] < self.maze_height - 2 or current[1] < self.maze_width - 1:
            if current[0] < self.maze_height - 2:
                maze[current[0] + 1][current[1]] = 0
                current[0] += 1
            elif current[1] < self.maze_width - 1:
                maze[current[0]][current[1] + 1] = 0
                current[1] += 1

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw maze
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, WHITE,
                                   (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        # Draw player
        pygame.draw.rect(self.screen, RED,
                        (self.player_pos[1] * self.cell_size, self.player_pos[0] * self.cell_size,
                         self.cell_size, self.cell_size))

        # Draw exit
        pygame.draw.rect(self.screen, GREEN,
                        (self.exit_pos[1] * self.cell_size, self.exit_pos[0] * self.cell_size,
                         self.cell_size, self.cell_size))

        pygame.display.flip()

    def move_player(self, dx: float, dy: float):
        # Try to move diagonally first
        new_x = round(self.player_pos[1] + dx)
        new_y = round(self.player_pos[0] + dy)
        
        if (0 <= new_x < self.maze_width and 0 <= new_y < self.maze_height and 
            self.maze[new_y][new_x] == 0):
            self.player_pos[1] = new_x
            self.player_pos[0] = new_y
        else:
            # If diagonal movement fails, try to slide along walls
            # Try horizontal movement first
            new_x = round(self.player_pos[1] + dx)
            if (0 <= new_x < self.maze_width and 
                self.maze[self.player_pos[0]][new_x] == 0):
                self.player_pos[1] = new_x
            
            # Then try vertical movement
            new_y = round(self.player_pos[0] + dy)
            if (0 <= new_y < self.maze_height and 
                self.maze[new_y][self.player_pos[1]] == 0):
                self.player_pos[0] = new_y
            
        # Check if player reached exit
        if self.player_pos == self.exit_pos:
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
            dx = dx / 1.414  # 1/√2 for diagonal movement
            dy = dy / 1.414
            
        if dx != 0 or dy != 0:
            self.move_player(dx, dy)

    def run(self):
        running = True
        while running:
            current_time = pygame.time.get_ticks()
            
            # Handle movement based on time
            if self.pressed_keys and not self.game_over:
                time_delta = (current_time - self.last_update_time) / 1000.0  # Convert to seconds
                if time_delta >= 1.0 / PLAYER_SPEED:  # Check if enough time has passed based on speed
                    self.handle_movement()
                    self.last_update_time = current_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.game_over:
                        # Reset game with larger maze, increasing both dimensions
                        self.maze_width += 1
                        self.maze_height += 1
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