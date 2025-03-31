import pygame
import random
import noise
import numpy as np
from typing import List, Tuple, Set

# Constants
CELL_SIZE = 30
MAZE_WIDTH = 25
MAZE_HEIGHT = 25
WINDOW_WIDTH = MAZE_WIDTH * CELL_SIZE
WINDOW_HEIGHT = MAZE_HEIGHT * CELL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game settings
PLAYER_SPEED = 200  # pixels per second

class MazeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Maze Game")
        self.clock = pygame.time.Clock()
        self.maze = self.generate_maze()
        self.player_pos = [1, 1]  # Starting position
        self.exit_pos = [MAZE_HEIGHT - 2, MAZE_WIDTH - 1]  # Exit position
        self.game_over = False
        
        # Movement tracking
        self.pressed_keys: Set[int] = set()
        self.last_update_time = 0

    def generate_maze(self) -> List[List[int]]:
        # Initialize maze with Perlin noise
        maze = [[0 for _ in range(MAZE_WIDTH)] for _ in range(MAZE_HEIGHT)]
        
        # Generate Perlin noise
        scale = 10.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        base = 0
        
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                # Generate noise value between 0 and 1
                noise_val = noise.pnoise2(x/scale, y/scale, octaves=octaves, 
                                        persistence=persistence, lacunarity=lacunarity, 
                                        base=base)
                # Convert to binary (wall or path)
                maze[y][x] = 1 if noise_val > 0 else 0

        # Ensure borders
        for y in range(MAZE_HEIGHT):
            maze[y][0] = 1  # Left border
            maze[y][MAZE_WIDTH-1] = 1  # Right border
        for x in range(MAZE_WIDTH):
            maze[0][x] = 1  # Top border
            maze[MAZE_HEIGHT-1][x] = 1  # Bottom border

        # Create exit
        maze[MAZE_HEIGHT-2][MAZE_WIDTH-1] = 0

        # Ensure starting position is clear
        maze[1][1] = 0

        # Make maze solvable using a simple path-finding algorithm
        self.make_solvable(maze)
        
        return maze

    def make_solvable(self, maze: List[List[int]]):
        # Create a path from start to exit
        current = [1, 1]
        while current[0] < MAZE_HEIGHT - 2 or current[1] < MAZE_WIDTH - 1:
            if current[0] < MAZE_HEIGHT - 2:
                maze[current[0] + 1][current[1]] = 0
                current[0] += 1
            elif current[1] < MAZE_WIDTH - 1:
                maze[current[0]][current[1] + 1] = 0
                current[1] += 1

    def draw(self):
        self.screen.fill(WHITE)
        
        # Draw maze
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, BLACK,
                                   (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Draw player
        pygame.draw.rect(self.screen, RED,
                        (self.player_pos[1] * CELL_SIZE, self.player_pos[0] * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE))

        # Draw exit
        pygame.draw.rect(self.screen, GREEN,
                        (self.exit_pos[1] * CELL_SIZE, self.exit_pos[0] * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE))

        pygame.display.flip()

    def move_player(self, dx: int, dy: int):
        new_x = self.player_pos[1] + dx
        new_y = self.player_pos[0] + dy
        
        if (0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and 
            self.maze[new_y][new_x] == 0):
            self.player_pos[1] = new_x
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
            self.move_player(int(dx), int(dy))

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
                elif event.type == pygame.KEYDOWN and not self.game_over:
                    self.pressed_keys.add(event.key)
                    self.handle_movement()
                elif event.type == pygame.KEYUP:
                    self.pressed_keys.discard(event.key)

            self.draw()
            
            if self.game_over:
                font = pygame.font.Font(None, 74)
                text = font.render('You Win!', True, BLUE)
                text_rect = text.get_rect(center=(WINDOW_WIDTH/2, WINDOW_HEIGHT/2))
                self.screen.blit(text, text_rect)
                pygame.display.flip()

            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    game = MazeGame()
    game.run() 