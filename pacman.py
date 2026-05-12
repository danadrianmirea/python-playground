# pacman.py - Classic Pac-Man game

import pygame
import random
import sys
import math

# Initialize pygame
pygame.init()

# Screen dimensions
TILE_SIZE = 30
COLS = 28
ROWS = 31
SCREEN_WIDTH = COLS * TILE_SIZE
SCREEN_HEIGHT = ROWS * TILE_SIZE + 50  # extra space for score
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pac-Man")

# Game constants
FPS = 60
TILE_CENTER = TILE_SIZE // 2

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
PINK = (255, 182, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 182, 85)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 100)
GHOST_HOUSE = (255, 150, 200)
DOT_COLOR = (255, 200, 200)
POWER_PELLET_COLOR = (255, 200, 200)
FRIGHTENED_COLOR = (50, 50, 255)
FRIGHTENED_FLASH = (255, 255, 255)

# Fonts
font = pygame.font.Font(None, 36)
score_font = pygame.font.Font(None, 48)
game_over_font = pygame.font.Font(None, 60)
title_font = pygame.font.Font(None, 28)

# Clock
clock = pygame.time.Clock()

# Maze layout
# 0 = empty, 1 = wall, 2 = dot, 3 = power pellet, 4 = ghost house, 5 = ghost door, 6 = empty (no dot)
MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
    [1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1],
    [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1],
    [1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 5, 5, 1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 4, 4, 4, 4, 4, 4, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 4, 4, 4, 4, 4, 4, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 4, 4, 4, 4, 4, 4, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
    [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
    [1, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 3, 1],
    [1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
    [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Ghost starting positions (in ghost house)
GHOST_START = {
    'blinky': (13, 11),  # outside ghost house
    'pinky': (13, 14),
    'inky': (11, 14),
    'clyde': (15, 14),
}

# Ghost scatter targets (corners)
SCATTER_TARGETS = {
    'blinky': (25, 0),
    'pinky': (2, 0),
    'inky': (27, 30),
    'clyde': (0, 30),
}

# Ghost colors
GHOST_COLORS = {
    'blinky': RED,
    'pinky': PINK,
    'inky': CYAN,
    'clyde': ORANGE,
}


def get_pixel_pos(col, row):
    """Convert grid position to pixel position (center of tile)."""
    return col * TILE_SIZE + TILE_CENTER, row * TILE_SIZE + TILE_CENTER


def get_grid_pos(x, y):
    """Convert pixel position to grid position."""
    return int(x // TILE_SIZE), int(y // TILE_SIZE)


def is_walkable(col, row):
    """Check if a tile is walkable (not a wall)."""
    if col < 0 or col >= COLS or row < 0 or row >= ROWS:
        return False
    return MAZE[row][col] != 1


def get_walkable_neighbors(col, row):
    """Get walkable neighboring positions."""
    neighbors = []
    for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nc, nr = col + dc, row + dr
        if is_walkable(nc, nr):
            neighbors.append((nc, nr))
    return neighbors


class PacMan:
    """The player character."""

    def __init__(self):
        self.col = 14
        self.row = 23
        self.x, self.y = get_pixel_pos(self.col, self.row)
        self.direction = (0, 0)  # (dc, dr)
        self.next_direction = (0, 0)
        self.speed = 2
        self.mouth_angle = 0
        self.mouth_opening = True
        self.anim_timer = 0

    def set_direction(self, direction):
        """Set the next direction to move."""
        self.next_direction = direction

    def update(self):
        """Update Pac-Man position and animation."""
        # Try to apply next direction at the next tile center
        grid_x, grid_y = get_grid_pos(self.x, self.y)
        at_center = (abs(self.x - (grid_x * TILE_SIZE + TILE_CENTER)) < self.speed and
                     abs(self.y - (grid_y * TILE_SIZE + TILE_CENTER)) < self.speed)

        if at_center:
            # Snap to center
            self.x = grid_x * TILE_SIZE + TILE_CENTER
            self.y = grid_y * TILE_SIZE + TILE_CENTER
            self.col, self.row = grid_x, grid_y

            # Try next direction
            if self.next_direction != (0, 0):
                nc, nr = self.col + self.next_direction[0], self.row + self.next_direction[1]
                if is_walkable(nc, nr):
                    self.direction = self.next_direction

            # Move in current direction
            if self.direction != (0, 0):
                nc, nr = self.col + self.direction[0], self.row + self.direction[1]
                if not is_walkable(nc, nr):
                    self.direction = (0, 0)

        # Update position
        self.x += self.direction[0] * self.speed
        self.y += self.direction[1] * self.speed

        # Tunnel wrapping
        if self.x < -TILE_CENTER:
            self.x = SCREEN_WIDTH - TILE_CENTER
        elif self.x > SCREEN_WIDTH - TILE_CENTER:
            self.x = -TILE_CENTER

        # Animation
        self.anim_timer += 1
        if self.anim_timer >= 3:
            self.anim_timer = 0
            if self.mouth_opening:
                self.mouth_angle += 5
                if self.mouth_angle >= 45:
                    self.mouth_opening = False
            else:
                self.mouth_angle -= 5
                if self.mouth_angle <= 0:
                    self.mouth_opening = True

    def draw(self, surface):
        """Draw Pac-Man."""
        # Get angle based on direction
        if self.direction == (1, 0):  # right
            start_angle = math.radians(self.mouth_angle)
            end_angle = math.radians(360 - self.mouth_angle)
        elif self.direction == (-1, 0):  # left
            start_angle = math.radians(180 + self.mouth_angle)
            end_angle = math.radians(180 - self.mouth_angle)
        elif self.direction == (0, -1):  # up
            start_angle = math.radians(90 + self.mouth_angle)
            end_angle = math.radians(90 - self.mouth_angle)
        elif self.direction == (0, 1):  # down
            start_angle = math.radians(270 + self.mouth_angle)
            end_angle = math.radians(270 - self.mouth_angle)
        else:
            start_angle = math.radians(self.mouth_angle)
            end_angle = math.radians(360 - self.mouth_angle)

        pygame.draw.arc(surface, YELLOW,
                        (self.x - TILE_CENTER + 2, self.y - TILE_CENTER + 2,
                         TILE_SIZE - 4, TILE_SIZE - 4),
                        start_angle, end_angle, TILE_SIZE // 2)


class Ghost:
    """A ghost enemy."""

    def __init__(self, name):
        self.name = name
        self.col, self.row = GHOST_START[name]
        self.x, self.y = get_pixel_pos(self.col, self.row)
        self.color = GHOST_COLORS[name]
        self.direction = (0, 0)
        self.speed = 1.5
        self.mode = 'scatter'  # scatter, chase, frightened
        self.in_house = name != 'blinky'
        self.house_timer = 0
        self.frightened_timer = 0
        self.flash_timer = 0
        self.eaten = False
        self.return_col, self.return_row = GHOST_START[name]

    def get_target(self, pacman, blinky_pos=None):
        """Get the target tile based on current mode."""
        if self.eaten:
            return (13, 11)  # return to ghost house

        if self.mode == 'frightened':
            # Random target
            return (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))

        if self.mode == 'scatter':
            return SCATTER_TARGETS[self.name]

        # Chase mode - different behavior per ghost
        if self.name == 'blinky':
            # Directly chases Pac-Man
            return (pacman.col, pacman.row)
        elif self.name == 'pinky':
            # Targets 4 tiles ahead of Pac-Man
            target_col = pacman.col + pacman.direction[0] * 4
            target_row = pacman.row + pacman.direction[1] * 4
            return (target_col, target_row)
        elif self.name == 'inky':
            # Uses Blinky's position to calculate target
            if blinky_pos:
                # Vector from Blinky to 2 tiles ahead of Pac-Man, doubled
                ahead_col = pacman.col + pacman.direction[0] * 2
                ahead_row = pacman.row + pacman.direction[1] * 2
                target_col = ahead_col + (ahead_col - blinky_pos[0])
                target_row = ahead_row + (ahead_row - blinky_pos[1])
                return (target_col, target_row)
            return (pacman.col, pacman.row)
        elif self.name == 'clyde':
            # Chases if far, scatters if close
            dist = math.sqrt((self.col - pacman.col) ** 2 + (self.row - pacman.row) ** 2)
            if dist > 8:
                return (pacman.col, pacman.row)
            else:
                return SCATTER_TARGETS[self.name]

        return (pacman.col, pacman.row)

    def choose_direction(self, pacman, blinky_pos=None):
        """Choose the best direction to move."""
        target = self.get_target(pacman, blinky_pos)
        neighbors = get_walkable_neighbors(self.col, self.row)

        # Can't reverse direction (except when eaten)
        reverse = (-self.direction[0], -self.direction[1])

        best_dist = float('inf')
        best_dir = self.direction

        for nc, nr in neighbors:
            dc, dr = nc - self.col, nr - self.row
            if (dc, dr) == reverse and not self.eaten:
                continue

            if self.mode == 'frightened':
                # Random direction
                return random.choice(neighbors)

            dist = (nc - target[0]) ** 2 + (nr - target[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_dir = (dc, dr)

        return best_dir

    def update(self, pacman, blinky_pos=None):
        """Update ghost position and behavior."""
        # Handle ghost house exit
        if self.in_house:
            self.house_timer += 1
            if self.house_timer > 120:  # 2 seconds
                self.in_house = False
                self.col, self.row = 13, 11  # exit position
                self.x, self.y = get_pixel_pos(self.col, self.row)
            return

        # Handle eaten state
        if self.eaten:
            target = (13, 11)
            if (self.col, self.row) == target:
                self.eaten = False
                self.mode = 'scatter'
                self.speed = 1.5

        # Update frightened timer
        if self.mode == 'frightened':
            self.frightened_timer -= 1
            self.flash_timer += 1
            if self.frightened_timer <= 0:
                self.mode = 'scatter'

        # Move ghost
        grid_x, grid_y = get_grid_pos(self.x, self.y)
        at_center = (abs(self.x - (grid_x * TILE_SIZE + TILE_CENTER)) < self.speed and
                     abs(self.y - (grid_y * TILE_SIZE + TILE_CENTER)) < self.speed)

        if at_center:
            self.x = grid_x * TILE_SIZE + TILE_CENTER
            self.y = grid_y * TILE_SIZE + TILE_CENTER
            self.col, self.row = grid_x, grid_y

            # Choose new direction
            self.direction = self.choose_direction(pacman, blinky_pos)

        # Move
        self.x += self.direction[0] * self.speed
        self.y += self.direction[1] * self.speed

        # Tunnel wrapping
        if self.x < -TILE_CENTER:
            self.x = SCREEN_WIDTH - TILE_CENTER
        elif self.x > SCREEN_WIDTH - TILE_CENTER:
            self.x = -TILE_CENTER

    def draw(self, surface):
        """Draw the ghost."""
        if self.in_house:
            return

        cx, cy = self.x, self.y
        color = self.color

        if self.eaten:
            # Draw just eyes
            eye_color = WHITE
            pygame.draw.circle(surface, eye_color, (cx - 5, cy - 3), 4)
            pygame.draw.circle(surface, eye_color, (cx + 5, cy - 3), 4)
            pygame.draw.circle(surface, BLACK, (cx - 5 + self.direction[0] * 2, cy - 3 + self.direction[1] * 2), 2)
            pygame.draw.circle(surface, BLACK, (cx + 5 + self.direction[0] * 2, cy - 3 + self.direction[1] * 2), 2)
            return

        if self.mode == 'frightened':
            if self.frightened_timer < 120 and self.flash_timer % 10 < 5:
                color = FRIGHTENED_FLASH
            else:
                color = FRIGHTENED_COLOR

        # Ghost body
        body_rect = pygame.Rect(cx - TILE_CENTER + 2, cy - TILE_CENTER + 2,
                                TILE_SIZE - 4, TILE_SIZE - 4)
        pygame.draw.ellipse(surface, color, body_rect)

        # Wavy bottom
        for i in range(3):
            wx = cx - 10 + i * 10
            wy = cy + 10
            pygame.draw.circle(surface, color, (wx, wy), 5)

        # Eyes
        pygame.draw.circle(surface, WHITE, (cx - 5, cy - 3), 4)
        pygame.draw.circle(surface, WHITE, (cx + 5, cy - 3), 4)

        if self.mode == 'frightened':
            # Frightened eyes (small dots)
            pygame.draw.circle(surface, RED, (cx - 5, cy - 3), 2)
            pygame.draw.circle(surface, RED, (cx + 5, cy - 3), 2)
        else:
            # Pupils look in direction of movement
            dx, dy = self.direction
            pygame.draw.circle(surface, BLACK, (cx - 5 + dx * 2, cy - 3 + dy * 2), 2)
            pygame.draw.circle(surface, BLACK, (cx + 5 + dx * 2, cy - 3 + dy * 2), 2)


class Game:
    """Main game class."""

    def __init__(self):
        self.pacman = PacMan()
        self.ghosts = [
            Ghost('blinky'),
            Ghost('pinky'),
            Ghost('inky'),
            Ghost('clyde'),
        ]
        self.score = 0
        self.lives = 3
        self.level = 1
        self.dots_total = 0
        self.dots_eaten = 0
        self.game_over = False
        self.won = False
        self.ghost_combo = 0
        self.power_timer = 0
        self.mode_timer = 0
        self.current_mode = 'scatter'
        self.mode_switch_interval = 420  # 7 seconds

        # Count dots
        for row in MAZE:
            for cell in row:
                if cell == 2 or cell == 3:
                    self.dots_total += 1

    def reset_level(self):
        """Reset positions for a new level or after death."""
        self.pacman = PacMan()
        self.ghosts = [
            Ghost('blinky'),
            Ghost('pinky'),
            Ghost('inky'),
            Ghost('clyde'),
        ]
        self.ghost_combo = 0
        self.power_timer = 0
        self.mode_timer = 0
        self.current_mode = 'scatter'

    def handle_event(self, event):
        """Handle keyboard input."""
        if event.type == pygame.KEYDOWN:
            if self.game_over or self.won:
                if event.key == pygame.K_SPACE:
                    self.__init__()
                return

            if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                self.pacman.set_direction((-1, 0))
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                self.pacman.set_direction((1, 0))
            elif event.key == pygame.K_UP or event.key == pygame.K_w:
                self.pacman.set_direction((0, -1))
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                self.pacman.set_direction((0, 1))

    def eat_dot(self):
        """Check if Pac-Man is on a dot and eat it."""
        col, row = self.pacman.col, self.pacman.row
        if 0 <= col < COLS and 0 <= row < ROWS:
            cell = MAZE[row][col]
            if cell == 2:
                MAZE[row][col] = 0
                self.score += 10
                self.dots_eaten += 1
                return True
            elif cell == 3:
                MAZE[row][col] = 0
                self.score += 50
                self.dots_eaten += 1
                self.activate_power_pellet()
                return True
        return False

    def activate_power_pellet(self):
        """Activate power pellet mode (ghosts become frightened)."""
        self.power_timer = 300  # 5 seconds
        self.ghost_combo = 0
        for ghost in self.ghosts:
            if not ghost.eaten and not ghost.in_house:
                ghost.mode = 'frightened'
                ghost.frightened_timer = 300
                ghost.flash_timer = 0
                # Reverse direction
                ghost.direction = (-ghost.direction[0], -ghost.direction[1])

    def check_ghost_collisions(self):
        """Check if Pac-Man collides with any ghost."""
        for ghost in self.ghosts:
            if ghost.in_house or ghost.eaten:
                continue
            if (self.pacman.col, self.pacman.row) == (ghost.col, ghost.row):
                if ghost.mode == 'frightened':
                    # Eat ghost
                    ghost.eaten = True
                    ghost.speed = 4
                    self.ghost_combo += 1
                    self.score += 200 * (2 ** self.ghost_combo)
                else:
                    # Pac-Man dies
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True
                    else:
                        self.reset_level()
                    return

    def update(self):
        """Update game state."""
        if self.game_over or self.won:
            return

        # Update Pac-Man
        self.pacman.update()

        # Eat dots
        self.eat_dot()

        # Check win condition
        if self.dots_eaten >= self.dots_total:
            self.won = True
            return

        # Update mode timer (scatter/chase cycle)
        self.mode_timer += 1
        if self.mode_timer >= self.mode_switch_interval:
            self.mode_timer = 0
            if self.current_mode == 'scatter':
                self.current_mode = 'chase'
            else:
                self.current_mode = 'scatter'

            # Update ghost modes (only if not frightened)
            for ghost in self.ghosts:
                if ghost.mode != 'frightened' and not ghost.eaten and not ghost.in_house:
                    ghost.mode = self.current_mode
                    ghost.direction = (-ghost.direction[0], -ghost.direction[1])

        # Update power timer
        if self.power_timer > 0:
            self.power_timer -= 1
            if self.power_timer <= 0:
                for ghost in self.ghosts:
                    if ghost.mode == 'frightened':
                        ghost.mode = self.current_mode

        # Update ghosts
        blinky_pos = (self.ghosts[0].col, self.ghosts[0].row)
        for ghost in self.ghosts:
            ghost.update(self.pacman, blinky_pos)

        # Check collisions
        self.check_ghost_collisions()

    def draw(self, surface):
        """Draw the entire game."""
        surface.fill(BLACK)

        # Draw maze
        for r in range(ROWS):
            for c in range(COLS):
                cell = MAZE[r][c]
                x = c * TILE_SIZE
                y = r * TILE_SIZE

                if cell == 1:  # Wall
                    self._draw_wall(surface, c, r)
                elif cell == 2:  # Dot
                    pygame.draw.circle(surface, DOT_COLOR,
                                       (x + TILE_CENTER, y + TILE_CENTER), 3)
                elif cell == 3:  # Power pellet
                    pygame.draw.circle(surface, POWER_PELLET_COLOR,
                                       (x + TILE_CENTER, y + TILE_CENTER), 7)
                elif cell == 4:  # Ghost house
                    pygame.draw.rect(surface, GHOST_HOUSE,
                                     (x + 1, y + 1, TILE_SIZE - 2, TILE_SIZE - 2), 1)
                elif cell == 5:  # Ghost door
                    pygame.draw.rect(surface, PINK,
                                     (x + 1, y + 1, TILE_SIZE - 2, TILE_SIZE - 2))

        # Draw ghosts
        for ghost in self.ghosts:
            ghost.draw(surface)

        # Draw Pac-Man
        self.pacman.draw(surface)

        # Draw score and lives
        score_text = score_font.render(f"Score: {self.score}", True, WHITE)
        surface.blit(score_text, (10, SCREEN_HEIGHT - 45))

        lives_text = font.render(f"Lives: {self.lives}", True, WHITE)
        surface.blit(lives_text, (SCREEN_WIDTH - 120, SCREEN_HEIGHT - 45))

        # Draw level
        level_text = font.render(f"Level: {self.level}", True, WHITE)
        surface.blit(level_text, (SCREEN_WIDTH // 2 - 40, SCREEN_HEIGHT - 45))

        # Draw game over / win
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            surface.blit(overlay, (0, 0))

            text = game_over_font.render("GAME OVER", True, RED)
            surface.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - 60))

            restart_text = font.render("Press SPACE to restart", True, WHITE)
            surface.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 10))

        if self.won:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            surface.blit(overlay, (0, 0))

            text = game_over_font.render("YOU WIN!", True, YELLOW)
            surface.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - 60))

            restart_text = font.render("Press SPACE to play again", True, WHITE)
            surface.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 10))

    def _draw_wall(self, surface, col, row):
        """Draw a wall tile with 3D effect."""
        x = col * TILE_SIZE
        y = row * TILE_SIZE
        rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

        # Wall base
        pygame.draw.rect(surface, DARK_BLUE, rect)

        # Check neighbors for wall connections
        has_top = row > 0 and MAZE[row - 1][col] == 1
        has_bottom = row < ROWS - 1 and MAZE[row + 1][col] == 1
        has_left = col > 0 and MAZE[row][col - 1] == 1
        has_right = col < COLS - 1 and MAZE[row][col + 1] == 1

        # Draw inner fill (lighter blue)
        inner = pygame.Rect(x + 2, y + 2, TILE_SIZE - 4, TILE_SIZE - 4)
        pygame.draw.rect(surface, BLUE, inner, border_radius=4)

        # Draw rounded corners based on connections
        if not has_top:
            pygame.draw.rect(surface, DARK_BLUE, (x, y, TILE_SIZE, 2))
        if not has_bottom:
            pygame.draw.rect(surface, DARK_BLUE, (x, y + TILE_SIZE - 2, TILE_SIZE, 2))
        if not has_left:
            pygame.draw.rect(surface, DARK_BLUE, (x, y, 2, TILE_SIZE))
        if not has_right:
            pygame.draw.rect(surface, DARK_BLUE, (x + TILE_SIZE - 2, y, 2, TILE_SIZE))

        # Highlight
        if has_top:
            pygame.draw.line(surface, (100, 100, 255), (x + 2, y + 2), (x + TILE_SIZE - 3, y + 2), 1)
        if has_left:
            pygame.draw.line(surface, (100, 100, 255), (x + 2, y + 2), (x + 2, y + TILE_SIZE - 3), 1)


def main():
    """Main game loop."""
    game = Game()
    running = True

    while running:
        dt = clock.tick(FPS)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            game.handle_event(event)

        # Update
        game.update()

        # Draw
        game.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
