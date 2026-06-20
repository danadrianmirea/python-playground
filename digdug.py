"""
Dig Dug - Classic 1982 arcade game implemented in Pygame.

Dig tunnels underground, defeat enemies by inflating them until they pop,
or drop rocks on them! Navigate through dirt and avoid enemies.

Controls:
- Arrow keys: Move Dig Dug
- Space / Z: Pump (inflate enemies)
- R: Restart after game over
- P: Pause
"""

import pygame
import random
import math
import sys

# Initialize pygame
pygame.init()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 448
SCREEN_HEIGHT = 576
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
YELLOW = (255, 255, 50)
CYAN = (50, 255, 255)
MAGENTA = (255, 50, 255)
ORANGE = (255, 165, 0)
PURPLE = (180, 50, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (40, 40, 40)
BROWN = (139, 69, 19)
TAN = (210, 180, 140)
DIRT_COLOR = (120, 80, 40)
DIRT_DARK = (90, 60, 30)
TUNNEL_COLOR = (30, 20, 10)
ROCK_COLOR = (140, 140, 150)

# Grid/tile system
TILE_SIZE = 16
GRID_COLS = SCREEN_WIDTH // TILE_SIZE  # 28
GRID_ROWS = (SCREEN_HEIGHT - 48) // TILE_SIZE  # 33 (leave 48px for HUD)
PLAY_OFFSET_Y = 48  # HUD area at top

# Player
PLAYER_SPEED = 2
PLAYER_SIZE = 14

# Pump
PUMP_RANGE = 32
PUMP_COOLDOWN = 15  # frames
PUMP_GROWTH_RATE = 0.3
MAX_PUMP_SIZE = 5.0  # enemy pops at this size

# Enemies
ENEMY_SPEED = 1.0
ENEMY_SPEED_FAST = 1.5
ENEMY_SPAWN_DELAY = 180  # frames between spawns
MAX_ENEMIES = 4

# Rocks
ROCK_FALL_SPEED = 3

# Scoring
SCORE_POOKA = 1000
SCORE_FYGAR = 2000
SCORE_ROCK_POOKA = 2000
SCORE_ROCK_FYGAR = 4000
SCORE_VEGETABLE = 500

# Fonts
font_small = pygame.font.Font(None, 20)
font_medium = pygame.font.Font(None, 32)
font_large = pygame.font.Font(None, 48)
font_huge = pygame.font.Font(None, 64)

# Game states
MENU = 0
PLAYING = 1
GAME_OVER = 2
STAGE_CLEAR = 3

# Direction constants
DIR_NONE = -1
DIR_UP = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_RIGHT = 3

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def grid_to_pixel(gx, gy):
    """Convert grid coordinates to pixel coordinates (center of tile)."""
    return gx * TILE_SIZE + TILE_SIZE // 2, gy * TILE_SIZE + TILE_SIZE // 2 + PLAY_OFFSET_Y

def pixel_to_grid(px, py):
    """Convert pixel coordinates to grid coordinates."""
    return px // TILE_SIZE, (py - PLAY_OFFSET_Y) // TILE_SIZE

def tile_rect(gx, gy):
    """Get the rect for a grid tile."""
    px, py = gx * TILE_SIZE, gy * TILE_SIZE + PLAY_OFFSET_Y
    return pygame.Rect(px, py, TILE_SIZE, TILE_SIZE)


class Particle:
    """Simple particle for explosions."""
    def __init__(self, x, y, color, speed, angle):
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.life = random.randint(15, 30)
        self.max_life = self.life
        self.size = random.randint(2, 4)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.05
        self.life -= 1
        return self.life > 0

    def draw(self, screen):
        alpha = self.life / self.max_life
        size = max(1, int(self.size * alpha))
        color = tuple(int(c * alpha) for c in self.color)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), size)


class DigDug:
    """The player character."""
    def __init__(self):
        self.size = PLAYER_SIZE
        # Start at center top of play area
        self.gx = GRID_COLS // 2
        self.gy = 0
        self.px, self.py = grid_to_pixel(self.gx, self.gy)
        self.speed = PLAYER_SPEED
        self.direction = DIR_NONE
        self.facing = DIR_DOWN
        self.moving = False
        self.alive = True
        self.pump_cooldown = 0
        self.pump_anim_timer = 0
        self.pump_anim_frame = 0

    def update(self, keys, grid):
        if not self.alive:
            return

        # Movement
        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:
            dx, dy = -1, 0
            self.facing = DIR_LEFT
        elif keys[pygame.K_RIGHT]:
            dx, dy = 1, 0
            self.facing = DIR_RIGHT
        elif keys[pygame.K_UP]:
            dx, dy = 0, -1
            self.facing = DIR_UP
        elif keys[pygame.K_DOWN]:
            dx, dy = 0, 1
            self.facing = DIR_DOWN

        self.moving = (dx != 0 or dy != 0)

        if self.moving:
            new_gx = self.gx + dx
            new_gy = self.gy + dy

            # Check bounds
            if 0 <= new_gx < GRID_COLS and 0 <= new_gy < GRID_ROWS:
                # Check if tile is diggable (dirt) or already tunnel
                if grid[new_gy][new_gx] == 1 or grid[new_gy][new_gx] == 0:
                    # Move to new tile
                    self.gx = new_gx
                    self.gy = new_gy
                    self.px, self.py = grid_to_pixel(self.gx, self.gy)
                    self.direction = dx + dy * 2  # encode direction

                    # Dig the tile (turn dirt into tunnel)
                    if grid[self.gy][self.gx] == 1:
                        grid[self.gy][self.gx] = 0

        # Pump cooldown
        if self.pump_cooldown > 0:
            self.pump_cooldown -= 1

        # Pump animation
        if self.pump_anim_timer > 0:
            self.pump_anim_timer -= 1
            if self.pump_anim_timer == 0:
                self.pump_anim_frame = 0

    def pump(self):
        """Try to pump. Returns True if pump was initiated."""
        if self.pump_cooldown == 0 and self.alive:
            self.pump_cooldown = PUMP_COOLDOWN
            self.pump_anim_timer = 10
            self.pump_anim_frame = 1
            return True
        return False

    def get_pump_pos(self):
        """Get the position where the pump nozzle extends to."""
        cx = self.px
        cy = self.py
        if self.facing == DIR_UP:
            return cx, cy - self.size - 8
        elif self.facing == DIR_DOWN:
            return cx, cy + self.size + 8
        elif self.facing == DIR_LEFT:
            return cx - self.size - 8, cy
        elif self.facing == DIR_RIGHT:
            return cx + self.size + 8, cy
        return cx, cy

    def draw(self, screen):
        if not self.alive:
            return

        cx, cy = self.px, self.py

        # Body (Dig Dug is a small character with a helmet)
        # Body
        body_color = (255, 200, 0)  # Yellow suit
        pygame.draw.circle(screen, body_color, (cx, cy), self.size)

        # Helmet
        helmet_color = (0, 100, 255)  # Blue helmet
        pygame.draw.arc(screen, helmet_color,
                        (cx - self.size, cy - self.size - 2, self.size * 2, self.size * 2),
                        math.pi, 2 * math.pi, 3)

        # Face
        pygame.draw.circle(screen, WHITE, (cx - 3, cy - 2), 3)
        pygame.draw.circle(screen, WHITE, (cx + 3, cy - 2), 3)
        pygame.draw.circle(screen, BLACK, (cx - 3, cy - 2), 1)
        pygame.draw.circle(screen, BLACK, (cx + 3, cy - 2), 1)

        # Pump nozzle (extends in facing direction)
        if self.pump_anim_frame > 0:
            px, py = self.get_pump_pos()
            # Draw pump line
            pygame.draw.line(screen, GRAY, (cx, cy), (px, py), 3)
            # Draw nozzle tip
            pygame.draw.circle(screen, WHITE, (px, py), 4)
            pygame.draw.circle(screen, GRAY, (px, py), 4, 1)

    def get_rect(self):
        return pygame.Rect(self.px - self.size, self.py - self.size,
                           self.size * 2, self.size * 2)

    def reset(self):
        self.gx = GRID_COLS // 2
        self.gy = 0
        self.px, self.py = grid_to_pixel(self.gx, self.gy)
        self.alive = True
        self.pump_cooldown = 0
        self.pump_anim_frame = 0
        self.direction = DIR_NONE
        self.facing = DIR_DOWN


class Enemy:
    """Base enemy class for Pooka and Fygar."""
    def __init__(self, gx, gy, etype):
        self.gx = gx
        self.gy = gy
        self.px, self.py = grid_to_pixel(gx, gy)
        self.type = etype  # 0 = Pooka, 1 = Fygar
        self.size = 12
        self.speed = ENEMY_SPEED
        self.alive = True
        self.direction = random.choice([DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT])
        self.move_timer = 0
        self.move_delay = 20
        self.anim_frame = 0
        self.anim_timer = 0

        # Pump state
        self.pump_size = 0.0  # 0 = normal, >0 = being inflated
        self.pump_flash = 0
        self.popping = False
        self.pop_timer = 0

        # Fygar fire breath
        self.fire_breathing = False
        self.fire_timer = 0
        self.fire_cooldown = 0

        # Ghost mode (Pooka can turn ghost to pass through dirt)
        self.ghost_mode = False
        self.ghost_timer = 0

        # Set type-specific properties
        if self.type == 0:  # Pooka
            self.color = RED
            self.secondary = (200, 0, 0)
            self.score_value = SCORE_POOKA
        else:  # Fygar
            self.color = GREEN
            self.secondary = (0, 150, 0)
            self.score_value = SCORE_FYGAR

    def update(self, grid, player_gx, player_gy):
        if not self.alive:
            if self.popping:
                self.pop_timer -= 1
                return self.pop_timer > 0
            return False

        # Animation
        self.anim_timer += 1
        if self.anim_timer > 10:
            self.anim_timer = 0
            self.anim_frame = (self.anim_frame + 1) % 2

        # Pump flash
        if self.pump_flash > 0:
            self.pump_flash -= 1

        # Ghost mode timer (Pooka)
        if self.ghost_mode:
            self.ghost_timer -= 1
            if self.ghost_timer <= 0:
                self.ghost_mode = False

        # Fire breath cooldown (Fygar)
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

        # Fire breath timer
        if self.fire_breathing:
            self.fire_timer -= 1
            if self.fire_timer <= 0:
                self.fire_breathing = False

        # Movement
        self.move_timer += 1
        if self.move_timer >= self.move_delay:
            self.move_timer = 0

            # Try to move toward player
            if random.random() < 0.7:
                # Move toward player
                dx = player_gx - self.gx
                dy = player_gy - self.gy
                if abs(dx) > abs(dy):
                    self.direction = DIR_RIGHT if dx > 0 else DIR_LEFT
                else:
                    self.direction = DIR_DOWN if dy > 0 else DIR_UP

            # Try to move in current direction
            new_gx, new_gy = self.gx, self.gy
            if self.direction == DIR_UP:
                new_gy -= 1
            elif self.direction == DIR_DOWN:
                new_gy += 1
            elif self.direction == DIR_LEFT:
                new_gx -= 1
            elif self.direction == DIR_RIGHT:
                new_gx += 1

            # Check if move is valid
            can_move = False
            if 0 <= new_gx < GRID_COLS and 0 <= new_gy < GRID_ROWS:
                tile = grid[new_gy][new_gx]
                if tile == 0:  # Tunnel
                    can_move = True
                elif tile == 1:  # Dirt
                    if self.ghost_mode:
                        can_move = True
                    elif self.type == 1 and random.random() < 0.3:  # Fygar can eat through dirt
                        grid[new_gy][new_gx] = 0
                        can_move = True

            if can_move:
                self.gx = new_gx
                self.gy = new_gy
                self.px, self.py = grid_to_pixel(self.gx, self.gy)
            else:
                # Pick a new random direction
                self.direction = random.choice([DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT])

            # Pooka ghost mode chance
            if self.type == 0 and random.random() < 0.05:
                self.ghost_mode = True
                self.ghost_timer = 60

            # Fygar fire breath chance
            if self.type == 1 and not self.fire_breathing and self.fire_cooldown <= 0:
                if random.random() < 0.1:
                    self.fire_breathing = True
                    self.fire_timer = 20
                    self.fire_cooldown = 120

        return True

    def start_pump(self):
        """Start inflating this enemy."""
        if self.pump_size == 0:
            self.pump_size = 0.5
            self.pump_flash = 10

    def continue_pump(self):
        """Continue inflating. Returns True if popped."""
        if self.pump_size > 0:
            self.pump_size += PUMP_GROWTH_RATE
            self.pump_flash = 5
            if self.pump_size >= MAX_PUMP_SIZE:
                self.popping = True
                self.pop_timer = 30
                self.alive = False
                return True
        return False

    def draw(self, screen):
        if not self.alive:
            if self.popping:
                # Draw pop animation
                pop_size = int(16 + (30 - self.pop_timer) * 0.5)
                alpha = int((self.pop_timer / 30) * 255)
                color = (255, min(255, alpha), min(255, alpha))
                pygame.draw.circle(screen, color, (int(self.px), int(self.py)), pop_size)
                pygame.draw.circle(screen, WHITE, (int(self.px), int(self.py)), pop_size, 2)
            return

        cx, cy = self.px, self.py
        pump_scale = 1.0 + self.pump_size * 0.3

        if self.type == 0:  # Pooka - red ball with goggles
            color = self.color
            if self.ghost_mode:
                color = (color[0], color[1], color[2], 128)
                # Draw semi-transparent
                s = pygame.Surface((int(24 * pump_scale), int(24 * pump_scale)),
                                   pygame.SRCALPHA)
                pygame.draw.circle(s, (*color[:3], 128),
                                   (int(12 * pump_scale), int(12 * pump_scale)),
                                   int(12 * pump_scale))
                screen.blit(s, (cx - int(12 * pump_scale), cy - int(12 * pump_scale)))
            else:
                # Body
                r = int(self.size * pump_scale)
                pygame.draw.circle(screen, color, (cx, cy), r)
                pygame.draw.circle(screen, self.secondary, (cx, cy), r, 2)

                # Goggles
                pygame.draw.ellipse(screen, WHITE,
                                    (cx - 8, cy - 6, 7, 6))
                pygame.draw.ellipse(screen, WHITE,
                                    (cx + 1, cy - 6, 7, 6))
                pygame.draw.circle(screen, BLACK, (cx - 5, cy - 3), 2)
                pygame.draw.circle(screen, BLACK, (cx + 4, cy - 3), 2)

                # Feet
                foot_offset = 2 if self.anim_frame == 0 else -2
                pygame.draw.ellipse(screen, color,
                                    (cx - 7, cy + r - 4 + foot_offset, 6, 4))
                pygame.draw.ellipse(screen, color,
                                    (cx + 1, cy + r - 4 + foot_offset, 6, 4))

        else:  # Fygar - green dragon
            r = int(self.size * pump_scale)
            # Body
            pygame.draw.ellipse(screen, self.color,
                                (cx - r, cy - r + 2, r * 2, r * 2 - 4))
            pygame.draw.ellipse(screen, self.secondary,
                                (cx - r, cy - r + 2, r * 2, r * 2 - 4), 2)

            # Eyes
            pygame.draw.circle(screen, WHITE, (cx - 4, cy - 3), 4)
            pygame.draw.circle(screen, WHITE, (cx + 4, cy - 3), 4)
            pygame.draw.circle(screen, BLACK, (cx - 4, cy - 3), 2)
            pygame.draw.circle(screen, BLACK, (cx + 4, cy - 3), 2)

            # Mouth
            pygame.draw.ellipse(screen, BLACK,
                                (cx - 4, cy + 2, 8, 4))

            # Fire breath
            if self.fire_breathing:
                fire_x = cx + 14
                fire_y = cy
                for i in range(3):
                    fsize = 8 - i * 2
                    fcolor = [(255, 200, 0), (255, 100, 0), (200, 50, 0)][i]
                    pygame.draw.circle(screen, fcolor,
                                       (fire_x + i * 6, fire_y + random.randint(-2, 2)),
                                       fsize)

            # Feet
            foot_offset = 2 if self.anim_frame == 0 else -2
            pygame.draw.ellipse(screen, self.color,
                                (cx - 7, cy + r - 4 + foot_offset, 6, 4))
            pygame.draw.ellipse(screen, self.color,
                                (cx + 1, cy + r - 4 + foot_offset, 6, 4))

        # Pump indicator (show when being inflated)
        if self.pump_size > 0 and not self.popping:
            # Draw inflation lines
            for i in range(int(self.pump_size * 2)):
                angle = random.uniform(0, math.pi * 2)
                dist = r + 2 + i * 2
                lx = cx + math.cos(angle) * dist
                ly = cy + math.sin(angle) * dist
                pygame.draw.line(screen, WHITE, (cx, cy), (lx, ly), 1)

    def get_rect(self):
        r = int(self.size * (1.0 + self.pump_size * 0.3))
        return pygame.Rect(self.px - r, self.py - r, r * 2, r * 2)


class Rock:
    """A falling rock that can crush enemies."""
    def __init__(self, gx, gy):
        self.gx = gx
        self.gy = gy
        self.px, self.py = grid_to_pixel(gx, gy)
        self.size = TILE_SIZE // 2
        self.falling = False
        self.fall_speed = ROCK_FALL_SPEED
        self.active = True
        self.fall_distance = 0
        self.crushed_enemy = False

    def update(self, grid):
        if not self.active:
            return False

        if not self.falling:
            # Check if there's a tunnel below
            if self.gy + 1 < GRID_ROWS:
                if grid[self.gy + 1][self.gx] == 0:
                    self.falling = True
        else:
            # Fall down
            self.py += self.fall_speed
            self.fall_distance += self.fall_speed

            # Check if we've reached the next tile
            if self.fall_distance >= TILE_SIZE:
                self.fall_distance = 0
                self.gy += 1
                self.py = self.gy * TILE_SIZE + TILE_SIZE // 2 + PLAY_OFFSET_Y

                # Check if we hit dirt or bottom
                if self.gy + 1 >= GRID_ROWS or grid[self.gy + 1][self.gx] == 1:
                    self.falling = False
                    self.active = False
                    return True  # Rock landed

        return True

    def draw(self, screen):
        if not self.active:
            return

        cx, cy = self.px, self.py
        s = self.size

        # Rock shape (irregular polygon)
        points = [
            (cx - s, cy - s + 2),
            (cx + s - 2, cy - s),
            (cx + s, cy + s - 3),
            (cx - s + 3, cy + s),
        ]
        pygame.draw.polygon(screen, ROCK_COLOR, points)
        pygame.draw.polygon(screen, (100, 100, 110), points, 2)

        # Highlight
        pygame.draw.line(screen, (180, 180, 190),
                         (cx - s + 2, cy - s + 2),
                         (cx + s - 4, cy - s + 2), 2)

    def get_rect(self):
        return pygame.Rect(self.px - self.size, self.py - self.size,
                           self.size * 2, self.size * 2)


class DigDugGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Dig Dug")
        self.clock = pygame.time.Clock()
        self.running = True

        # Sound
        self.sound_enabled = True
        try:
            pygame.mixer.init()
            self.pump_sound = self.create_sound(300, 0.08)
            self.pop_sound = self.create_sound(800, 0.2)
            self.rock_sound = self.create_sound(100, 0.4)
        except Exception:
            self.sound_enabled = False

        self.reset_game()

    def create_sound(self, freq, duration):
        """Generate a simple sound wave."""
        sample_rate = 22050
        n_samples = int(sample_rate * duration)
        buf = pygame.sndarray.make_sound(
            [[int(127 * math.sin(2 * math.pi * freq * t / sample_rate))
              for _ in range(1)] for t in range(n_samples)]
        )
        return buf

    def generate_grid(self):
        """Generate the dirt/tunnel grid for a stage."""
        grid = [[1 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        # Store pre-computed colors for each dirt tile (no flickering)
        self.dirt_colors = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

        # Carve initial tunnels
        # Top row is always tunnel (entry point)
        for x in range(GRID_COLS):
            grid[0][x] = 0

        # Carve some random tunnels
        num_tunnels = 8 + self.stage * 2
        for _ in range(num_tunnels):
            # Start from a random existing tunnel
            tunnels = [(x, y) for y in range(GRID_ROWS) for x in range(GRID_COLS)
                       if grid[y][x] == 0]
            if not tunnels:
                break
            sx, sy = random.choice(tunnels)

            # Carve in a random direction
            length = random.randint(5, 15)
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            for i in range(length):
                nx, ny = sx + dx * i, sy + dy * i
                if 0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS:
                    grid[ny][nx] = 0

        # Pre-compute dirt colors and texture dots
        self.dirt_dots = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        for gy in range(GRID_ROWS):
            for gx in range(GRID_COLS):
                if grid[gy][gx] == 1:
                    shade = random.randint(-15, 15)
                    r = max(0, min(255, DIRT_COLOR[0] + shade))
                    g = max(0, min(255, DIRT_COLOR[1] + shade))
                    b = max(0, min(255, DIRT_COLOR[2] + shade))
                    self.dirt_colors[gy][gx] = (r, g, b)
                    # Pre-compute texture dots
                    dots = []
                    for _ in range(3):
                        if random.random() < 0.3:
                            dx = gx * TILE_SIZE + random.randint(3, TILE_SIZE - 3)
                            dy = gy * TILE_SIZE + random.randint(3, TILE_SIZE - 3) + PLAY_OFFSET_Y
                            dots.append((dx, dy))
                    self.dirt_dots[gy][gx] = dots

        # Place some rocks on dirt tiles near the top
        self.rocks = []
        for _ in range(3 + self.stage):
            attempts = 0
            while attempts < 50:
                rx = random.randint(1, GRID_COLS - 2)
                ry = random.randint(1, min(10, GRID_ROWS - 2))
                if grid[ry][rx] == 1:
                    rock = Rock(rx, ry)
                    self.rocks.append(rock)
                    break
                attempts += 1

        return grid

    def reset_game(self):
        self.state = MENU
        self.score = 0
        self.high_score = 0
        self.lives = 3
        self.stage = 1
        self.player = DigDug()
        self.grid = self.generate_grid()
        self.enemies = []
        self.rocks = []
        self.particles = []
        self.spawn_timer = 0
        self.stage_clear_timer = 0
        self.game_over_timer = 0
        self.life_lost_timer = 0
        self.respawn_timer = 0
        self.flash_timer = 0
        self.pump_target = None  # Enemy currently being pumped
        self.total_enemies_spawned = 0  # Track total spawned for stage clear check

    def spawn_enemy(self):
        """Spawn a new enemy at a random tunnel entrance."""
        # Find tunnel tiles on the top row
        entrances = [x for x in range(GRID_COLS) if self.grid[0][x] == 0]
        if entrances and len(self.enemies) < MAX_ENEMIES:
            ex = random.choice(entrances)
            ey = 0
            etype = 0 if random.random() < 0.6 else 1  # 60% Pooka, 40% Fygar
            enemy = Enemy(ex, ey, etype)
            self.enemies.append(enemy)
            self.total_enemies_spawned += 1

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_SPACE or event.key == pygame.K_z):
                    if self.state == PLAYING:
                        if self.player.pump():
                            if self.sound_enabled:
                                self.pump_sound.play()
                            # Check if any enemy is in pump range
                            self.check_pump()

                if event.key == pygame.K_r and self.state == GAME_OVER:
                    self.reset_game()
                    self.state = PLAYING

                if event.key == pygame.K_p:
                    if self.state == PLAYING:
                        self.state = MENU
                    elif self.state == MENU:
                        self.state = PLAYING

                if event.key == pygame.K_SPACE and self.state == MENU:
                    self.state = PLAYING

    def check_pump(self):
        """Check if pump hits any enemy."""
        px, py = self.player.get_pump_pos()
        pump_rect = pygame.Rect(px - 8, py - 8, 16, 16)

        for enemy in self.enemies:
            if enemy.alive and not enemy.popping:
                if pump_rect.colliderect(enemy.get_rect()):
                    if enemy.pump_size == 0:
                        enemy.start_pump()
                        self.pump_target = enemy
                    elif enemy.pump_size > 0:
                        if enemy.continue_pump():
                            # Enemy popped!
                            self.score += enemy.score_value
                            if self.sound_enabled:
                                self.pop_sound.play()
                            # Explosion particles
                            for _ in range(20):
                                angle = random.uniform(0, math.pi * 2)
                                speed = random.uniform(1, 5)
                                self.particles.append(Particle(
                                    enemy.px, enemy.py, enemy.color, speed, angle
                                ))
                            self.pump_target = None
                    break

    def update_enemies(self):
        """Update all enemies."""
        for enemy in self.enemies[:]:
            alive = enemy.update(self.grid, self.player.gx, self.player.gy)
            if not alive and enemy.popping and enemy.pop_timer <= 0:
                self.enemies.remove(enemy)

    def check_collisions(self):
        """Check all collisions."""
        player_rect = self.player.get_rect()

        # Enemy vs player
        for enemy in self.enemies:
            if enemy.alive and not enemy.popping:
                if player_rect.colliderect(enemy.get_rect()):
                    self.player_hit()
                    break

        # Rocks vs enemies
        for rock in self.rocks:
            if rock.active and rock.falling:
                rock_rect = rock.get_rect()
                for enemy in self.enemies[:]:
                    if enemy.alive and not enemy.popping:
                        if rock_rect.colliderect(enemy.get_rect()):
                            # Enemy crushed!
                            enemy.alive = False
                            enemy.popping = True
                            enemy.pop_timer = 20
                            score_mult = 2 if enemy.type == 0 else 2
                            self.score += enemy.score_value * 2
                            if self.sound_enabled:
                                self.pop_sound.play()
                            for _ in range(20):
                                angle = random.uniform(0, math.pi * 2)
                                speed = random.uniform(1, 5)
                                self.particles.append(Particle(
                                    enemy.px, enemy.py, enemy.color, speed, angle
                                ))
                            rock.crushed_enemy = True

        # Rocks vs player
        for rock in self.rocks:
            if rock.active and rock.falling:
                if player_rect.colliderect(rock.get_rect()):
                    self.player_hit()
                    break

    def player_hit(self):
        """Handle player getting hit."""
        self.lives -= 1
        self.life_lost_timer = 60

        # Explosion particles
        cx, cy = self.player.px, self.player.py
        for _ in range(20):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(1, 5)
            self.particles.append(Particle(cx, cy, YELLOW, speed, angle))

        if self.lives <= 0:
            self.state = GAME_OVER
            self.game_over_timer = 120
            if self.score > self.high_score:
                self.high_score = self.score
        else:
            self.player.alive = False
            self.respawn_timer = 90

    def check_stage_clear(self):
        """Check if all enemies are defeated."""
        # Only check stage clear if at least some enemies have been spawned
        # (prevents immediate stage clear at start before enemies appear)
        if self.total_enemies_spawned < 3:
            return
        alive_enemies = [e for e in self.enemies if e.alive or e.popping]
        if not alive_enemies and self.state == PLAYING:
            self.state = STAGE_CLEAR
            self.stage_clear_timer = 120

    def update_particles(self):
        """Update all particles."""
        self.particles = [p for p in self.particles if p.update()]

    def update(self):
        """Main update loop."""
        if self.state == PLAYING:
            keys = pygame.key.get_pressed()
            self.player.update(keys, self.grid)

            # Auto-pump with space held
            if (keys[pygame.K_SPACE] or keys[pygame.K_z]):
                if self.player.pump():
                    if self.sound_enabled:
                        self.pump_sound.play()
                    self.check_pump()

            # Spawn enemies
            self.spawn_timer += 1
            if self.spawn_timer >= ENEMY_SPAWN_DELAY:
                self.spawn_timer = 0
                self.spawn_enemy()

            self.update_enemies()
            self.check_collisions()
            self.check_stage_clear()
            self.update_particles()

            # Update rocks
            for rock in self.rocks:
                rock.update(self.grid)

            # Respawn player
            if not self.player.alive and self.lives > 0:
                self.respawn_timer -= 1
                if self.respawn_timer <= 0:
                    self.player.reset()

        elif self.state == STAGE_CLEAR:
            self.stage_clear_timer -= 1
            self.update_particles()
            if self.stage_clear_timer <= 0:
                self.stage += 1
                self.grid = self.generate_grid()
                self.player.reset()
                self.enemies.clear()
                self.spawn_timer = 0
                self.total_enemies_spawned = 0
                self.state = PLAYING

        elif self.state == GAME_OVER:
            self.game_over_timer -= 1
            self.update_particles()

        if self.life_lost_timer > 0:
            self.life_lost_timer -= 1

    def draw_background(self):
        """Draw the dirt/tunnel grid background."""
        self.screen.fill(BLACK)

        # Draw HUD background
        pygame.draw.rect(self.screen, DARK_GRAY, (0, 0, SCREEN_WIDTH, PLAY_OFFSET_Y))
        pygame.draw.line(self.screen, GRAY, (0, PLAY_OFFSET_Y), (SCREEN_WIDTH, PLAY_OFFSET_Y), 2)

        # Draw grid
        for gy in range(GRID_ROWS):
            for gx in range(GRID_COLS):
                tile = self.grid[gy][gx]
                rect = tile_rect(gx, gy)
                if tile == 1:  # Dirt
                    # Use pre-computed color (no random per frame = no flicker)
                    color = self.dirt_colors[gy][gx]
                    if color is None:
                        color = DIRT_COLOR
                    pygame.draw.rect(self.screen, color, rect)
                    # Dirt texture dots (pre-computed positions)
                    if hasattr(self, 'dirt_dots') and self.dirt_dots[gy][gx]:
                        for dx, dy in self.dirt_dots[gy][gx]:
                            pygame.draw.circle(self.screen, DIRT_DARK, (dx, dy), 2)
                elif tile == 0:  # Tunnel
                    pygame.draw.rect(self.screen, TUNNEL_COLOR, rect)
                    # Tunnel border
                    pygame.draw.rect(self.screen, (20, 15, 5), rect, 1)

    def draw_hud(self):
        """Draw the HUD (score, lives, stage)."""
        # Score
        score_text = font_medium.render(f"{self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # High score
        high_text = font_small.render(f"HIGH SCORE: {self.high_score}", True, WHITE)
        self.screen.blit(high_text, (SCREEN_WIDTH // 2 - high_text.get_width() // 2, 10))

        # Stage
        stage_text = font_small.render(f"STAGE {self.stage}", True, WHITE)
        self.screen.blit(stage_text, (SCREEN_WIDTH - 100, 10))

        # Lives
        for i in range(self.lives):
            ship_x = 20 + i * 20
            ship_y = SCREEN_HEIGHT - 20
            # Draw small Dig Dug icon for lives
            pygame.draw.circle(self.screen, (255, 200, 0), (ship_x + 6, ship_y + 6), 6)
            pygame.draw.arc(self.screen, (0, 100, 255),
                            (ship_x, ship_y + 2, 12, 12),
                            math.pi, 2 * math.pi, 2)

    def draw(self):
        """Main draw loop."""
        self.draw_background()

        if self.state == MENU:
            # Title
            title = font_huge.render("DIG DUG", True, YELLOW)
            self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 120))

            subtitle = font_medium.render("~ 1982 NAMCO ~", True, CYAN)
            self.screen.blit(subtitle, (SCREEN_WIDTH // 2 - subtitle.get_width() // 2, 180))

            # Draw enemy examples
            # Pooka
            pygame.draw.circle(self.screen, RED, (SCREEN_WIDTH // 2 - 40, 250), 14)
            pygame.draw.circle(self.screen, (200, 0, 0), (SCREEN_WIDTH // 2 - 40, 250), 14, 2)
            pygame.draw.ellipse(self.screen, WHITE, (SCREEN_WIDTH // 2 - 48, 244, 7, 6))
            pygame.draw.ellipse(self.screen, WHITE, (SCREEN_WIDTH // 2 - 39, 244, 7, 6))
            pooka_label = font_small.render("POOKA", True, RED)
            self.screen.blit(pooka_label, (SCREEN_WIDTH // 2 - 40 - pooka_label.get_width() // 2, 270))

            # Fygar
            pygame.draw.ellipse(self.screen, GREEN, (SCREEN_WIDTH // 2 + 20, 238, 28, 24))
            pygame.draw.ellipse(self.screen, (0, 150, 0), (SCREEN_WIDTH // 2 + 20, 238, 28, 24), 2)
            fygar_label = font_small.render("FYGAR", True, GREEN)
            self.screen.blit(fygar_label, (SCREEN_WIDTH // 2 + 34 - fygar_label.get_width() // 2, 270))

            start = font_medium.render("Press SPACE to Start", True, WHITE)
            self.screen.blit(start, (SCREEN_WIDTH // 2 - start.get_width() // 2, 330))

            controls = [
                "Arrow Keys - Move",
                "Space / Z - Pump (inflate enemies)",
                "P - Pause",
            ]
            for i, text in enumerate(controls):
                ctrl = font_small.render(text, True, GRAY)
                self.screen.blit(ctrl, (SCREEN_WIDTH // 2 - ctrl.get_width() // 2, 380 + i * 22))

        elif self.state == PLAYING or self.state == STAGE_CLEAR:
            # Draw rocks
            for rock in self.rocks:
                rock.draw(self.screen)

            # Draw enemies
            for enemy in self.enemies:
                enemy.draw(self.screen)

            # Draw player
            self.player.draw(self.screen)

            # Draw particles
            for particle in self.particles:
                particle.draw(self.screen)

            self.draw_hud()

            # Life lost popup
            if self.life_lost_timer > 0:
                life_lost_text = font_large.render("LIFE LOST!", True, RED)
                self.screen.blit(life_lost_text,
                                 (SCREEN_WIDTH // 2 - life_lost_text.get_width() // 2,
                                  SCREEN_HEIGHT // 2 - 50))

            if self.state == STAGE_CLEAR:
                clear_text = font_large.render("STAGE CLEAR!", True, GREEN)
                self.screen.blit(clear_text,
                                 (SCREEN_WIDTH // 2 - clear_text.get_width() // 2,
                                  SCREEN_HEIGHT // 2 - 50))

        elif self.state == GAME_OVER:
            # Draw final state
            for rock in self.rocks:
                rock.draw(self.screen)
            for enemy in self.enemies:
                enemy.draw(self.screen)
            self.player.draw(self.screen)
            for particle in self.particles:
                particle.draw(self.screen)
            self.draw_hud()

            # Game over overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(160)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))

            game_over = font_large.render("GAME OVER", True, RED)
            self.screen.blit(game_over,
                             (SCREEN_WIDTH // 2 - game_over.get_width() // 2,
                              SCREEN_HEIGHT // 2 - 80))

            final_score = font_medium.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(final_score,
                             (SCREEN_WIDTH // 2 - final_score.get_width() // 2,
                              SCREEN_HEIGHT // 2 - 20))

            if self.score >= self.high_score and self.score > 0:
                new_high = font_medium.render("NEW HIGH SCORE!", True, YELLOW)
                self.screen.blit(new_high,
                                 (SCREEN_WIDTH // 2 - new_high.get_width() // 2,
                                  SCREEN_HEIGHT // 2 + 30))

            restart = font_medium.render("Press R to Restart", True, WHITE)
            self.screen.blit(restart,
                             (SCREEN_WIDTH // 2 - restart.get_width() // 2,
                              SCREEN_HEIGHT // 2 + 80))

        pygame.display.flip()

    def run(self):
        """Main game loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = DigDugGame()
    game.run()
