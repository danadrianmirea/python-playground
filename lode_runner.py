"""
Lode Runner - A retro puzzle-action game inspired by the 1983 classic.

Run through levels collecting gold while avoiding guards.
Dig holes to trap guards, then run over them to collect their gold.
Climb ladders and traverse ropes to reach all the gold.

Controls:
  LEFT/RIGHT arrows - Move
  UP arrow - Climb ladder / Jump onto rope
  DOWN arrow - Descend ladder / Drop off rope
  SPACE - Dig hole forward
  D - Dig hole backward
  R - Restart level after game over
  ESC - Quit
"""

import pygame
import random
import sys

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 80, 200)
RED = (220, 50, 50)
GREEN = (50, 200, 50)
YELLOW = (255, 220, 50)
GOLD = (255, 200, 0)
BROWN = (140, 100, 50)
DARK_BROWN = (100, 70, 30)
GRAY = (100, 100, 100)
DARK_GRAY = (60, 60, 60)
DIM_WHITE = (180, 180, 180)
SKIN = (255, 200, 150)
ORANGE = (255, 150, 50)
PURPLE = (180, 50, 200)

# Tile size
TILE_SIZE = 32
GRID_WIDTH = 25  # 800 / 32
GRID_HEIGHT = 18  # 576 / 32 (leaving 24px for HUD)

# Tile types
TILE_EMPTY = 0
TILE_BRICK = 1
TILE_LADDER = 2
TILE_ROPE = 3
TILE_GOLD = 4
TILE_HOLE_LEFT = 5
TILE_HOLE_RIGHT = 6
TILE_HOLE_BOTH = 7

# Player constants
PLAYER_SPEED = 3
DIG_TIMER = 180  # frames hole stays open (3 seconds at 60 FPS)

# Guard constants
GUARD_SPEED = 1.5
GUARD_SPAWN_DELAY = 120  # frames between guard spawns
GUARD_RESPAWN_DELAY = 180  # frames after guard dies before respawning


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.grid_x = x // TILE_SIZE
        self.grid_y = y // TILE_SIZE
        self.vx = 0
        self.vy = 0
        self.on_ladder = False
        self.on_rope = False
        self.facing_right = True
        self.alive = True
        self.anim_frame = 0
        self.anim_timer = 0
        self.digging = False
        self.dig_direction = 0  # -1 left, 1 right
        self.dig_timer = 0

    def update(self, keys, level, dig_forward=False, dig_backward=False):
        if not self.alive:
            return

        self.vx = 0
        self.vy = 0

        # Check what tile we're on
        cx = int(self.x // TILE_SIZE)
        cy = int(self.y // TILE_SIZE)
        # Also check feet position
        feet_y = int((self.y + TILE_SIZE - 2) // TILE_SIZE)
        feet_x = int((self.x + TILE_SIZE // 2) // TILE_SIZE)

        # Check if on ladder
        self.on_ladder = False
        if 0 <= cy < GRID_HEIGHT and 0 <= cx < GRID_WIDTH:
            if level.grid[cy][cx] == TILE_LADDER:
                self.on_ladder = True
        if 0 <= feet_y < GRID_HEIGHT and 0 <= feet_x < GRID_WIDTH:
            if level.grid[feet_y][feet_x] == TILE_LADDER:
                self.on_ladder = True

        # Check if on rope
        self.on_rope = False
        if 0 <= cy < GRID_HEIGHT and 0 <= cx < GRID_WIDTH:
            if level.grid[cy][cx] == TILE_ROPE:
                self.on_rope = True

        # Horizontal movement
        if keys[pygame.K_LEFT]:
            self.vx = -PLAYER_SPEED
            self.facing_right = False
        elif keys[pygame.K_RIGHT]:
            self.vx = PLAYER_SPEED
            self.facing_right = True

        # Vertical movement
        if self.on_ladder:
            if keys[pygame.K_UP]:
                self.vy = -PLAYER_SPEED
            elif keys[pygame.K_DOWN]:
                self.vy = PLAYER_SPEED

        # Rope movement
        if self.on_rope:
            if keys[pygame.K_UP]:
                self.vy = -PLAYER_SPEED
            elif keys[pygame.K_DOWN]:
                self.vy = PLAYER_SPEED

        # Digging (event-based, no auto-repeat)
        self.digging = False
        if dig_forward:
            self.digging = True
            self.dig_direction = 1 if self.facing_right else -1
            self.dig_timer = DIG_TIMER
            level.dig_hole(self, self.dig_direction)
        if dig_backward:
            self.digging = True
            self.dig_direction = -1 if self.facing_right else 1
            self.dig_timer = DIG_TIMER
            level.dig_hole(self, self.dig_direction)

        # Apply movement
        new_x = self.x + self.vx
        new_y = self.y + self.vy

        # Collision detection
        if not self.on_ladder and not self.on_rope:
            # Gravity
            if not self.is_on_ground(level):
                new_y += 3

        # Check wall collisions
        if not self.check_collision(new_x, self.y, level):
            self.x = new_x
        if not self.check_collision(self.x, new_y, level):
            self.y = new_y

        # Clamp to grid
        self.x = max(0, min(WINDOW_WIDTH - TILE_SIZE, self.x))
        self.y = max(0, min(WINDOW_HEIGHT - TILE_SIZE - 24, self.y))

        # Update grid position
        self.grid_x = int((self.x + TILE_SIZE // 2) // TILE_SIZE)
        self.grid_y = int((self.y + TILE_SIZE // 2) // TILE_SIZE)

        # Animation
        if abs(self.vx) > 0 or abs(self.vy) > 0:
            self.anim_timer += 1
            if self.anim_timer > 8:
                self.anim_timer = 0
                self.anim_frame = (self.anim_frame + 1) % 4

        # Collect gold
        self.collect_gold(level)

    def is_on_ground(self, level):
        feet_y = int((self.y + TILE_SIZE) // TILE_SIZE)
        feet_x = int((self.x + TILE_SIZE // 2) // TILE_SIZE)

        if feet_y >= GRID_HEIGHT:
            return True

        if 0 <= feet_x < GRID_WIDTH and 0 <= feet_y < GRID_HEIGHT:
            tile = level.grid[feet_y][feet_x]
            if tile in [TILE_BRICK, TILE_LADDER, TILE_ROPE]:
                return True
            # A hole that has been filled (entity fell in) is walkable
            if tile in [TILE_HOLE_LEFT, TILE_HOLE_RIGHT, TILE_HOLE_BOTH]:
                if level.is_hole_filled(feet_x, feet_y):
                    return True
        return False

    def check_collision(self, x, y, level):
        # Check four corners of player bounding box
        corners = [
            (int((x + 2) // TILE_SIZE), int((y + 2) // TILE_SIZE)),
            (int((x + TILE_SIZE - 2) // TILE_SIZE), int((y + 2) // TILE_SIZE)),
            (int((x + 2) // TILE_SIZE), int((y + TILE_SIZE - 2) // TILE_SIZE)),
            (int((x + TILE_SIZE - 2) // TILE_SIZE), int((y + TILE_SIZE - 2) // TILE_SIZE)),
        ]

        for cx, cy in corners:
            if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT:
                tile = level.grid[cy][cx]
                if tile == TILE_BRICK:
                    return True
                if tile in [TILE_HOLE_LEFT, TILE_HOLE_RIGHT, TILE_HOLE_BOTH]:
                    # Can walk over holes
                    pass
        return False

    def collect_gold(self, level):
        cx = int((self.x + TILE_SIZE // 2) // TILE_SIZE)
        cy = int((self.y + TILE_SIZE // 2) // TILE_SIZE)

        for gy in range(max(0, cy - 1), min(GRID_HEIGHT, cy + 2)):
            for gx in range(max(0, cx - 1), min(GRID_WIDTH, cx + 2)):
                if level.grid[gy][gx] == TILE_GOLD:
                    # Check distance
                    gx_center = gx * TILE_SIZE + TILE_SIZE // 2
                    gy_center = gy * TILE_SIZE + TILE_SIZE // 2
                    px = self.x + TILE_SIZE // 2
                    py = self.y + TILE_SIZE // 2
                    dist = ((gx_center - px) ** 2 + (gy_center - py) ** 2) ** 0.5
                    if dist < TILE_SIZE:
                        level.grid[gy][gx] = TILE_EMPTY
                        level.gold_collected += 1
                        level.score += 10
                        return True
        return False

    def draw(self, screen):
        if not self.alive:
            return

        # Draw player as a small character
        px = int(self.x)
        py = int(self.y)

        # Body
        body_color = BLUE
        pygame.draw.rect(screen, body_color, (px + 6, py + 10, 20, 14))

        # Head
        pygame.draw.circle(screen, SKIN, (px + 16, py + 6), 6)

        # Hat
        pygame.draw.rect(screen, RED, (px + 8, py, 16, 4))
        pygame.draw.rect(screen, RED, (px + 10, py - 2, 12, 4))

        # Legs
        leg_offset = 2 if self.anim_frame % 2 == 0 else -2
        pygame.draw.rect(screen, DARK_BROWN, (px + 8, py + 24, 6, 6))
        pygame.draw.rect(screen, DARK_BROWN, (px + 18, py + 24, 6, 6))

        # Direction indicator (gun/arm)
        if self.facing_right:
            pygame.draw.line(screen, SKIN, (px + 26, py + 14), (px + 30, py + 10), 3)
        else:
            pygame.draw.line(screen, SKIN, (px + 6, py + 14), (px + 2, py + 10), 3)

        # If on ladder, draw climbing pose
        if self.on_ladder:
            pygame.draw.line(screen, SKIN, (px + 8, py + 12), (px + 4, py + 18), 2)
            pygame.draw.line(screen, SKIN, (px + 24, py + 12), (px + 28, py + 18), 2)


class Guard:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = GUARD_SPEED
        self.vy = 0
        self.facing_right = True
        self.alive = True
        self.falling = False
        self.fall_timer = 0
        self.anim_frame = 0
        self.anim_timer = 0
        self.direction_change_timer = 0

    def update(self, level, player):
        if not self.alive:
            return

        # Simple AI: move horizontally, change direction at walls
        new_x = self.x + self.vx

        # Check wall ahead
        if self.vx > 0:
            check_x = int((new_x + TILE_SIZE) // TILE_SIZE)
        else:
            check_x = int((new_x - 1) // TILE_SIZE)
        check_y = int((self.y + TILE_SIZE // 2) // TILE_SIZE)

        wall_ahead = False
        if 0 <= check_x < GRID_WIDTH and 0 <= check_y < GRID_HEIGHT:
            if level.grid[check_y][check_x] == TILE_BRICK:
                wall_ahead = True

        # Check if edge ahead (no ground)
        if self.vx > 0:
            edge_check_x = int((new_x + TILE_SIZE) // TILE_SIZE)
        else:
            edge_check_x = int(new_x // TILE_SIZE)
        edge_check_y = int((self.y + TILE_SIZE + 2) // TILE_SIZE)

        edge_ahead = False
        if 0 <= edge_check_x < GRID_WIDTH and 0 <= edge_check_y < GRID_HEIGHT:
            tile = level.grid[edge_check_y][edge_check_x]
            if tile not in [TILE_BRICK, TILE_LADDER, TILE_ROPE, TILE_HOLE_LEFT, TILE_HOLE_RIGHT, TILE_HOLE_BOTH]:
                edge_ahead = True
            elif tile in [TILE_HOLE_LEFT, TILE_HOLE_RIGHT, TILE_HOLE_BOTH]:
                # A hole that has been filled is solid ground - don't treat as edge
                if level.is_hole_filled(edge_check_x, edge_check_y):
                    edge_ahead = False

        # Change direction if needed
        self.direction_change_timer -= 1
        if wall_ahead or edge_ahead or self.direction_change_timer <= 0:
            self.vx *= -1
            self.facing_right = not self.facing_right
            self.direction_change_timer = random.randint(30, 120)

        # Apply horizontal movement
        self.x += self.vx

        # Gravity
        if not self.is_on_ground(level):
            self.y += 3

        # Clamp
        self.x = max(0, min(WINDOW_WIDTH - TILE_SIZE, self.x))
        self.y = max(0, min(WINDOW_HEIGHT - TILE_SIZE - 24, self.y))

        # Animation
        self.anim_timer += 1
        if self.anim_timer > 10:
            self.anim_timer = 0
            self.anim_frame = (self.anim_frame + 1) % 4

    def is_on_ground(self, level):
        feet_y = int((self.y + TILE_SIZE) // TILE_SIZE)
        feet_x = int((self.x + TILE_SIZE // 2) // TILE_SIZE)

        if feet_y >= GRID_HEIGHT:
            return True

        if 0 <= feet_x < GRID_WIDTH and 0 <= feet_y < GRID_HEIGHT:
            tile = level.grid[feet_y][feet_x]
            if tile in [TILE_BRICK, TILE_LADDER, TILE_ROPE]:
                return True
            # A hole that has been filled (entity fell in) is walkable
            if tile in [TILE_HOLE_LEFT, TILE_HOLE_RIGHT, TILE_HOLE_BOTH]:
                if level.is_hole_filled(feet_x, feet_y):
                    return True
        return False

    def draw(self, screen):
        if not self.alive:
            return

        px = int(self.x)
        py = int(self.y)

        # Body
        body_color = RED
        pygame.draw.rect(screen, body_color, (px + 6, py + 10, 20, 14))

        # Head
        pygame.draw.circle(screen, SKIN, (px + 16, py + 6), 6)

        # Helmet
        pygame.draw.rect(screen, GRAY, (px + 8, py - 1, 16, 5))
        pygame.draw.rect(screen, DARK_GRAY, (px + 10, py - 3, 12, 4))

        # Legs
        leg_offset = 2 if self.anim_frame % 2 == 0 else -2
        pygame.draw.rect(screen, DARK_BROWN, (px + 8, py + 24, 6, 6))
        pygame.draw.rect(screen, DARK_BROWN, (px + 18, py + 24, 6, 6))

        # Eyes (angry)
        eye_x = px + 20 if self.facing_right else px + 10
        pygame.draw.circle(screen, WHITE, (eye_x, py + 5), 2)
        pygame.draw.circle(screen, BLACK, (eye_x, py + 5), 1)

        # If falling in hole, draw differently
        if self.falling:
            pygame.draw.rect(screen, body_color, (px + 4, py + 8, 24, 16))
            # Arms up
            pygame.draw.line(screen, SKIN, (px + 6, py + 10), (px, py + 4), 2)
            pygame.draw.line(screen, SKIN, (px + 26, py + 10), (px + 32, py + 4), 2)


class Level:
    def __init__(self, level_num):
        self.level_num = level_num
        self.grid = [[TILE_EMPTY for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.player_start = (3 * TILE_SIZE, 16 * TILE_SIZE)
        self.guard_spawns = []
        self.gold_collected = 0
        self.total_gold = 0
        self.score = 0
        self.holes = {}  # (x, y) -> timer
        self.generate_level()

    def generate_level(self):
        # Build a classic Lode Runner style level
        # Floor at bottom
        for x in range(GRID_WIDTH):
            self.grid[GRID_HEIGHT - 1][x] = TILE_BRICK

        # Random platforms
        for y in range(GRID_HEIGHT - 4, 0, -3):
            platform_start = random.randint(0, GRID_WIDTH - 8)
            platform_end = platform_start + random.randint(4, 8)
            for x in range(platform_start, min(platform_end, GRID_WIDTH)):
                self.grid[y][x] = TILE_BRICK
            # Add some gaps
            if random.random() < 0.3 and platform_end - platform_start > 4:
                gap_x = random.randint(platform_start + 1, platform_end - 2)
                self.grid[y][gap_x] = TILE_EMPTY

        # Add ladders connecting platforms
        for y in range(GRID_HEIGHT - 2, 1, -1):
            if random.random() < 0.4:
                ladder_x = random.randint(1, GRID_WIDTH - 2)
                # Check if there's a platform above
                has_platform_above = False
                for check_y in range(max(0, y - 3), y):
                    if self.grid[check_y][ladder_x] == TILE_BRICK:
                        has_platform_above = True
                        break
                if has_platform_above or random.random() < 0.3:
                    for ly in range(y, max(0, y - 4), -1):
                        if self.grid[ly][ladder_x] != TILE_BRICK:
                            self.grid[ly][ladder_x] = TILE_LADDER

        # Add ropes
        for y in range(2, GRID_HEIGHT - 2):
            if random.random() < 0.15:
                rope_x = random.randint(1, GRID_WIDTH - 3)
                if self.grid[y][rope_x] == TILE_EMPTY:
                    self.grid[y][rope_x] = TILE_ROPE
                    if rope_x + 1 < GRID_WIDTH and self.grid[y][rope_x + 1] == TILE_EMPTY:
                        self.grid[y][rope_x + 1] = TILE_ROPE

        # Place gold on platforms
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] == TILE_BRICK and random.random() < 0.3:
                    # Place gold above the brick
                    if y > 0 and self.grid[y - 1][x] == TILE_EMPTY:
                        self.grid[y - 1][x] = TILE_GOLD
                        self.total_gold += 1

        # Ensure minimum gold
        if self.total_gold < 5:
            for _ in range(10):
                x = random.randint(1, GRID_WIDTH - 2)
                y = random.randint(1, GRID_HEIGHT - 3)
                if self.grid[y][x] == TILE_EMPTY and self.grid[y + 1][x] == TILE_BRICK:
                    self.grid[y][x] = TILE_GOLD
                    self.total_gold += 1

        # Guard spawn points
        for _ in range(min(3, 1 + self.level_num // 2)):
            x = random.randint(2, GRID_WIDTH - 3)
            y = random.randint(1, GRID_HEIGHT - 3)
            if self.grid[y][x] == TILE_EMPTY and self.grid[y + 1][x] == TILE_BRICK:
                self.guard_spawns.append((x * TILE_SIZE, y * TILE_SIZE))

        if not self.guard_spawns:
            self.guard_spawns.append((5 * TILE_SIZE, 14 * TILE_SIZE))

    def dig_hole(self, player, direction):
        """Dig a hole in the brick below and to the side of the player."""
        px = int((player.x + TILE_SIZE // 2) // TILE_SIZE)
        py = int((player.y + TILE_SIZE) // TILE_SIZE)

        dig_x = px + direction
        dig_y = py

        if 0 <= dig_x < GRID_WIDTH and 0 <= dig_y < GRID_HEIGHT:
            if self.grid[dig_y][dig_x] == TILE_BRICK:
                self.grid[dig_y][dig_x] = TILE_HOLE_BOTH
                if (dig_x, dig_y) not in self.holes:
                    self.holes[(dig_x, dig_y)] = {"timer": DIG_TIMER, "filled": False}

    def fill_hole(self, x, y):
        """Fill a hole back to brick."""
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            if self.grid[y][x] in [TILE_HOLE_LEFT, TILE_HOLE_RIGHT, TILE_HOLE_BOTH]:
                self.grid[y][x] = TILE_BRICK

    def is_hole_filled(self, x, y):
        """Check if a hole at (x, y) has been filled by an entity falling into it."""
        return (x, y) in self.holes and self.holes[(x, y)]["filled"]

    def update_holes(self, player, guards):
        """Update hole timers. When an entity falls into a hole, mark it as filled
        (walkable by other entities). When the timer expires, close the hole
        and kill whatever entity is inside."""
        to_remove = []
        for (hx, hy), info in list(self.holes.items()):
            hole_py = hy * TILE_SIZE
            info["timer"] -= 1

            # Check if any entity fell into this hole (feet at or below the bottom of the hole tile)
            if not info["filled"]:
                entity_in_hole = False
                if player.alive:
                    px = int((player.x + TILE_SIZE // 2) // TILE_SIZE)
                    if px == hx:
                        player_feet = player.y + TILE_SIZE
                        if player_feet >= hole_py + TILE_SIZE:
                            entity_in_hole = True

                if not entity_in_hole:
                    for guard in guards:
                        if guard.alive:
                            gx = int((guard.x + TILE_SIZE // 2) // TILE_SIZE)
                            if gx == hx:
                                guard_feet = guard.y + TILE_SIZE
                                if guard_feet >= hole_py + TILE_SIZE:
                                    entity_in_hole = True
                                    break

                if entity_in_hole:
                    info["filled"] = True

            # When timer expires, close the hole and kill anything inside
            if info["timer"] <= 0:
                # Kill player if inside this hole
                if player.alive:
                    px = int((player.x + TILE_SIZE // 2) // TILE_SIZE)
                    if px == hx:
                        player_feet = player.y + TILE_SIZE
                        if player_feet > hole_py:
                            player.alive = False

                # Kill any guard inside this hole
                for guard in guards:
                    if guard.alive:
                        gx = int((guard.x + TILE_SIZE // 2) // TILE_SIZE)
                        if gx == hx:
                            guard_feet = guard.y + TILE_SIZE
                            if guard_feet > hole_py:
                                guard.alive = False
                                self.score += 25

                self.fill_hole(hx, hy)
                to_remove.append((hx, hy))

        for pos in to_remove:
            if pos in self.holes:
                del self.holes[pos]

    def draw(self, screen):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                tile = self.grid[y][x]
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)

                if tile == TILE_BRICK:
                    # Brick pattern
                    pygame.draw.rect(screen, BROWN, rect)
                    pygame.draw.rect(screen, DARK_BROWN, rect, 1)
                    # Brick lines
                    if (x + y) % 2 == 0:
                        pygame.draw.line(screen, DARK_BROWN,
                                         (x * TILE_SIZE, y * TILE_SIZE + TILE_SIZE // 2),
                                         (x * TILE_SIZE + TILE_SIZE, y * TILE_SIZE + TILE_SIZE // 2), 1)

                elif tile == TILE_LADDER:
                    pygame.draw.rect(screen, DARK_GRAY, rect)
                    # Rungs
                    for r in range(4):
                        ry = y * TILE_SIZE + 4 + r * 8
                        pygame.draw.line(screen, GRAY,
                                         (x * TILE_SIZE + 2, ry),
                                         (x * TILE_SIZE + TILE_SIZE - 2, ry), 2)
                    # Rails
                    pygame.draw.line(screen, GRAY,
                                     (x * TILE_SIZE + 4, y * TILE_SIZE),
                                     (x * TILE_SIZE + 4, y * TILE_SIZE + TILE_SIZE), 2)
                    pygame.draw.line(screen, GRAY,
                                     (x * TILE_SIZE + TILE_SIZE - 4, y * TILE_SIZE),
                                     (x * TILE_SIZE + TILE_SIZE - 4, y * TILE_SIZE + TILE_SIZE), 2)

                elif tile == TILE_ROPE:
                    pygame.draw.rect(screen, BLACK, rect)
                    # Rope line
                    pygame.draw.line(screen, BROWN,
                                     (x * TILE_SIZE, y * TILE_SIZE + TILE_SIZE // 2),
                                     (x * TILE_SIZE + TILE_SIZE, y * TILE_SIZE + TILE_SIZE // 2), 3)
                    # Knots
                    for k in range(3):
                        kx = x * TILE_SIZE + 6 + k * 10
                        pygame.draw.circle(screen, DARK_BROWN, (kx, y * TILE_SIZE + TILE_SIZE // 2), 3)

                elif tile == TILE_GOLD:
                    # Gold nugget
                    pygame.draw.rect(screen, BLACK, rect)
                    pygame.draw.circle(screen, GOLD,
                                       (x * TILE_SIZE + TILE_SIZE // 2,
                                        y * TILE_SIZE + TILE_SIZE // 2),
                                       TILE_SIZE // 3)
                    pygame.draw.circle(screen, YELLOW,
                                       (x * TILE_SIZE + TILE_SIZE // 2 - 2,
                                        y * TILE_SIZE + TILE_SIZE // 2 - 2),
                                       TILE_SIZE // 4)

                elif tile in [TILE_HOLE_LEFT, TILE_HOLE_RIGHT, TILE_HOLE_BOTH]:
                    # Hole (dark pit)
                    pygame.draw.rect(screen, BLACK, rect)
                    # Hole edges
                    if tile in [TILE_HOLE_LEFT, TILE_HOLE_BOTH]:
                        pygame.draw.line(screen, DARK_BROWN,
                                         (x * TILE_SIZE, y * TILE_SIZE),
                                         (x * TILE_SIZE, y * TILE_SIZE + TILE_SIZE), 2)
                    if tile in [TILE_HOLE_RIGHT, TILE_HOLE_BOTH]:
                        pygame.draw.line(screen, DARK_BROWN,
                                         (x * TILE_SIZE + TILE_SIZE, y * TILE_SIZE),
                                         (x * TILE_SIZE + TILE_SIZE, y * TILE_SIZE + TILE_SIZE), 2)

                elif tile == TILE_EMPTY:
                    # Background
                    pygame.draw.rect(screen, BLACK, rect)


class LodeRunnerGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("LODE RUNNER")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.level_num = 1
        self.level = Level(self.level_num)
        self.player = Player(*self.level.player_start)
        self.guards = []
        self.guard_spawn_timer = 0
        self.total_score = 0
        self.state = "playing"  # playing, game_over, level_complete, title
        self.level_complete_timer = 0
        self.game_over_timer = 0

        # Spawn initial guards
        for spawn in self.level.guard_spawns[:2]:
            self.guards.append(Guard(*spawn))

    def handle_events(self):
        self.dig_forward = False
        self.dig_backward = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r and self.state == "game_over":
                    self.reset_game()
                if event.key == pygame.K_SPACE and self.state == "title":
                    self.state = "playing"
                    self.reset_game()
                if self.state == "playing":
                    if event.key == pygame.K_SPACE:
                        self.dig_forward = True
                    if event.key == pygame.K_d:
                        self.dig_backward = True
        return True

    def reset_game(self):
        self.level_num = 1
        self.total_score = 0
        self.load_level()

    def load_level(self):
        self.level = Level(self.level_num)
        self.player = Player(*self.level.player_start)
        self.guards = []
        self.guard_spawn_timer = 0
        self.state = "playing"

        for spawn in self.level.guard_spawns[:2]:
            self.guards.append(Guard(*spawn))

    def update(self):
        if self.state == "title":
            return

        if self.state == "game_over":
            self.game_over_timer -= 1
            return

        if self.state == "level_complete":
            self.level_complete_timer -= 1
            if self.level_complete_timer <= 0:
                self.level_num += 1
                self.load_level()
            return

        keys = pygame.key.get_pressed()
        self.player.update(keys, self.level, self.dig_forward, self.dig_backward)

        # Check player death (touching guard) - skip guards that are inside filled holes
        for guard in self.guards:
            if guard.alive and self.player.alive:
                # Check if guard is inside a filled hole (trapped, can't hurt player)
                guard_in_filled_hole = False
                gx = int((guard.x + TILE_SIZE // 2) // TILE_SIZE)
                gy = int((guard.y + TILE_SIZE) // TILE_SIZE)
                if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
                    tile = self.level.grid[gy][gx]
                    if tile in [TILE_HOLE_LEFT, TILE_HOLE_RIGHT, TILE_HOLE_BOTH]:
                        if self.level.is_hole_filled(gx, gy):
                            guard_in_filled_hole = True

                if not guard_in_filled_hole:
                    dx = (self.player.x + TILE_SIZE // 2) - (guard.x + TILE_SIZE // 2)
                    dy = (self.player.y + TILE_SIZE // 2) - (guard.y + TILE_SIZE // 2)
                    if abs(dx) < TILE_SIZE - 4 and abs(dy) < TILE_SIZE - 4:
                        self.player.alive = False
                        self.state = "game_over"
                        self.game_over_timer = 120
                        return

        # Update guards
        for guard in self.guards:
            guard.update(self.level, self.player)

        # Remove dead guards
        self.guards = [g for g in self.guards if g.alive]

        # Spawn new guards
        self.guard_spawn_timer -= 1
        if self.guard_spawn_timer <= 0 and len(self.guards) < len(self.level.guard_spawns):
            spawn = random.choice(self.level.guard_spawns)
            # Check if spawn point is clear
            clear = True
            for guard in self.guards:
                if abs(guard.x - spawn[0]) < TILE_SIZE and abs(guard.y - spawn[1]) < TILE_SIZE:
                    clear = False
                    break
            if clear:
                self.guards.append(Guard(*spawn))
            self.guard_spawn_timer = GUARD_SPAWN_DELAY

        # Update holes
        self.level.update_holes(self.player, self.guards)

        # Check if player died from a hole closing
        if not self.player.alive and self.state == "playing":
            self.state = "game_over"
            self.game_over_timer = 120
            return

        # Check level complete
        if self.level.gold_collected >= self.level.total_gold:
            self.state = "level_complete"
            self.level_complete_timer = 120
            self.total_score += self.level.score

    def draw_title(self):
        self.screen.fill(BLACK)

        title = self.font_large.render("LODE RUNNER", True, GOLD)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 80))
        self.screen.blit(title, title_rect)

        subtitle = self.font_medium.render("Collect all the gold!", True, WHITE)
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 20))
        self.screen.blit(subtitle, subtitle_rect)

        controls = [
            "ARROW KEYS - Move & Climb",
            "SPACE - Dig forward    D - Dig backward",
            "Trap guards in holes, then grab their gold!",
        ]
        for i, text in enumerate(controls):
            ctrl = self.font_small.render(text, True, DIM_WHITE)
            ctrl_rect = ctrl.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 30 + i * 30))
            self.screen.blit(ctrl, ctrl_rect)

        if pygame.time.get_ticks() // 500 % 2 == 0:
            start = self.font_medium.render("PRESS SPACE TO START", True, WHITE)
            start_rect = start.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 160))
            self.screen.blit(start, start_rect)

        pygame.display.flip()

    def draw(self):
        if self.state == "title":
            self.draw_title()
            return

        self.screen.fill(BLACK)

        # Draw level
        self.level.draw(self.screen)

        # Draw guards
        for guard in self.guards:
            guard.draw(self.screen)

        # Draw player
        self.player.draw(self.screen)

        # Draw HUD
        hud_y = WINDOW_HEIGHT - 24
        pygame.draw.rect(self.screen, DARK_GRAY, (0, hud_y, WINDOW_WIDTH, 24))

        score_text = self.font_small.render(f"SCORE: {self.total_score + self.level.score}", True, WHITE)
        self.screen.blit(score_text, (10, hud_y + 2))

        gold_text = self.font_small.render(f"GOLD: {self.level.gold_collected}/{self.level.total_gold}", True, GOLD)
        self.screen.blit(gold_text, (WINDOW_WIDTH // 2 - 60, hud_y + 2))

        level_text = self.font_small.render(f"LEVEL: {self.level_num}", True, WHITE)
        level_rect = level_text.get_rect(topright=(WINDOW_WIDTH - 10, hud_y + 2))
        self.screen.blit(level_text, level_rect)

        # Game over overlay
        if self.state == "game_over":
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))

            go_text = self.font_large.render("GAME OVER", True, RED)
            go_rect = go_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 40))
            self.screen.blit(go_text, go_rect)

            score_text = self.font_medium.render(f"SCORE: {self.total_score + self.level.score}", True, WHITE)
            score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
            self.screen.blit(score_text, score_rect)

            restart_text = self.font_small.render("PRESS R TO RESTART", True, DIM_WHITE)
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 70))
            self.screen.blit(restart_text, restart_rect)

        # Level complete overlay
        if self.state == "level_complete":
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))

            lc_text = self.font_large.render("LEVEL COMPLETE!", True, GOLD)
            lc_rect = lc_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 20))
            self.screen.blit(lc_text, lc_rect)

            next_text = self.font_small.render("Loading next level...", True, WHITE)
            next_rect = next_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 30))
            self.screen.blit(next_text, next_rect)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = LodeRunnerGame()
    game.run()
