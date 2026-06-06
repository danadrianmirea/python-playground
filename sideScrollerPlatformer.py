# sideScrollerPlatformer.py - Side-scrolling platformer with axe combat

import pygame
import random
import sys
import math

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 500
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Axe Jumper")

# Game constants
GRAVITY = 0.6
JUMP_STRENGTH = -12
MOVE_SPEED = 5
PLAYER_WIDTH = 30
PLAYER_HEIGHT = 40
AXE_RANGE = 50
AXE_SWING_TIME = 15  # frames
TILE_SIZE = 40
FPS = 60
LEVEL_LENGTH = 100  # tiles wide

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (100, 150, 255)
SKY_BLUE = (135, 206, 235)
GREEN = (0, 180, 0)
DARK_GREEN = (0, 120, 0)
BROWN = (139, 69, 19)
DARK_BROWN = (100, 50, 10)
GRAY = (150, 150, 150)
DARK_GRAY = (80, 80, 80)
RED = (220, 50, 50)
YELLOW = (255, 255, 0)
GOLD = (255, 215, 0)
ORANGE = (255, 165, 0)
PURPLE = (150, 50, 200)
PINK = (255, 100, 150)

# Fonts
font_large = pygame.font.SysFont("Arial", 48, bold=True)
font_medium = pygame.font.SysFont("Arial", 28)
font_small = pygame.font.SysFont("Arial", 20)
font_tiny = pygame.font.SysFont("Arial", 12, bold=True)

# Pre-rendered coin dollar sign
coin_dollar = font_tiny.render("$", True, (200, 150, 0))

clock = pygame.time.Clock()

# Camera
camera_x = 0


class Camera:
    def __init__(self):
        self.x = 0

    def apply(self, rect):
        return rect.move(-self.x, 0)

    def update(self, target):
        target_x = target.x - SCREEN_WIDTH // 3
        self.x += (target_x - self.x) * 0.1
        self.x = max(0, min(self.x, LEVEL_LENGTH * TILE_SIZE - SCREEN_WIDTH))


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False
        self.facing_right = True
        self.axe_swinging = False
        self.axe_timer = 0
        self.axe_angle = 0
        self.invincible = 0
        self.alive = True
        self.health = 3  # hits per life
        self.lives = 3
        self.max_health = 3

    def update(self):
        self.vel_y += GRAVITY
        self.x += self.vel_x
        self.y += self.vel_y

        # Don't let player go past left or right edge
        if self.x < 0:
            self.x = 0
            self.vel_x = 0
        if self.x > LEVEL_LENGTH * TILE_SIZE - self.width:
            self.x = LEVEL_LENGTH * TILE_SIZE - self.width
            self.vel_x = 0

        if self.invincible > 0:
            self.invincible -= 1

        # Axe swing animation
        if self.axe_swinging:
            self.axe_timer += 1
            self.axe_angle = -90 + (self.axe_timer / AXE_SWING_TIME) * 180
            if self.axe_timer >= AXE_SWING_TIME:
                self.axe_swinging = False
                self.axe_timer = 0
                self.axe_angle = 0

    def jump(self):
        if self.on_ground:
            self.vel_y = JUMP_STRENGTH
            self.on_ground = False

    def swing_axe(self):
        if not self.axe_swinging:
            self.axe_swinging = True
            self.axe_timer = 0

    def get_axe_hitbox(self):
        """Returns the area where the axe can hit."""
        if not self.axe_swinging:
            return None

        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2

        if self.facing_right:
            hit_x = center_x
        else:
            hit_x = center_x - AXE_RANGE

        return pygame.Rect(hit_x, center_y - 20, AXE_RANGE, 40)

    def draw(self, surface, cam):
        # Flash when invincible
        if self.invincible > 0 and self.invincible % 6 < 3:
            return

        screen_x = self.x - cam.x

        # Body
        body_color = (50, 150, 255) if self.alive else GRAY
        body_rect = pygame.Rect(screen_x, self.y, self.width, self.height)
        pygame.draw.rect(surface, body_color, body_rect)
        pygame.draw.rect(surface, BLACK, body_rect, 2)

        # Head
        head_rect = pygame.Rect(screen_x - 2, self.y - 8, self.width + 4, 16)
        pygame.draw.ellipse(surface, (255, 200, 150), head_rect)
        pygame.draw.ellipse(surface, BLACK, head_rect, 2)

        # Eyes
        eye_offset = 5 if self.facing_right else -5
        pygame.draw.circle(surface, BLACK, (int(screen_x + self.width // 2 + eye_offset), int(self.y - 2)), 3)

        # Legs (always same size to avoid flickering)
        leg_color = (0, 80, 200)
        leg_height = 7
        pygame.draw.rect(surface, leg_color, (screen_x + 4, self.y + self.height - leg_height, 8, leg_height))
        pygame.draw.rect(surface, leg_color, (screen_x + self.width - 12, self.y + self.height - leg_height, 8, leg_height))

        # Axe
        if self.axe_swinging:
            center_x = screen_x + self.width // 2
            center_y = self.y + self.height // 2
            angle_rad = math.radians(self.axe_angle)
            dir = 1 if self.facing_right else -1
            handle_len = 30
            handle_end_x = center_x + dir * handle_len * math.cos(angle_rad)
            handle_end_y = center_y + handle_len * math.sin(angle_rad)

            # Handle
            pygame.draw.line(surface, BROWN, (center_x, center_y), (handle_end_x, handle_end_y), 4)

            # Axe head
            head_x = handle_end_x + dir * 10 * math.cos(angle_rad)
            head_y = handle_end_y + 10 * math.sin(angle_rad)
            axe_points = [
                (head_x, head_y),
                (head_x + dir * 8, head_y - 6),
                (head_x + dir * 8, head_y + 6),
            ]
            pygame.draw.polygon(surface, GRAY, axe_points)
            pygame.draw.polygon(surface, BLACK, axe_points, 2)
        else:
            # Idle axe on back
            ax = screen_x + (self.width if self.facing_right else -8)
            pygame.draw.line(surface, BROWN, (ax, self.y + 5), (ax, self.y + 25), 3)
            pygame.draw.rect(surface, GRAY, (ax - 4, self.y + 5, 8, 6))
            pygame.draw.rect(surface, BLACK, (ax - 4, self.y + 5, 8, 6), 1)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Enemy:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 30
        self.vel_x = -1
        self.vel_y = 0
        self.alive = True
        self.death_timer = 0
        self.type = random.choice(["slime", "bat", "spider"])

    def update(self, platforms):
        if not self.alive:
            self.death_timer += 1
            return

        self.vel_y += GRAVITY
        self.x += self.vel_x
        self.y += self.vel_y

        # Platform collision
        self.on_ground = False
        for plat in platforms:
            if self.get_rect().colliderect(plat.get_rect()):
                # Determine overlap amounts
                overlap_left = (self.x + self.width) - plat.x
                overlap_right = (plat.x + plat.width) - self.x
                overlap_top = (self.y + self.height) - plat.y
                overlap_bottom = (plat.y + plat.height) - self.y

                # Find smallest overlap
                min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

                if min_overlap == overlap_top and self.vel_y >= 0:
                    # Landing on top
                    self.y = plat.y - self.height
                    self.vel_y = 0
                    self.on_ground = True
                elif min_overlap == overlap_bottom and self.vel_y <= 0:
                    # Hitting head
                    self.y = plat.y + plat.height
                    self.vel_y = 0
                elif min_overlap == overlap_left:
                    # Hitting left side - reverse direction
                    self.x = plat.x - self.width
                    self.vel_x *= -1
                elif min_overlap == overlap_right:
                    # Hitting right side - reverse direction
                    self.x = plat.x + plat.width
                    self.vel_x *= -1

        # Reverse direction at edges
        if self.on_ground:
            # Check if there's ground ahead
            ahead_x = self.x + self.vel_x * 20
            on_ground_ahead = False
            for plat in platforms:
                if (ahead_x + self.width > plat.x and ahead_x < plat.x + plat.width
                        and abs(self.y + self.height - plat.y) < 5):
                    on_ground_ahead = True
                    break
            if not on_ground_ahead:
                self.vel_x *= -1

        # Bounce off world boundaries
        if self.x < 0:
            self.x = 0
            self.vel_x *= -1
        elif self.x > LEVEL_LENGTH * TILE_SIZE - self.width:
            self.x = LEVEL_LENGTH * TILE_SIZE - self.width
            self.vel_x *= -1

    def draw(self, surface, cam):
        screen_x = self.x - cam.x

        if not self.alive:
            if self.death_timer < 20:
                # Death animation - simple shrink
                scale = 1.0 - self.death_timer / 20
                w = int(self.width * scale)
                h = int(self.height * scale)
                if w > 0 and h > 0:
                    offset_x = (self.width - w) // 2
                    offset_y = (self.height - h) // 2
                    if self.type == "slime":
                        color = (100, 255, 100)
                    elif self.type == "bat":
                        color = (100, 100, 255)
                    else:
                        color = (255, 100, 100)
                    pygame.draw.ellipse(surface, color, (screen_x + offset_x, self.y + offset_y, w, h))
            return

        if self.type == "slime":
            # Green slime blob
            pygame.draw.ellipse(surface, (50, 200, 50), (screen_x, self.y, self.width, self.height))
            pygame.draw.ellipse(surface, BLACK, (screen_x, self.y, self.width, self.height), 2)
            # Eyes
            pygame.draw.circle(surface, WHITE, (int(screen_x + 8), int(self.y + 10)), 5)
            pygame.draw.circle(surface, WHITE, (int(screen_x + self.width - 8), int(self.y + 10)), 5)
            pygame.draw.circle(surface, BLACK, (int(screen_x + 8), int(self.y + 10)), 2)
            pygame.draw.circle(surface, BLACK, (int(screen_x + self.width - 8), int(self.y + 10)), 2)
        elif self.type == "bat":
            # Purple bat
            body_rect = pygame.Rect(screen_x + 5, self.y + 8, self.width - 10, self.height - 16)
            pygame.draw.ellipse(surface, (120, 50, 180), body_rect)
            # Wings (frame-based animation)
            wing_flap = ((self.x + self.y) % 10) - 5
            pygame.draw.ellipse(surface, (100, 40, 160),
                                (screen_x - 10, self.y + wing_flap, 15, 15))
            pygame.draw.ellipse(surface, (100, 40, 160),
                                (screen_x + self.width - 5, self.y + wing_flap, 15, 15))
            # Eyes
            pygame.draw.circle(surface, RED, (int(screen_x + 8), int(self.y + 10)), 3)
            pygame.draw.circle(surface, RED, (int(screen_x + self.width - 8), int(self.y + 10)), 3)
        else:
            # Red spider
            pygame.draw.ellipse(surface, (200, 50, 50), (screen_x, self.y + 5, self.width, self.height - 5))
            pygame.draw.ellipse(surface, BLACK, (screen_x, self.y + 5, self.width, self.height - 5), 2)
            # Legs
            for side in [-1, 1]:
                for i in range(3):
                    lx = screen_x + (self.width // 2) + side * (5 + i * 5)
                    ly = self.y + 10 + i * 5
                    pygame.draw.line(surface, BLACK, (lx, ly), (lx + side * 10, ly + 10), 2)
            # Eyes
            pygame.draw.circle(surface, YELLOW, (int(screen_x + 8), int(self.y + 10)), 3)
            pygame.draw.circle(surface, YELLOW, (int(screen_x + self.width - 8), int(self.y + 10)), 3)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Coin:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 8
        self.collected = False

    def draw(self, surface, cam):
        if self.collected:
            return
        screen_x = self.x - cam.x
        # Coin
        pygame.draw.circle(surface, GOLD, (int(screen_x), int(self.y)), self.radius)
        pygame.draw.circle(surface, (200, 170, 0), (int(screen_x), int(self.y)), self.radius, 2)
        # Dollar sign (pre-rendered)
        surface.blit(coin_dollar, (screen_x - 4, self.y - 6))

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)


class Platform:
    def __init__(self, x, y, width, height=TILE_SIZE, platform_type="ground"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = platform_type  # "ground", "brick", "stone"

    def draw(self, surface, cam):
        screen_x = self.x - cam.x
        rect = pygame.Rect(screen_x, self.y, self.width, self.height)

        if self.type == "ground":
            # Dirt with grass top
            pygame.draw.rect(surface, BROWN, rect)
            pygame.draw.rect(surface, DARK_BROWN, rect, 2)
            # Grass top
            grass_rect = pygame.Rect(screen_x, self.y, self.width, 6)
            pygame.draw.rect(surface, GREEN, grass_rect)
            # Grass blades (deterministic based on position)
            for gx in range(int(screen_x), int(screen_x + self.width), 8):
                gh = 2 + ((gx * 7 + self.y * 13) % 4)
                pygame.draw.line(surface, DARK_GREEN, (gx, self.y), (gx, self.y - gh), 2)
        elif self.type == "brick":
            pygame.draw.rect(surface, (180, 100, 50), rect)
            pygame.draw.rect(surface, (120, 60, 20), rect, 2)
            # Brick pattern
            for bx in range(int(screen_x), int(screen_x + self.width), TILE_SIZE // 2):
                pygame.draw.line(surface, (120, 60, 20), (bx, self.y), (bx, self.y + self.height), 1)
            for by in range(self.y, self.y + self.height, TILE_SIZE // 2):
                offset = TILE_SIZE // 4 if ((by - self.y) // (TILE_SIZE // 2)) % 2 else 0
                pygame.draw.line(surface, (120, 60, 20),
                                 (screen_x + offset, by), (screen_x + self.width + offset, by), 1)
        elif self.type == "stone":
            pygame.draw.rect(surface, GRAY, rect)
            pygame.draw.rect(surface, DARK_GRAY, rect, 2)
            # Stone texture (deterministic based on position)
            for i in range(3):
                sx = screen_x + 5 + ((self.x * 11 + i * 17) % (self.width - 10))
                sy = self.y + 5 + ((self.y * 7 + i * 13) % (self.height - 10))
                pygame.draw.circle(surface, DARK_GRAY, (int(sx), int(sy)), 3)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Exit:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 60
        self.reached = False
        self.player_near = False

    def draw(self, surface, cam):
        screen_x = self.x - cam.x
        # Door frame
        door_rect = pygame.Rect(screen_x, self.y, self.width, self.height)
        pygame.draw.rect(surface, BROWN, door_rect)
        pygame.draw.rect(surface, DARK_BROWN, door_rect, 3)
        # Door arch
        arch_rect = pygame.Rect(screen_x + 2, self.y - 10, self.width - 4, 20)
        pygame.draw.ellipse(surface, BROWN, arch_rect)
        pygame.draw.ellipse(surface, DARK_BROWN, arch_rect, 3)
        # Door knob
        knob_x = screen_x + self.width - 10
        knob_y = self.y + self.height // 2
        pygame.draw.circle(surface, GOLD, (int(knob_x), int(knob_y)), 4)
        # Arrow above door
        arrow_y = self.y - 25
        pygame.draw.polygon(surface, YELLOW, [
            (screen_x + self.width // 2, arrow_y - 8),
            (screen_x + self.width // 2 - 8, arrow_y),
            (screen_x + self.width // 2 + 8, arrow_y),
        ])
        # "EXIT" text
        exit_text = font_small.render("EXIT", True, YELLOW)
        surface.blit(exit_text, (screen_x + 2, arrow_y - 25))

        # "Press DOWN" prompt when player is near
        if self.player_near and not self.reached:
            prompt = font_small.render("Press DOWN to enter", True, WHITE)
            pr = prompt.get_rect(center=(screen_x + self.width // 2, self.y - 55))
            surface.blit(prompt, pr)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


def generate_level():
    """Procedurally generate a varied but completable level."""
    platforms = []
    enemies = []
    coins = []
    exit_door = None

    # Max jump height: v²/(2g) = 144/(2*0.6) = 120 pixels
    MAX_JUMP_HEIGHT = 120
    SAFE_VERTICAL_GAP = 100  # safe margin

    ground_y = SCREEN_HEIGHT - 60

    # --- Generate varied terrain ---
    # Ground segments with gaps
    x = 0
    while x < (LEVEL_LENGTH - 2) * TILE_SIZE:
        seg_width = random.randint(3, 8) * TILE_SIZE
        if x > 0 and random.random() < 0.2:
            gap = random.randint(1, 2) * TILE_SIZE
            x += gap
            continue
        platforms.append(Platform(x, ground_y, seg_width, TILE_SIZE, "ground"))
        x += seg_width

    def platform_overlaps(new_plat, existing_platforms):
        """Check if a new platform overlaps with any existing one."""
        new_rect = new_plat.get_rect()
        for p in existing_platforms:
            if new_rect.colliderect(p.get_rect()):
                return True
        return False

    # Floating platforms at various heights
    for _ in range(40):
        px = random.randint(2, LEVEL_LENGTH - 3) * TILE_SIZE
        py = random.randint(ground_y - 200, ground_y - 30)
        pw = random.randint(1, 3) * TILE_SIZE
        ptype = random.choice(["brick", "stone", "brick"])
        new_plat = Platform(px, py, pw, TILE_SIZE // 2, ptype)
        if not platform_overlaps(new_plat, platforms):
            platforms.append(new_plat)

    # Staircase sections
    for _ in range(4):
        stair_start = random.randint(5, LEVEL_LENGTH - 15) * TILE_SIZE
        stair_height = random.randint(3, 6)
        direction = random.choice([-1, 1])
        for i in range(stair_height):
            sy = ground_y + direction * (i + 1) * (TILE_SIZE // 2)
            sy = max(ground_y - 200, min(ground_y, sy))
            new_plat = Platform(
                stair_start + i * TILE_SIZE, sy,
                TILE_SIZE, TILE_SIZE // 2, "stone"
            )
            if not platform_overlaps(new_plat, platforms):
                platforms.append(new_plat)

    # --- Ensure completability by adding bridge platforms ---
    # Sort platforms by x position
    platforms.sort(key=lambda p: p.x)

    # Check each platform is reachable from a previous one
    # We do this by checking vertical gaps between nearby platforms
    reachable = {0: True}  # index 0 (first ground) is always reachable
    bridges_added = []
    for i, plat in enumerate(platforms):
        if i == 0:
            continue
        # Check if this platform is reachable from any earlier platform
        can_reach = False
        for j in range(i):
            if not reachable.get(j, False):
                continue
            prev = platforms[j]
            # Horizontal distance
            h_dist = plat.x - (prev.x + prev.width)
            if h_dist < -TILE_SIZE:  # platform is behind, skip
                continue
            # Vertical distance
            v_dist = prev.y - plat.y  # positive = plat is higher
            # Check if jumpable: need to be within jump height
            # and horizontal gap should be traversable
            if v_dist > SAFE_VERTICAL_GAP:
                continue  # too high to jump
            if v_dist < -SAFE_VERTICAL_GAP:
                continue  # too far down (would take damage or can't reach back)

            # Check horizontal gap - player can jump about 3-4 tiles horizontally
            if h_dist > 4 * TILE_SIZE:
                continue

            can_reach = True
            break

        reachable[i] = can_reach

        # If not reachable, add a bridge platform
        if not can_reach:
            # Find the closest reachable platform to the left
            best_j = -1
            best_dist = float('inf')
            for j in range(i):
                if not reachable.get(j, False):
                    continue
                prev = platforms[j]
                dist = plat.x - (prev.x + prev.width)
                if 0 < dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0:
                prev = platforms[best_j]
                # Add stepping stone platforms between prev and plat
                steps_needed = max(1, best_dist // (3 * TILE_SIZE))
                for step in range(1, steps_needed + 1):
                    t = step / (steps_needed + 1)
                    bridge_x = int(prev.x + prev.width + t * (plat.x - (prev.x + prev.width)))
                    bridge_y = int(prev.y + t * (plat.y - prev.y))
                    # Clamp bridge y to be reachable
                    bridge_y = max(ground_y - 200, min(ground_y, bridge_y))
                    bridge = Platform(bridge_x, bridge_y, TILE_SIZE, TILE_SIZE // 2, "stone")
                    # Check bridge doesn't overlap existing platforms
                    if not platform_overlaps(bridge, platforms):
                        bridges_added.append(bridge)

    # Add all bridges at once (after the loop to avoid modifying during iteration)
    for bridge in bridges_added:
        platforms.append(bridge)
        reachable[len(platforms) - 1] = True

    # --- Place enemies on platforms (avoid overlaps) ---
    for plat in platforms:
        if random.random() < 0.2 and plat.width >= TILE_SIZE:
            ex = plat.x + random.randint(0, max(0, plat.width - 30))
            ey = plat.y - 30
            new_enemy = Enemy(ex, ey)
            # Check enemy doesn't overlap with existing enemies
            overlap = False
            for e in enemies:
                if new_enemy.get_rect().colliderect(e.get_rect()):
                    overlap = True
                    break
            if not overlap:
                enemies.append(new_enemy)

    # --- Place exactly NUM_COINS coins on reachable platforms ---
    NUM_COINS = 40
    reachable_platforms = [p for i, p in enumerate(platforms) if reachable.get(i, False)]
    if not reachable_platforms:
        reachable_platforms = platforms

    def coin_is_reachable(cx, cy, platforms):
        """Check if a coin position is reachable (not inside or trapped between platforms)."""
        coin_rect = pygame.Rect(cx - 8, cy - 8, 16, 16)
        # Check if coin is inside any platform
        for p in platforms:
            if coin_rect.colliderect(p.get_rect()):
                return False
        # Check if there's a platform directly above the coin (blocking access from above)
        # or if the gap between platforms is too small for the player to fit
        player_clearance = PLAYER_HEIGHT + 10
        for p in platforms:
            if p.x < cx < p.x + p.width:
                # Platform above the coin (coin is below the platform's bottom edge)
                if p.y <= cy <= p.y + p.height:
                    return False  # coin is inside or directly under a platform
                # Two platforms sandwiching the coin vertically with too small a gap
                if p.y > cy and p.y - cy < player_clearance:
                    return False  # too tight to fit between this platform and the one below
        return True

    attempts = 0
    while len(coins) < NUM_COINS and attempts < NUM_COINS * 10:
        attempts += 1
        plat = random.choice(reachable_platforms)
        coin_x = plat.x + random.randint(10, max(11, plat.width - 10))
        # Mix of coins on platforms and in the air
        if random.random() < 0.5:
            coin_y = plat.y - 25
        else:
            coin_y = plat.y - random.randint(25, 60)
        if coin_is_reachable(coin_x, coin_y, platforms):
            coins.append(Coin(coin_x, coin_y))

    # --- Exit door at the rightmost platform ---
    rightmost = max(platforms, key=lambda p: p.x)
    exit_door = Exit(rightmost.x + rightmost.width // 2 - 20, rightmost.y - 60)

    return platforms, enemies, coins, exit_door


def check_collisions(player, platforms, enemies, coins, exit_door):
    """Handle all collision logic. Returns True if player lost a life."""
    player.on_ground = False
    lost_life = False

    # Platform collisions - use fresh rect each time to handle multi-platform contact
    for plat in platforms:
        if not player.get_rect().colliderect(plat.get_rect()):
            continue

        # Determine overlap amounts
        overlap_left = (player.x + player.width) - plat.x
        overlap_right = (plat.x + plat.width) - player.x
        overlap_top = (player.y + player.height) - plat.y
        overlap_bottom = (plat.y + plat.height) - player.y

        # Find smallest overlap to determine collision side
        min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

        if min_overlap == overlap_top and player.vel_y >= 0:
            # Landing on top of platform
            player.y = plat.y - player.height
            player.vel_y = 0
            player.on_ground = True
        elif min_overlap == overlap_bottom and player.vel_y <= 0:
            # Hitting head on bottom of platform
            player.y = plat.y + plat.height
            player.vel_y = 0
        elif min_overlap == overlap_left:
            # Hitting left side
            player.x = plat.x - player.width
            player.vel_x = 0
        elif min_overlap == overlap_right:
            # Hitting right side
            player.x = plat.x + plat.width
            player.vel_x = 0

    # Enemy collisions
    player_rect = player.get_rect()
    for enemy in enemies:
        if not enemy.alive:
            continue
        enemy_rect = enemy.get_rect()

        # Check axe hit
        axe_hitbox = player.get_axe_hitbox()
        if axe_hitbox and axe_hitbox.colliderect(enemy_rect):
            enemy.alive = False
            continue

        # Check player collision with enemy
        if player_rect.colliderect(enemy_rect) and player.invincible == 0:
            # Check if player is falling onto the enemy (stomp kill like Mario)
            if player.vel_y > 0 and player.y + player.height - enemy.y < 20:
                # Stomp the enemy!
                enemy.alive = False
                player.vel_y = -8  # Bounce up
            else:
                # Take damage from side/bottom collision
                player.invincible = 30
                player.vel_y = -8
                player.vel_x = -5 if player.facing_right else 5
                player.health -= 1
                if player.health <= 0:
                    player.lives -= 1
                    lost_life = True

    # Coin collection
    for coin in coins:
        if not coin.collected and player.get_rect().colliderect(coin.get_rect()):
            coin.collected = True

    # Exit check - just mark proximity, actual activation requires DOWN key
    if exit_door and player.get_rect().colliderect(exit_door.get_rect()):
        exit_door.player_near = True
    elif exit_door:
        exit_door.player_near = False

    return lost_life


def draw_background(surface, cam):
    """Draw parallax background."""
    # Sky - solid fill for performance
    surface.fill(SKY_BLUE)

    # Mountains (parallax) - fixed heights for performance
    mountain_color = (100, 150, 100)
    mountain_heights = [120, 90, 140, 80, 110, 130, 95, 115, 100, 125, 85]
    for i, mx in enumerate(range(0, SCREEN_WIDTH + 50, 80)):
        mh = mountain_heights[i % len(mountain_heights)]
        pygame.draw.polygon(surface, mountain_color, [
            (mx - 60, SCREEN_HEIGHT - 60),
            (mx, SCREEN_HEIGHT - 60 - mh),
            (mx + 60, SCREEN_HEIGHT - 60),
        ])

    # Clouds (parallax)
    cloud_color = (240, 240, 255)
    for i in range(5):
        cx = (i * 200 + cam.x * 0.1) % (SCREEN_WIDTH + 100) - 50
        cy = 30 + i * 20
        pygame.draw.ellipse(surface, cloud_color, (cx, cy, 60, 25))
        pygame.draw.ellipse(surface, cloud_color, (cx + 15, cy - 8, 40, 20))
        pygame.draw.ellipse(surface, cloud_color, (cx + 35, cy, 30, 18))


# Pre-created HUD background surface
hud_bg = pygame.Surface((SCREEN_WIDTH, 40))
hud_bg.set_alpha(120)
hud_bg.fill(BLACK)


def show_hud(surface, player, coins_collected, total_coins, level):
    """Display HUD with health bar and lives."""
    surface.blit(hud_bg, (0, 0))

    coin_text = font_small.render(f"Coins: {coins_collected}/{total_coins}", True, GOLD)
    surface.blit(coin_text, (10, 8))

    level_text = font_small.render(f"Level {level}", True, WHITE)
    surface.blit(level_text, (SCREEN_WIDTH // 2 - 30, 8))

    # Health bar
    bar_x = SCREEN_WIDTH - 200
    bar_y = 8
    bar_width = 80
    bar_height = 12
    # Background
    pygame.draw.rect(surface, (60, 0, 0), (bar_x, bar_y, bar_width, bar_height))
    pygame.draw.rect(surface, WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
    # Health fill
    health_pct = player.health / player.max_health
    fill_width = int(bar_width * health_pct)
    health_color = GREEN if health_pct > 0.5 else (YELLOW if health_pct > 0.25 else RED)
    pygame.draw.rect(surface, health_color, (bar_x + 1, bar_y + 1, fill_width - 2, bar_height - 2))

    # Lives (heart icons)
    heart_x = bar_x - 50
    for i in range(player.lives):
        hx = heart_x + i * 16
        hy = bar_y + 2
        pygame.draw.polygon(surface, RED, [
            (hx + 4, hy + 2), (hx + 2, hy + 5), (hx + 4, hy + 9),
            (hx + 7, hy + 6), (hx + 10, hy + 9), (hx + 12, hy + 5),
            (hx + 10, hy + 2), (hx + 7, hy + 4),
        ])

    # Axe indicator
    axe_text = font_small.render("[SPACE] Axe", True, (200, 200, 200))
    surface.blit(axe_text, (SCREEN_WIDTH - 120, 8))


def show_level_complete(surface, coins_collected, total_coins):
    """Display level complete screen."""
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill(BLACK)
    surface.blit(overlay, (0, 0))

    complete_text = font_large.render("LEVEL COMPLETE!", True, GOLD)
    complete_shadow = font_large.render("LEVEL COMPLETE!", True, (150, 120, 0))
    cr = complete_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60))
    surface.blit(complete_shadow, (cr.x + 3, cr.y + 3))
    surface.blit(complete_text, cr)

    coin_text = font_medium.render(f"Coins: {coins_collected}/{total_coins}", True, GOLD)
    cr2 = coin_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    surface.blit(coin_text, cr2)

    next_text = font_medium.render("Press SPACE for Next Level", True, WHITE)
    cr3 = next_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60))
    surface.blit(next_text, cr3)

    pygame.display.flip()


def show_game_over(surface, coins_collected):
    """Display game over screen."""
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill(BLACK)
    surface.blit(overlay, (0, 0))

    go_text = font_large.render("GAME OVER", True, RED)
    go_shadow = font_large.render("GAME OVER", True, (100, 0, 0))
    gr = go_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60))
    surface.blit(go_shadow, (gr.x + 3, gr.y + 3))
    surface.blit(go_text, gr)

    coin_text = font_medium.render(f"Coins: {coins_collected}", True, GOLD)
    cr = coin_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    surface.blit(coin_text, cr)

    restart_text = font_medium.render("Press SPACE to Restart", True, WHITE)
    rr = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60))
    surface.blit(restart_text, rr)

    pygame.display.flip()


def show_start_screen(surface):
    """Display start screen."""
    draw_background(surface, Camera())

    title_text = font_large.render("AXE JUMPER", True, ORANGE)
    title_shadow = font_large.render("AXE JUMPER", True, (150, 80, 0))
    tr = title_text.get_rect(center=(SCREEN_WIDTH // 2, 120))
    surface.blit(title_shadow, (tr.x + 3, tr.y + 3))
    surface.blit(title_text, tr)

    subtitle = font_medium.render("A Side-Scrolling Platformer", True, WHITE)
    sr = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 170))
    surface.blit(subtitle, sr)

    # Instructions
    instructions = [
        "Arrow Keys / WASD - Move and Jump",
        "SPACE - Swing Axe (kill enemies!)",
        "Collect coins and reach the EXIT!",
    ]
    for i, inst in enumerate(instructions):
        text = font_small.render(inst, True, (220, 220, 220))
        tr = text.get_rect(center=(SCREEN_WIDTH // 2, 260 + i * 30))
        surface.blit(text, tr)

    start_text = font_medium.render("Press SPACE to Start", True, YELLOW)
    str_ = start_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100))
    surface.blit(start_text, str_)

    pygame.display.flip()


def main():
    """Main game loop."""
    level = 1
    total_coins_all = 0
    running = True
    game_state = "start"

    while running:
        # Generate level
        platforms, enemies, coins, exit_door = generate_level()
        player = Player(100, SCREEN_HEIGHT - 120)
        camera = Camera()
        coins_collected = 0
        total_coins = len(coins)
        level_complete = False
        game_over = False

        restart_level = False
        life_lost_pending = False
        while game_state == "playing" and running:
            clock.tick(FPS)

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    game_state = "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        game_state = "quit"
                    if event.key == pygame.K_SPACE:
                        if level_complete:
                            level += 1
                            total_coins_all += coins_collected
                            restart_level = True
                        elif game_over:
                            level = 1
                            total_coins_all = 0
                            restart_level = True
                        elif life_lost_pending:
                            # Respawn instantly at start
                            player.health = player.max_health
                            player.x = 100
                            player.y = SCREEN_HEIGHT - 120
                            player.vel_x = 0
                            player.vel_y = 0
                            player.invincible = 60
                            life_lost_pending = False
                        else:
                            player.swing_axe()

            if game_state == "quit":
                break

            if restart_level:
                break

            if level_complete or game_over or life_lost_pending:
                # Still draw the frame even when paused
                if life_lost_pending:
                    draw_background(screen, camera)
                    for plat in platforms:
                        plat.draw(screen, camera)
                    for coin in coins:
                        coin.draw(screen, camera)
                    for enemy in enemies:
                        enemy.draw(screen, camera)
                    if exit_door:
                        exit_door.draw(screen, camera)
                    player.draw(screen, camera)
                    show_hud(screen, player, coins_collected, total_coins, level)

                    # Life lost overlay
                    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                    overlay.set_alpha(180)
                    overlay.fill(BLACK)
                    screen.blit(overlay, (0, 0))

                    life_text = font_large.render("LIFE LOST!", True, RED)
                    life_shadow = font_large.render("LIFE LOST!", True, (100, 0, 0))
                    lr = life_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
                    screen.blit(life_shadow, (lr.x + 3, lr.y + 3))
                    screen.blit(life_text, lr)

                    lives_text = font_medium.render(f"{player.lives} lives remaining", True, WHITE)
                    lr2 = lives_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
                    screen.blit(lives_text, lr2)

                    continue_text = font_small.render("Press SPACE to continue", True, YELLOW)
                    lr3 = continue_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60))
                    screen.blit(continue_text, lr3)

                    pygame.display.flip()
                continue

            # Input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                player.vel_x = -MOVE_SPEED
                player.facing_right = False
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                player.vel_x = MOVE_SPEED
                player.facing_right = True
            else:
                player.vel_x = 0

            if keys[pygame.K_UP] or keys[pygame.K_w]:
                player.jump()

            # Update
            player.update()
            camera.update(player)

            for enemy in enemies:
                enemy.update(platforms)

            lost_life = check_collisions(player, platforms, enemies, coins, exit_door)

            # Handle life lost
            if lost_life:
                if player.lives <= 0:
                    game_over = True
                else:
                    life_lost_pending = True

            # Count collected coins
            coins_collected = sum(1 for c in coins if c.collected)

            # Check death (fell off world)
            if player.y > SCREEN_HEIGHT + 100:
                player.lives -= 1
                if player.lives <= 0:
                    game_over = True
                else:
                    life_lost_pending = True

            # Check exit (activated by pressing DOWN when near)
            if exit_door and exit_door.player_near and (keys[pygame.K_DOWN] or keys[pygame.K_s]):
                exit_door.reached = True

            if exit_door and exit_door.reached:
                level_complete = True

            # Draw
            draw_background(screen, camera)

            # Draw platforms
            for plat in platforms:
                plat.draw(screen, camera)

            # Draw coins
            for coin in coins:
                coin.draw(screen, camera)

            # Draw enemies
            for enemy in enemies:
                enemy.draw(screen, camera)

            # Draw exit
            if exit_door:
                exit_door.draw(screen, camera)

            # Draw player
            player.draw(screen, camera)

            # Draw HUD
            show_hud(screen, player, coins_collected, total_coins, level)

            # Draw level complete or game over
            if level_complete:
                show_level_complete(screen, coins_collected, total_coins)
            elif game_over:
                show_game_over(screen, coins_collected)

            pygame.display.flip()

        if game_state == "start":
            show_start_screen(screen)
            waiting = True
            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                        if event.key == pygame.K_SPACE:
                            game_state = "playing"
                            waiting = False

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
