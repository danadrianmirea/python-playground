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

    def update(self):
        self.vel_y += GRAVITY
        self.x += self.vel_x
        self.y += self.vel_y

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
                if self.vel_y > 0:
                    self.y = plat.y - self.height
                    self.vel_y = 0
                    self.on_ground = True

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

        # Bounce off walls
        if self.x < 0 or self.x > LEVEL_LENGTH * TILE_SIZE - self.width:
            self.vel_x *= -1

    def draw(self, surface, cam):
        screen_x = self.x - cam.x

        if not self.alive:
            if self.death_timer < 20:
                # Death animation - fade out
                alpha = max(0, 255 - self.death_timer * 12)
                s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                if self.type == "slime":
                    pygame.draw.ellipse(s, (100, 255, 100, alpha), (0, 0, self.width, self.height))
                elif self.type == "bat":
                    pygame.draw.ellipse(s, (100, 100, 255, alpha), (0, 0, self.width, self.height))
                else:
                    pygame.draw.ellipse(s, (255, 100, 100, alpha), (0, 0, self.width, self.height))
                surface.blit(s, (screen_x, self.y))
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
            # Wings
            wing_flap = math.sin(pygame.time.get_ticks() * 0.01) * 5
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
        # Glow effect
        glow = int(50 + 30 * math.sin(pygame.time.get_ticks() * 0.005))
        pygame.draw.circle(surface, (255, 255, 200, glow), (int(screen_x), int(self.y)), self.radius + 4)
        # Coin
        pygame.draw.circle(surface, GOLD, (int(screen_x), int(self.y)), self.radius)
        pygame.draw.circle(surface, (200, 170, 0), (int(screen_x), int(self.y)), self.radius, 2)
        # Dollar sign
        font = pygame.font.SysFont("Arial", 12, bold=True)
        text = font.render("$", True, (200, 150, 0))
        surface.blit(text, (screen_x - 4, self.y - 6))

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

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


def generate_level():
    """Procedurally generate a level."""
    platforms = []
    enemies = []
    coins = []
    exit_door = None

    # Ground segments (with gaps)
    ground_y = SCREEN_HEIGHT - 60
    x = 0
    while x < LEVEL_LENGTH * TILE_SIZE:
        segment_width = random.randint(3, 8) * TILE_SIZE
        if x > 0 and random.random() < 0.2:
            # Gap
            gap_width = random.randint(1, 2) * TILE_SIZE
            x += gap_width
            continue
        platforms.append(Platform(x, ground_y, segment_width, TILE_SIZE, "ground"))
        x += segment_width

    # Floating platforms
    for _ in range(30):
        px = random.randint(2, LEVEL_LENGTH - 3) * TILE_SIZE
        py = random.randint(ground_y - 150, ground_y - 40)
        pw = random.randint(1, 3) * TILE_SIZE
        ptype = random.choice(["brick", "stone", "brick"])
        platforms.append(Platform(px, py, pw, TILE_SIZE // 2, ptype))

    # Staircase sections
    for _ in range(3):
        stair_start = random.randint(5, LEVEL_LENGTH - 15) * TILE_SIZE
        stair_height = random.randint(3, 6)
        for i in range(stair_height):
            platforms.append(Platform(
                stair_start + i * TILE_SIZE,
                ground_y - (i + 1) * TILE_SIZE,
                TILE_SIZE, TILE_SIZE, "stone"
            ))

    # Enemies
    for _ in range(15):
        ex = random.randint(3, LEVEL_LENGTH - 3) * TILE_SIZE
        ey = ground_y - TILE_SIZE - random.randint(0, 100)
        enemies.append(Enemy(ex, ey))

    # Coins
    for _ in range(30):
        cx = random.randint(2, LEVEL_LENGTH - 2) * TILE_SIZE
        cy = random.randint(ground_y - 200, ground_y - 40)
        coins.append(Coin(cx, cy))

    # Place coins along paths (above platforms)
    for plat in platforms:
        if random.random() < 0.3:
            coin_x = plat.x + random.randint(10, plat.width - 10)
            coin_y = plat.y - 25
            coins.append(Coin(coin_x, coin_y))

    # Exit door at the end
    exit_door = Exit((LEVEL_LENGTH - 2) * TILE_SIZE, ground_y - 60)

    return platforms, enemies, coins, exit_door


def check_collisions(player, platforms, enemies, coins, exit_door):
    """Handle all collision logic."""
    player.on_ground = False

    # Platform collisions - vertical first
    player_rect = player.get_rect()
    for plat in platforms:
        plat_rect = plat.get_rect()
        if player_rect.colliderect(plat_rect):
            if player.vel_y > 0:  # Falling
                player.y = plat.y - player.height
                player.vel_y = 0
                player.on_ground = True
            elif player.vel_y < 0:  # Jumping up into platform
                player.y = plat.y + plat.height
                player.vel_y = 0

    # Side collision with platforms (only if not standing on top)
    player_rect = player.get_rect()
    for plat in platforms:
        plat_rect = plat.get_rect()
        if player_rect.colliderect(plat_rect):
            # Only resolve side collision if player isn't standing on this platform
            if player.y + player.height > plat.y + 5:
                if player.vel_x > 0:
                    player.x = plat.x - player.width
                elif player.vel_x < 0:
                    player.x = plat.x + plat.width

    # Enemy collisions
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
            player.invincible = 30
            player.vel_y = -8
            player.vel_x = -5 if player.facing_right else 5

    # Coin collection
    for coin in coins:
        if not coin.collected and player_rect.colliderect(coin.get_rect()):
            coin.collected = True

    # Exit check
    if exit_door and player_rect.colliderect(exit_door.get_rect()):
        exit_door.reached = True


def draw_background(surface, cam):
    """Draw parallax background."""
    # Sky - solid fill for performance
    surface.fill(SKY_BLUE)

    # Mountains (parallax)
    mountain_color = (100, 150, 100)
    for mx in range(0, SCREEN_WIDTH + 50, 80):
        mh = 100 + 50 * math.sin((mx + cam.x * 0.2) * 0.01)
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


def show_hud(surface, coins_collected, total_coins, level):
    """Display HUD."""
    hud_bg = pygame.Surface((SCREEN_WIDTH, 40))
    hud_bg.set_alpha(120)
    hud_bg.fill(BLACK)
    surface.blit(hud_bg, (0, 0))

    coin_text = font_small.render(f"Coins: {coins_collected}/{total_coins}", True, GOLD)
    surface.blit(coin_text, (10, 8))

    level_text = font_small.render(f"Level {level}", True, WHITE)
    surface.blit(level_text, (SCREEN_WIDTH // 2 - 30, 8))

    # Axe indicator
    axe_text = font_small.render("[SPACE] Swing Axe", True, (200, 200, 200))
    surface.blit(axe_text, (SCREEN_WIDTH - 160, 8))


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
                            break  # Restart loop with new level
                        elif game_over:
                            level = 1
                            total_coins_all = 0
                            break  # Restart loop
                        else:
                            player.swing_axe()

            if game_state == "quit":
                break

            if level_complete or game_over:
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

            check_collisions(player, platforms, enemies, coins, exit_door)

            # Count collected coins
            coins_collected = sum(1 for c in coins if c.collected)

            # Check death (fell off world)
            if player.y > SCREEN_HEIGHT + 100:
                game_over = True

            # Check exit
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
            show_hud(screen, coins_collected, total_coins, level)

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
