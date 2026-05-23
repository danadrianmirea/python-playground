"""
Paratroopers - A retro ZX Spectrum classic game.

In the bottom center of the screen there is a turret you control with
the arrow keys (or A/D) to aim and Space to fire.
In the upper part of the screen planes fly either from the left or the right,
and sometimes they drop a paratrooper.
If three paratroopers land to the left of the turret, or three paratroopers
land to the right, the player loses because the turret is destroyed.
The player has to fire either on the planes before they drop paratroopers
or on the paratroopers themselves to stop them.
"""

import pygame
import random
import math
import sys

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Colors
SKY_BLUE = (100, 160, 220)
DARK_BLUE = (40, 60, 120)
GREEN = (40, 160, 40)
DARK_GREEN = (20, 100, 20)
BROWN = (120, 80, 40)
DARK_BROWN = (80, 50, 20)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 50, 50)
YELLOW = (255, 220, 50)
ORANGE = (255, 160, 20)
GRAY = (160, 160, 160)
DARK_GRAY = (80, 80, 80)
SKIN = (220, 180, 140)
PARA_GREEN = (60, 140, 60)
PARA_WHITE = (240, 240, 240)

# Game dimensions
GROUND_HEIGHT = 40
TURRET_BASE_Y = WINDOW_HEIGHT - GROUND_HEIGHT
TURRET_CENTER_X = WINDOW_WIDTH // 2
TURRET_CENTER_Y = TURRET_BASE_Y - 10

# Plane constants
PLANE_WIDTH = 50
PLANE_HEIGHT = 16
PLANE_SPEED = 3
PLANE_Y_MIN = 40
PLANE_Y_MAX = 180

# Paratrooper constants
PARA_WIDTH = 12
PARA_HEIGHT = 20
PARA_FALL_SPEED = 1.5
PARA_CHUTE_SPEED = 0.8
PARA_SWING_AMPLITUDE = 15
PARA_SWING_SPEED = 0.03

# Bullet constants
BULLET_SPEED = 8
BULLET_RADIUS = 3

# Turret constants
TURRET_BARREL_LENGTH = 35
TURRET_ANGLE_MIN = -80  # degrees from vertical (left)
TURRET_ANGLE_MAX = 80   # degrees from vertical (right)
TURRET_ROTATION_SPEED = 2  # degrees per frame

# Gameplay
MAX_LANDED_PARAS = 3  # paratroopers on one side to lose
PARA_DROP_CHANCE = 0.006  # per frame per plane
PLANE_SPAWN_INTERVAL = 90  # frames between plane spawns (~1.5 seconds)
SHOOT_COOLDOWN = 15  # frames between shots


class Turret:
    """The player's turret at the bottom center of the screen."""

    def __init__(self):
        self.x = TURRET_CENTER_X
        self.y = TURRET_CENTER_Y
        self.angle = 0  # degrees, 0 = straight up, negative = left, positive = right
        self.shoot_cooldown = 0

    def aim_left(self):
        self.angle = max(TURRET_ANGLE_MIN, self.angle - TURRET_ROTATION_SPEED)

    def aim_right(self):
        self.angle = min(TURRET_ANGLE_MAX, self.angle + TURRET_ROTATION_SPEED)

    def get_barrel_end(self):
        """Get the position of the barrel tip."""
        rad = math.radians(self.angle - 90)  # -90 because 0 is up in screen coords
        ex = self.x + math.cos(rad) * TURRET_BARREL_LENGTH
        ey = self.y + math.sin(rad) * TURRET_BARREL_LENGTH
        return (ex, ey)

    def get_fire_direction(self):
        """Get the direction vector of the barrel."""
        rad = math.radians(self.angle - 90)
        return (math.cos(rad), math.sin(rad))

    def shoot(self):
        """Fire a bullet. Returns a Bullet or None if on cooldown."""
        if self.shoot_cooldown > 0:
            return None
        self.shoot_cooldown = SHOOT_COOLDOWN
        bx, by = self.get_barrel_end()
        dx, dy = self.get_fire_direction()
        return Bullet(bx, by, dx * BULLET_SPEED, dy * BULLET_SPEED)

    def update(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

    def draw(self, screen):
        # Base platform
        base_rect = pygame.Rect(self.x - 30, TURRET_BASE_Y - 8, 60, 12)
        pygame.draw.rect(screen, DARK_GRAY, base_rect, border_radius=3)
        pygame.draw.rect(screen, GRAY, base_rect, 2, border_radius=3)

        # Turret base circle
        pygame.draw.circle(screen, DARK_GRAY, (self.x, self.y), 12)
        pygame.draw.circle(screen, GRAY, (self.x, self.y), 12, 2)

        # Barrel
        ex, ey = self.get_barrel_end()
        pygame.draw.line(screen, DARK_GRAY, (self.x, self.y), (int(ex), int(ey)), 6)
        pygame.draw.line(screen, GRAY, (self.x, self.y), (int(ex), int(ey)), 3)

        # Barrel tip
        pygame.draw.circle(screen, RED, (int(ex), int(ey)), 4)

        # Center dot
        pygame.draw.circle(screen, RED, (self.x, self.y), 3)


class Bullet:
    """A bullet fired from the turret."""

    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.active = True
        self.trail = []

    def update(self):
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 10:
            self.trail.pop(0)

        self.x += self.vx
        self.y += self.vy

        # Check bounds
        if (self.x < -20 or self.x > WINDOW_WIDTH + 20 or
                self.y < -20 or self.y > WINDOW_HEIGHT + 20):
            self.active = False

    def get_rect(self):
        return pygame.Rect(self.x - BULLET_RADIUS, self.y - BULLET_RADIUS,
                          BULLET_RADIUS * 2, BULLET_RADIUS * 2)

    def draw(self, screen):
        # Draw trail
        for i, pos in enumerate(self.trail):
            alpha = i / len(self.trail) if self.trail else 0
            size = max(1, int(3 * alpha))
            if size > 0:
                pygame.draw.circle(screen, YELLOW, pos, size)

        # Draw bullet
        if self.active:
            pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), BULLET_RADIUS)
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), BULLET_RADIUS - 1)


class Plane:
    """An enemy plane that flies across the screen and drops paratroopers."""

    def __init__(self, direction):
        self.direction = direction  # 1 = right to left, -1 = left to right
        self.width = PLANE_WIDTH
        self.height = PLANE_HEIGHT
        self.y = random.randint(PLANE_Y_MIN, PLANE_Y_MAX)

        if direction == 1:  # flying left (spawn on right)
            self.x = WINDOW_WIDTH + 20
        else:  # flying right (spawn on left)
            self.x = -20 - PLANE_WIDTH

        self.speed = -PLANE_SPEED * direction
        self.active = True
        self.has_dropped = False

    def update(self):
        self.x += self.speed

        # Check if off screen
        if self.x < -60 or self.x > WINDOW_WIDTH + 60:
            self.active = False

    def try_drop_paratrooper(self):
        """Randomly decide to drop a paratrooper. Returns a Paratrooper or None."""
        if self.has_dropped:
            return None

        # Only drop when roughly in the middle 60% of the screen
        if self.x < WINDOW_WIDTH * 0.2 or self.x > WINDOW_WIDTH * 0.8:
            return None

        if random.random() < PARA_DROP_CHANCE:
            self.has_dropped = True
            return Paratrooper(self.x + self.width // 2, self.y + self.height)

        return None

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, screen):
        # Plane body
        body_color = (180, 180, 180)
        if self.direction == 1:  # flying left
            # Body
            pygame.draw.ellipse(screen, body_color,
                              (self.x, self.y + 3, self.width, self.height - 6))
            # Nose (pointing left)
            pygame.draw.polygon(screen, body_color, [
                (self.x, self.y + self.height // 2),
                (self.x - 10, self.y + self.height // 2 - 4),
                (self.x - 10, self.y + self.height // 2 + 4),
            ])
            # Tail
            pygame.draw.polygon(screen, body_color, [
                (self.x + self.width, self.y + 2),
                (self.x + self.width + 8, self.y - 4),
                (self.x + self.width + 8, self.y + 8),
            ])
            # Wings
            pygame.draw.polygon(screen, DARK_GRAY, [
                (self.x + 15, self.y + self.height // 2),
                (self.x + 25, self.y - 6),
                (self.x + 35, self.y - 6),
                (self.x + 25, self.y + self.height // 2),
            ])
            pygame.draw.polygon(screen, DARK_GRAY, [
                (self.x + 15, self.y + self.height // 2),
                (self.x + 25, self.y + self.height + 2),
                (self.x + 35, self.y + self.height + 2),
                (self.x + 25, self.y + self.height // 2),
            ])
            # Cockpit
            pygame.draw.ellipse(screen, (100, 180, 255),
                              (self.x + 5, self.y + 4, 12, 8))
        else:  # flying right
            # Body
            pygame.draw.ellipse(screen, body_color,
                              (self.x, self.y + 3, self.width, self.height - 6))
            # Nose (pointing right)
            pygame.draw.polygon(screen, body_color, [
                (self.x + self.width, self.y + self.height // 2),
                (self.x + self.width + 10, self.y + self.height // 2 - 4),
                (self.x + self.width + 10, self.y + self.height // 2 + 4),
            ])
            # Tail
            pygame.draw.polygon(screen, body_color, [
                (self.x, self.y + 2),
                (self.x - 8, self.y - 4),
                (self.x - 8, self.y + 8),
            ])
            # Wings
            pygame.draw.polygon(screen, DARK_GRAY, [
                (self.x + 15, self.y + self.height // 2),
                (self.x + 25, self.y - 6),
                (self.x + 35, self.y - 6),
                (self.x + 25, self.y + self.height // 2),
            ])
            pygame.draw.polygon(screen, DARK_GRAY, [
                (self.x + 15, self.y + self.height // 2),
                (self.x + 25, self.y + self.height + 2),
                (self.x + 35, self.y + self.height + 2),
                (self.x + 25, self.y + self.height // 2),
            ])
            # Cockpit
            pygame.draw.ellipse(screen, (100, 180, 255),
                              (self.x + self.width - 17, self.y + 4, 12, 8))

        # Window/detail line
        pygame.draw.line(screen, DARK_GRAY,
                        (self.x + 5, self.y + self.height // 2),
                        (self.x + self.width - 5, self.y + self.height // 2), 1)


class Paratrooper:
    """A paratrooper falling from a plane."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PARA_WIDTH
        self.height = PARA_HEIGHT
        self.chute_open = True
        self.landed = False
        self.active = True
        self.swing_offset = random.uniform(0, math.pi * 2)
        self.swing_x = 0

        # Determine which side of the turret this paratrooper will land on
        # based on initial x position
        self.side = "left" if x < TURRET_CENTER_X else "right"

    def update(self):
        if self.landed or not self.active:
            return

        # Swing motion
        self.swing_x = math.sin(self.swing_offset) * PARA_SWING_AMPLITUDE
        self.swing_offset += PARA_SWING_SPEED

        # Fall
        if self.chute_open:
            self.y += PARA_CHUTE_SPEED
        else:
            self.y += PARA_FALL_SPEED

        # Apply horizontal swing
        self.x += self.swing_x * 0.1

        # Check if landed
        if self.y >= TURRET_BASE_Y - self.height:
            self.y = TURRET_BASE_Y - self.height
            self.landed = True

        # Update side based on current position
        self.side = "left" if self.x < TURRET_CENTER_X else "right"

    def get_rect(self):
        return pygame.Rect(self.x - self.width // 2, self.y - self.height,
                          self.width, self.height)

    def draw(self, screen):
        if not self.active:
            return

        cx = int(self.x)
        top = int(self.y - self.height)

        if self.chute_open and not self.landed:
            # Parachute
            chute_width = 24
            chute_height = 14
            chute_top = top - chute_height

            # Parachute dome
            pygame.draw.ellipse(screen, PARA_WHITE,
                              (cx - chute_width // 2, chute_top, chute_width, chute_height))
            pygame.draw.ellipse(screen, DARK_GRAY,
                              (cx - chute_width // 2, chute_top, chute_width, chute_height), 1)

            # Parachute lines
            for offset in [-8, -3, 3, 8]:
                line_top = (cx + offset, chute_top + chute_height)
                line_bot = (cx + offset, top + 4)
                pygame.draw.line(screen, DARK_GRAY, line_top, line_bot, 1)

            # Parachute stripes
            stripe_y = chute_top + chute_height // 2
            pygame.draw.arc(screen, RED,
                          (cx - chute_width // 2, chute_top, chute_width, chute_height),
                          math.pi, 0, 2)

        # Body
        body_rect = pygame.Rect(cx - 5, top + 4, 10, 12)
        pygame.draw.rect(screen, PARA_GREEN, body_rect)
        pygame.draw.rect(screen, DARK_GREEN, body_rect, 1)

        # Head
        pygame.draw.circle(screen, SKIN, (cx, top + 2), 4)

        # Legs
        if self.landed:
            # Standing
            pygame.draw.line(screen, PARA_GREEN, (cx - 3, top + 16), (cx - 4, top + 20), 2)
            pygame.draw.line(screen, PARA_GREEN, (cx + 3, top + 16), (cx + 4, top + 20), 2)
        else:
            # Dangling
            leg_swing = math.sin(self.swing_offset * 2) * 3
            pygame.draw.line(screen, PARA_GREEN,
                           (cx - 3, top + 16), (cx - 5 + leg_swing, top + 22), 2)
            pygame.draw.line(screen, PARA_GREEN,
                           (cx + 3, top + 16), (cx + 5 + leg_swing, top + 22), 2)

        # Arms holding parachute lines
        if self.chute_open and not self.landed:
            pygame.draw.line(screen, SKIN, (cx - 5, top + 6), (cx - 8, top + 2), 2)
            pygame.draw.line(screen, SKIN, (cx + 5, top + 6), (cx + 8, top + 2), 2)

        # If landed, draw a small shadow
        if self.landed:
            pygame.draw.ellipse(screen, (0, 0, 0, 60),
                              (cx - 8, TURRET_BASE_Y - 2, 16, 4))


class Explosion:
    """A simple explosion effect."""

    def __init__(self, x, y, size=1.0):
        self.x = x
        self.y = y
        self.particles = []
        self.active = True
        self.timer = 0
        self.max_timer = 20

        num_particles = int(12 * size)
        for _ in range(num_particles):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(1, 5) * size
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'size': random.randint(2, 5),
                'color': random.choice([YELLOW, ORANGE, RED, WHITE]),
            })

    def update(self):
        self.timer += 1
        if self.timer >= self.max_timer:
            self.active = False
            return

        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1  # gravity
            p['size'] = max(0, p['size'] - 0.2)

    def draw(self, screen):
        for p in self.particles:
            if p['size'] > 0:
                pygame.draw.circle(screen, p['color'],
                                 (int(p['x']), int(p['y'])), int(p['size']))


class ParatroopersGame:
    """Main game class."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Paratroopers")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.running = True
        self.state = "menu"  # menu, playing, game_over

        self.turret = Turret()
        self.bullets = []
        self.planes = []
        self.paratroopers = []
        self.explosions = []
        self.plane_spawn_timer = 0

        # Landing counts
        self.landed_left = 0
        self.landed_right = 0

        # Score
        self.score = 0
        self.high_score = 0

        # Game over message
        self.game_over_message = ""
        self.game_over_side = ""

        # Pre-generate grass tuft positions (so they don't flicker)
        self.grass_tufts = []
        for _ in range(30):
            gx = random.randint(0, WINDOW_WIDTH)
            gy = WINDOW_HEIGHT - GROUND_HEIGHT
            self.grass_tufts.append((gx, gy,
                                     random.randint(-3, 3),
                                     random.randint(3, 6)))

        # Pre-render background to a surface (avoids per-frame gradient drawing)
        self.background = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self._render_background()

    def _render_background(self):
        """Pre-render the static background to a surface."""
        # Sky gradient
        for y in range(WINDOW_HEIGHT - GROUND_HEIGHT):
            color_ratio = y / (WINDOW_HEIGHT - GROUND_HEIGHT)
            r = int(100 + color_ratio * 30)
            g = int(160 - color_ratio * 20)
            b = int(220 - color_ratio * 40)
            pygame.draw.line(self.background, (r, g, b), (0, y), (WINDOW_WIDTH, y))

        # Ground
        ground_rect = pygame.Rect(0, WINDOW_HEIGHT - GROUND_HEIGHT,
                                 WINDOW_WIDTH, GROUND_HEIGHT)
        pygame.draw.rect(self.background, GREEN, ground_rect)
        pygame.draw.rect(self.background, DARK_GREEN, ground_rect, 2)

        # Ground texture lines
        for gx in range(0, WINDOW_WIDTH, 20):
            pygame.draw.line(self.background, DARK_GREEN,
                           (gx, WINDOW_HEIGHT - GROUND_HEIGHT),
                           (gx, WINDOW_HEIGHT), 1)

        # Some grass tufts (pre-generated positions to avoid flickering)
        for gx, gy, dx, dy in self.grass_tufts:
            pygame.draw.line(self.background, DARK_GREEN,
                           (gx, gy), (gx + dx, gy - dy), 1)

    def reset_game(self):
        """Reset the game state for a new game."""
        self.turret = Turret()
        self.bullets = []
        self.planes = []
        self.paratroopers = []
        self.explosions = []
        self.plane_spawn_timer = 0
        self.landed_left = 0
        self.landed_right = 0
        self.score = 0
        self.game_over_message = ""
        self.game_over_side = ""
        # Spawn an initial plane immediately so the game isn't empty
        self.spawn_plane()

    def spawn_plane(self):
        """Spawn a new plane from a random direction."""
        direction = random.choice([-1, 1])
        self.planes.append(Plane(direction))

    def check_collisions(self):
        """Check bullet collisions with planes and paratroopers."""
        for bullet in self.bullets[:]:
            if not bullet.active:
                continue

            bullet_rect = bullet.get_rect()

            # Check collision with planes
            for plane in self.planes[:]:
                if not plane.active:
                    continue
                if bullet_rect.colliderect(plane.get_rect()):
                    bullet.active = False
                    plane.active = False
                    self.score += 10
                    self.explosions.append(Explosion(plane.x + plane.width // 2,
                                                    plane.y + plane.height // 2, 1.5))
                    break

            # Check collision with paratroopers
            for para in self.paratroopers[:]:
                if not para.active or para.landed:
                    continue
                if bullet_rect.colliderect(para.get_rect()):
                    bullet.active = False
                    para.active = False
                    self.score += 5
                    self.explosions.append(Explosion(para.x, para.y, 0.8))
                    break

    def update(self):
        """Update game state."""
        if self.state != "playing":
            return

        # Handle continuous key input
        keys = pygame.key.get_pressed()

        # Turret aiming
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.turret.aim_left()
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.turret.aim_right()

        # Turret firing
        if keys[pygame.K_SPACE]:
            bullet = self.turret.shoot()
            if bullet:
                self.bullets.append(bullet)

        # Update turret
        self.turret.update()

        # Spawn planes
        self.plane_spawn_timer += 1
        if self.plane_spawn_timer >= PLANE_SPAWN_INTERVAL:
            self.plane_spawn_timer = 0
            if len(self.planes) < 4:  # Max 4 planes at once
                self.spawn_plane()

        # Update planes
        for plane in self.planes[:]:
            plane.update()
            if not plane.active:
                self.planes.remove(plane)
                continue

            # Try to drop a paratrooper
            para = plane.try_drop_paratrooper()
            if para:
                self.paratroopers.append(para)

        # Update paratroopers
        for para in self.paratroopers[:]:
            para.update()
            if not para.active:
                self.paratroopers.remove(para)
                continue

            # Check if landed
            if para.landed:
                if para.side == "left":
                    self.landed_left += 1
                else:
                    self.landed_right += 1
                self.paratroopers.remove(para)

                # Check game over condition
                if self.landed_left >= MAX_LANDED_PARAS:
                    self.game_over_message = "Paratroopers overran the LEFT side!"
                    self.game_over_side = "left"
                    self.state = "game_over"
                    if self.score > self.high_score:
                        self.high_score = self.score
                    return
                elif self.landed_right >= MAX_LANDED_PARAS:
                    self.game_over_message = "Paratroopers overran the RIGHT side!"
                    self.game_over_side = "right"
                    self.state = "game_over"
                    if self.score > self.high_score:
                        self.high_score = self.score
                    return

        # Update bullets
        for bullet in self.bullets[:]:
            bullet.update()
            if not bullet.active:
                self.bullets.remove(bullet)

        # Check collisions
        self.check_collisions()

        # Update explosions
        for explosion in self.explosions[:]:
            explosion.update()
            if not explosion.active:
                self.explosions.remove(explosion)

    def draw_background(self):
        """Draw the background (sky, ground, etc.) using pre-rendered surface."""
        self.screen.blit(self.background, (0, 0))

    def draw_ui(self):
        """Draw the game UI."""
        # Top bar
        pygame.draw.rect(self.screen, (0, 0, 0, 160), (0, 0, WINDOW_WIDTH, 30))
        pygame.draw.rect(self.screen, WHITE, (0, 0, WINDOW_WIDTH, 30), 1)

        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 5))

        # High score
        high_text = self.font_small.render(f"High Score: {self.high_score}", True, YELLOW)
        high_rect = high_text.get_rect(topright=(WINDOW_WIDTH - 10, 5))
        self.screen.blit(high_text, high_rect)

        # Landing indicators
        left_text = self.font_small.render(
            f"Left: {self.landed_left}/{MAX_LANDED_PARAS}", True,
            RED if self.landed_left >= MAX_LANDED_PARAS - 1 else WHITE)
        self.screen.blit(left_text, (WINDOW_WIDTH // 2 - 100, 5))

        right_text = self.font_small.render(
            f"Right: {self.landed_right}/{MAX_LANDED_PARAS}", True,
            RED if self.landed_right >= MAX_LANDED_PARAS - 1 else WHITE)
        self.screen.blit(right_text, (WINDOW_WIDTH // 2 + 20, 5))

        # Danger zone indicators on the ground
        # Left danger zone
        if self.landed_left > 0:
            danger_width = 30 * self.landed_left
            pygame.draw.rect(self.screen, (255, 0, 0, 80),
                           (10, WINDOW_HEIGHT - GROUND_HEIGHT - 5, danger_width, 5))
            label = self.font_small.render(f"{self.landed_left}", True, RED)
            self.screen.blit(label, (15, WINDOW_HEIGHT - GROUND_HEIGHT - 22))

        # Right danger zone
        if self.landed_right > 0:
            danger_width = 30 * self.landed_right
            pygame.draw.rect(self.screen, (255, 0, 0, 80),
                           (WINDOW_WIDTH - 10 - danger_width,
                            WINDOW_HEIGHT - GROUND_HEIGHT - 5, danger_width, 5))
            label = self.font_small.render(f"{self.landed_right}", True, RED)
            label_rect = label.get_rect(topright=(WINDOW_WIDTH - 15, WINDOW_HEIGHT - GROUND_HEIGHT - 22))
            self.screen.blit(label, label_rect)

    def draw_menu(self):
        """Draw the main menu."""
        self.draw_background()

        # Title
        title = self.font_large.render("PARATROOPERS", True, YELLOW)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 120))
        shadow = self.font_large.render("PARATROOPERS", True, (100, 80, 0))
        shadow_rect = shadow.get_rect(center=(WINDOW_WIDTH // 2 + 3, 123))
        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(title, title_rect)

        subtitle = self.font_medium.render("A ZX Spectrum Classic", True, WHITE)
        sub_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 170))
        self.screen.blit(subtitle, sub_rect)

        # Instructions
        instr = [
            "Defend your turret from paratroopers!",
            "",
            "Controls:",
            "LEFT / A  - Aim turret left",
            "RIGHT / D - Aim turret right",
            "SPACE     - Fire!",
            "",
            "Shoot planes before they drop paratroopers,",
            "or shoot the paratroopers themselves!",
            f"Three paratroopers landing on one side = Game Over!",
        ]
        for i, line in enumerate(instr):
            color = YELLOW if "Controls:" in line else (220, 220, 200)
            t = self.font_small.render(line, True, color)
            t_rect = t.get_rect(center=(WINDOW_WIDTH // 2, 220 + i * 22))
            self.screen.blit(t, t_rect)

        # Play button
        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 460, 200, 50)
        mouse_pos = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mouse_pos)
        color = (60, 180, 60) if hover else (40, 140, 40)
        pygame.draw.rect(self.screen, color, btn_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, btn_rect, 2, border_radius=10)
        text = self.font_medium.render("Play Game", True, WHITE)
        text_rect = text.get_rect(center=btn_rect.center)
        self.screen.blit(text, text_rect)

        # High score
        if self.high_score > 0:
            hs_text = self.font_medium.render(f"High Score: {self.high_score}", True, YELLOW)
            hs_rect = hs_text.get_rect(center=(WINDOW_WIDTH // 2, 530))
            self.screen.blit(hs_text, hs_rect)

    def draw_game(self):
        """Draw the game screen."""
        self.draw_background()

        # Draw paratroopers (behind planes)
        for para in self.paratroopers:
            para.draw(self.screen)

        # Draw planes
        for plane in self.planes:
            plane.draw(self.screen)

        # Draw bullets
        for bullet in self.bullets:
            bullet.draw(self.screen)

        # Draw turret
        self.turret.draw(self.screen)

        # Draw explosions
        for explosion in self.explosions:
            explosion.draw(self.screen)

        # Draw UI
        self.draw_ui()

    def draw_game_over(self):
        """Draw the game over screen."""
        self.draw_game()

        # Overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(160)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        # Game Over text
        go_text = self.font_large.render("GAME OVER", True, RED)
        go_rect = go_text.get_rect(center=(WINDOW_WIDTH // 2, 180))
        shadow = self.font_large.render("GAME OVER", True, (100, 0, 0))
        shadow_rect = shadow.get_rect(center=(WINDOW_WIDTH // 2 + 3, 183))
        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(go_text, go_rect)

        # Game over message
        msg_text = self.font_medium.render(self.game_over_message, True, YELLOW)
        msg_rect = msg_text.get_rect(center=(WINDOW_WIDTH // 2, 240))
        self.screen.blit(msg_text, msg_rect)

        # Score
        score_text = self.font_medium.render(f"Final Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 290))
        self.screen.blit(score_text, score_rect)

        if self.score >= self.high_score and self.score > 0:
            hs_text = self.font_medium.render("NEW HIGH SCORE!", True, YELLOW)
            hs_rect = hs_text.get_rect(center=(WINDOW_WIDTH // 2, 330))
            self.screen.blit(hs_text, hs_rect)

        # Play Again button
        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 380, 200, 50)
        mouse_pos = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mouse_pos)
        color = (60, 180, 60) if hover else (40, 140, 40)
        pygame.draw.rect(self.screen, color, btn_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, btn_rect, 2, border_radius=10)
        text = self.font_medium.render("Play Again", True, WHITE)
        text_rect = text.get_rect(center=btn_rect.center)
        self.screen.blit(text, text_rect)

        # Main Menu button
        menu_btn = pygame.Rect(WINDOW_WIDTH // 2 - 100, 450, 200, 50)
        hover2 = menu_btn.collidepoint(mouse_pos)
        color2 = (150, 150, 150) if hover2 else (100, 100, 100)
        pygame.draw.rect(self.screen, color2, menu_btn, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, menu_btn, 2, border_radius=10)
        menu_text = self.font_medium.render("Main Menu", True, WHITE)
        menu_rect = menu_text.get_rect(center=menu_btn.center)
        self.screen.blit(menu_text, menu_rect)

        return btn_rect, menu_btn

    def run(self):
        """Main game loop."""
        while self.running:
            dt = min(self.clock.tick(FPS) / 16.667, 3.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.state == "menu":
                        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 460, 200, 50)
                        if btn_rect.collidepoint(event.pos):
                            self.reset_game()
                            self.state = "playing"

                    elif self.state == "game_over":
                        btn_rect, menu_btn = self.draw_game_over()
                        if btn_rect.collidepoint(event.pos):
                            self.reset_game()
                            self.state = "playing"
                        elif menu_btn.collidepoint(event.pos):
                            self.state = "menu"

            # Update game state
            self.update()

            # Draw based on state
            if self.state == "menu":
                self.draw_menu()
            elif self.state == "playing":
                self.draw_game()
            elif self.state == "game_over":
                self.draw_game_over()

            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = ParatroopersGame()
    game.run()
