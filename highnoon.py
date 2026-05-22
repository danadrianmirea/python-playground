"""
High Noon - A retro Wild West duel game inspired by the ZX80 era.

Two cowboys face off in a dusty town. Real-time action!
Player 1 (left): WASD to move, Space to shoot
Player 2 (right): Arrow keys to move, Enter to shoot
Obstacles like rocks, cactuses, and slowly rising chariots provide cover.
First to hit the other wins the round!
"""

import pygame
import random
import math
import sys

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
GROUND_HEIGHT = 60
COWBOY_WIDTH = 30
COWBOY_HEIGHT = 60
MOVE_SPEED = 3
BULLET_SPEED = 10
SHOOT_COOLDOWN = 60  # frames between shots (1 per second at 60 FPS)

# Colors
SKY_COLOR = (135, 180, 220)
SUN_COLOR = (255, 220, 50)
GROUND_COLOR = (180, 140, 80)
GROUND_DARK = (140, 100, 50)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 50, 50)
DARK_BROWN = (80, 50, 20)
COWBOY_HAT = (160, 100, 60)
COWBOY_BODY_P1 = (100, 150, 200)
COWBOY_BODY_P2 = (200, 100, 100)
COWBOY_BOOTS = (60, 40, 20)
CACTUS_GREEN = (60, 140, 40)
CACTUS_DARK = (40, 100, 30)
ROCK_GRAY = (130, 130, 120)
ROCK_DARK = (90, 90, 85)
CHARIOT_BROWN = (160, 120, 60)
CHARIOT_DARK = (100, 70, 30)
BULLET_COLOR = (255, 200, 50)
BULLET_TRAIL = (255, 180, 80)
DUST_COLOR = (200, 180, 140)


class Bullet:
    """A bullet fired by a cowboy - flies straight horizontally."""
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.vx = BULLET_SPEED * direction  # direction is 1 (right) or -1 (left)
        self.vy = 0  # no vertical movement
        self.active = True
        self.trail = []

    def get_rect(self):
        """Get the bounding box for collision detection."""
        return pygame.Rect(self.x - 4, self.y - 4, 8, 8)

    def update(self, dt):
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 15:
            self.trail.pop(0)

        self.x += self.vx * dt

        # Check bounds
        if self.x < -20 or self.x > WINDOW_WIDTH + 20:
            self.active = False

    def draw(self, screen):
        # Draw trail
        for i, pos in enumerate(self.trail):
            alpha = i / len(self.trail) if self.trail else 0
            size = max(1, int(3 * alpha))
            if size > 0:
                pygame.draw.circle(screen, BULLET_TRAIL, pos, size)

        # Draw bullet
        if self.active:
            pygame.draw.circle(screen, BULLET_COLOR, (int(self.x), int(self.y)), 4)
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), 2)


class Cowboy:
    """A cowboy character."""
    def __init__(self, x, y, facing_right, player_num):
        self.x = x
        self.y = y
        self.facing_right = facing_right
        self.player_num = player_num
        self.alive = True
        self.shoot_cooldown = 0
        self.body_color = COWBOY_BODY_P1 if player_num == 1 else COWBOY_BODY_P2

    def draw(self, screen):
        if not self.alive:
            return

        # Body
        body_rect = pygame.Rect(self.x - COWBOY_WIDTH // 2,
                                self.y - COWBOY_HEIGHT,
                                COWBOY_WIDTH, COWBOY_HEIGHT)
        pygame.draw.rect(screen, self.body_color, body_rect, border_radius=5)

        # Head
        head_center = (self.x, self.y - COWBOY_HEIGHT - 8)
        pygame.draw.circle(screen, (220, 180, 140), head_center, 10)

        # Hat
        hat_points = [
            (self.x - 16, self.y - COWBOY_HEIGHT - 12),
            (self.x + 16, self.y - COWBOY_HEIGHT - 12),
            (self.x + 12, self.y - COWBOY_HEIGHT - 20),
            (self.x - 12, self.y - COWBOY_HEIGHT - 20),
        ]
        pygame.draw.polygon(screen, COWBOY_HAT, hat_points)
        pygame.draw.rect(screen, COWBOY_HAT,
                        (self.x - 18, self.y - COWBOY_HEIGHT - 14, 36, 5))

        # Eyes
        eye_offset = 4 if self.facing_right else -4
        pygame.draw.circle(screen, BLACK, (self.x + eye_offset - 2, self.y - COWBOY_HEIGHT - 9), 2)
        pygame.draw.circle(screen, BLACK, (self.x + eye_offset + 3, self.y - COWBOY_HEIGHT - 9), 2)

        # Arms
        arm_dir = 1 if self.facing_right else -1
        # Left arm
        pygame.draw.line(screen, self.body_color,
                        (self.x - 12, self.y - COWBOY_HEIGHT + 15),
                        (self.x - 20, self.y - COWBOY_HEIGHT + 5), 4)
        # Right arm (gun arm)
        pygame.draw.line(screen, self.body_color,
                        (self.x + 12, self.y - COWBOY_HEIGHT + 15),
                        (self.x + 20 * arm_dir, self.y - COWBOY_HEIGHT + 5), 4)

        # Gun - always points in the facing direction
        gun_x = self.x + 22 * arm_dir
        gun_y = self.y - COWBOY_HEIGHT + 5
        pygame.draw.rect(screen, DARK_BROWN,
                        (min(gun_x, gun_x + 10 * arm_dir), gun_y - 2, 12, 5))

        # Legs
        pygame.draw.line(screen, COWBOY_BOOTS,
                        (self.x - 8, self.y),
                        (self.x - 12, self.y + 12), 5)
        pygame.draw.line(screen, COWBOY_BOOTS,
                        (self.x + 8, self.y),
                        (self.x + 12, self.y + 12), 5)

        # Boots
        pygame.draw.ellipse(screen, COWBOY_BOOTS,
                          (self.x - 16, self.y + 8, 12, 8))
        pygame.draw.ellipse(screen, COWBOY_BOOTS,
                          (self.x + 4, self.y + 8, 12, 8))

    def get_hitbox(self):
        """Get the bounding box for collision detection.
        Extended upward to cover the head/hat area."""
        return pygame.Rect(self.x - COWBOY_WIDTH // 2,
                          self.y - COWBOY_HEIGHT - 16,
                          COWBOY_WIDTH, COWBOY_HEIGHT + 16)

    def get_gun_position(self):
        """Get the position of the gun barrel tip."""
        arm_dir = 1 if self.facing_right else -1
        return (self.x + 22 * arm_dir, self.y - COWBOY_HEIGHT + 3)

    def shoot(self):
        """Fire a bullet straight in the facing direction."""
        if self.shoot_cooldown > 0:
            return None

        self.shoot_cooldown = SHOOT_COOLDOWN

        gun_x, gun_y = self.get_gun_position()
        direction = 1 if self.facing_right else -1

        return Bullet(gun_x, gun_y, direction)

    def move(self, dx, dy, obstacles, chariots):
        """Move the cowboy with sliding collision.
        Tries X movement first, then Y, so the player slides along obstacles."""
        all_solids = obstacles + chariots

        # Clamp target to screen bounds
        target_x = max(COWBOY_WIDTH // 2, min(WINDOW_WIDTH - COWBOY_WIDTH // 2, self.x + dx))
        target_y = max(COWBOY_HEIGHT, min(WINDOW_HEIGHT - GROUND_HEIGHT, self.y + dy))

        # Try moving on X axis only
        test_rect = pygame.Rect(target_x - COWBOY_WIDTH // 2,
                               self.y - COWBOY_HEIGHT,
                               COWBOY_WIDTH, COWBOY_HEIGHT)
        blocked_x = any(test_rect.colliderect(s.get_rect()) for s in all_solids)

        # Try moving on Y axis only
        test_rect = pygame.Rect(self.x - COWBOY_WIDTH // 2,
                               target_y - COWBOY_HEIGHT,
                               COWBOY_WIDTH, COWBOY_HEIGHT)
        blocked_y = any(test_rect.colliderect(s.get_rect()) for s in all_solids)

        # Apply movement: if X is blocked, keep old X; if Y is blocked, keep old Y
        if not blocked_x:
            self.x = target_x
        if not blocked_y:
            self.y = target_y


class Obstacle:
    """Base class for obstacles."""
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.active = True

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def collides_with(self, x, y, radius=5):
        """Check if a point collides with this obstacle."""
        rect = self.get_rect()
        closest_x = max(rect.left, min(x, rect.right))
        closest_y = max(rect.top, min(y, rect.bottom))
        dist = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
        return dist < radius


class Rock(Obstacle):
    """A rock obstacle."""
    def __init__(self, x, y):
        size = random.randint(20, 40)
        super().__init__(x, y, size, size * 0.6)
        self.size = size

    def draw(self, screen):
        rect = self.get_rect()
        pygame.draw.ellipse(screen, ROCK_GRAY, rect)
        highlight = pygame.Rect(rect.x + 3, rect.y + 2, rect.width * 0.4, rect.height * 0.3)
        pygame.draw.ellipse(screen, (160, 160, 150), highlight)
        pygame.draw.ellipse(screen, ROCK_DARK, rect, 2)


class Cactus(Obstacle):
    """A cactus obstacle."""
    def __init__(self, x, y):
        height = random.randint(40, 70)
        width = 12
        super().__init__(x, y, width, height)

    def draw(self, screen):
        rect = self.get_rect()
        pygame.draw.rect(screen, CACTUS_GREEN, rect, border_radius=3)
        arm_y = rect.y + rect.height * 0.3
        pygame.draw.rect(screen, CACTUS_GREEN,
                        (rect.x - 10, arm_y, 10, 8), border_radius=2)
        pygame.draw.rect(screen, CACTUS_GREEN,
                        (rect.x - 10, arm_y - 8, 8, 8), border_radius=2)
        pygame.draw.rect(screen, CACTUS_GREEN,
                        (rect.x + rect.width, arm_y + 10, 10, 8), border_radius=2)
        pygame.draw.rect(screen, CACTUS_GREEN,
                        (rect.x + rect.width + 2, arm_y + 2, 8, 8), border_radius=2)
        pygame.draw.rect(screen, CACTUS_DARK, rect, 2, border_radius=3)
        for sy in range(int(rect.y + 5), int(rect.bottom - 5), 10):
            pygame.draw.line(screen, CACTUS_DARK,
                           (rect.x, sy), (rect.x - 3, sy - 2), 1)
            pygame.draw.line(screen, CACTUS_DARK,
                           (rect.right, sy), (rect.right + 3, sy + 2), 1)


class Chariot:
    """A chariot that slowly rises from the bottom of the screen."""
    def __init__(self):
        self.width = 60
        self.height = 40
        self.x = random.randint(200, WINDOW_WIDTH - 200 - self.width)
        self.y = WINDOW_HEIGHT + 10
        self.speed = random.uniform(0.3, 0.8)
        self.active = True

    def update(self, dt):
        self.y -= self.speed * dt
        if self.y < -self.height - 20:
            self.active = False

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def collides_with(self, x, y, radius=5):
        rect = self.get_rect()
        closest_x = max(rect.left, min(x, rect.right))
        closest_y = max(rect.top, min(y, rect.bottom))
        dist = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
        return dist < radius

    def draw(self, screen):
        rect = self.get_rect()
        pygame.draw.circle(screen, CHARIOT_BROWN,
                         (rect.centerx, rect.centery + 5), 20)
        pygame.draw.circle(screen, CHARIOT_DARK,
                         (rect.centerx, rect.centery + 5), 20, 3)
        for angle in range(0, 360, 45):
            a = math.radians(angle)
            ex = rect.centerx + math.cos(a) * 15
            ey = rect.centery + 5 + math.sin(a) * 15
            pygame.draw.line(screen, CHARIOT_DARK,
                           (rect.centerx, rect.centery + 5),
                           (int(ex), int(ey)), 2)
        pygame.draw.circle(screen, CHARIOT_DARK,
                         (rect.centerx, rect.centery + 5), 4)
        pygame.draw.rect(screen, CHARIOT_BROWN,
                        (rect.x, rect.y, rect.width, 8))
        pygame.draw.rect(screen, CHARIOT_DARK,
                        (rect.x, rect.y, rect.width, 8), 2)
        pygame.draw.line(screen, CHARIOT_DARK,
                        (rect.centerx, rect.y),
                        (rect.centerx, rect.y - 15), 3)


class DustParticle:
    """A dust particle effect."""
    def __init__(self, x, y):
        self.x = x + random.uniform(-5, 5)
        self.y = y + random.uniform(-5, 5)
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-2, -0.5)
        self.life = random.uniform(0.3, 0.8)
        self.max_life = self.life
        self.size = random.randint(2, 5)

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 0.05 * dt
        self.life -= 0.02 * dt

    def draw(self, screen):
        if self.life > 0:
            size = max(1, int(self.size * (self.life / self.max_life)))
            pygame.draw.circle(screen, DUST_COLOR, (int(self.x), int(self.y)), size)


class HighNoonGame:
    """Main game class."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("High Noon")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.running = True
        self.state = "menu"  # menu, playing, round_over

        self.cowboys = []
        self.bullets = []
        self.obstacles = []
        self.chariots = []
        self.dust_particles = []
        self.round_num = 1
        self.scores = [0, 0]
        self.message = ""
        self.message_timer = 0
        self.chariot_spawn_timer = 0
        self.chariot_spawn_interval = 120

        # Decorative cactuses for the menu (created once to avoid flickering)
        self.menu_cactuses = []
        for cx in [100, 300, 500, 700]:
            cy = WINDOW_HEIGHT - GROUND_HEIGHT
            self.menu_cactuses.append(Cactus(cx, cy - 40))

        self.setup_round()

    def setup_round(self):
        """Set up a new round."""
        self.cowboys = []
        self.bullets = []
        self.obstacles = []
        self.chariots = []
        self.dust_particles = []
        self.chariot_spawn_timer = 0

        # Create cowboys
        cowboy1 = Cowboy(80, WINDOW_HEIGHT - GROUND_HEIGHT - COWBOY_HEIGHT, True, 1)
        cowboy2 = Cowboy(WINDOW_WIDTH - 80, WINDOW_HEIGHT - GROUND_HEIGHT - COWBOY_HEIGHT, False, 2)
        self.cowboys = [cowboy1, cowboy2]

        # Generate obstacles
        self.generate_obstacles()

        self.message = f"Round {self.round_num}! Fight!"
        self.message_timer = pygame.time.get_ticks()

    def generate_obstacles(self):
        """Generate rocks and cactuses across the entire screen,
        avoiding the initial player positions."""
        self.obstacles = []

        # Safe zones around initial player positions (no obstacles here)
        safe_zones = [
            pygame.Rect(0, 0, 140, WINDOW_HEIGHT),                    # Player 1 zone
            pygame.Rect(WINDOW_WIDTH - 140, 0, 140, WINDOW_HEIGHT),   # Player 2 zone
        ]

        num_obstacles = random.randint(8, 14)
        for _ in range(num_obstacles):
            for attempt in range(30):
                x = random.randint(20, WINDOW_WIDTH - 20)
                y = random.randint(30, WINDOW_HEIGHT - GROUND_HEIGHT - 10)

                # Check safe zones (don't place on top of initial player positions)
                in_safe_zone = False
                for sz in safe_zones:
                    if sz.collidepoint(x, y):
                        in_safe_zone = True
                        break
                if in_safe_zone:
                    continue

                # Check no overlap with existing obstacles
                overlap = False
                new_rect = pygame.Rect(x - 15, y - 15, 30, 30)
                for obs in self.obstacles:
                    if new_rect.colliderect(obs.get_rect()):
                        overlap = True
                        break
                if overlap:
                    continue

                # Place obstacle
                if random.random() < 0.5:
                    self.obstacles.append(Rock(x, y))
                else:
                    self.obstacles.append(Cactus(x, y))
                break

    def handle_menu_click(self, pos):
        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 370, 200, 50)
        if btn_rect.collidepoint(pos):
            self.round_num = 1
            self.scores = [0, 0]
            self.setup_round()
            self.state = "playing"

    def check_collisions(self):
        """Check bullet collisions with cowboys, obstacles, and chariots using bounding boxes."""
        for bullet in self.bullets[:]:
            if not bullet.active:
                continue

            bullet_rect = bullet.get_rect()

            # Check collision with cowboys
            for i, cowboy in enumerate(self.cowboys):
                if not cowboy.alive:
                    continue

                if bullet_rect.colliderect(cowboy.get_hitbox()):
                    # HIT!
                    bullet.active = False
                    cowboy.alive = False
                    # Determine who shot this bullet (the other player)
                    shooter = 1 - i
                    self.scores[shooter] += 1
                    self.message = f"Player {shooter + 1} wins the round!"
                    self.message_timer = pygame.time.get_ticks()
                    self.state = "round_over"
                    return

            # Check collision with obstacles
            for obstacle in self.obstacles[:]:
                if bullet_rect.colliderect(obstacle.get_rect()):
                    bullet.active = False
                    for _ in range(8):
                        self.dust_particles.append(DustParticle(bullet.x, bullet.y))
                    break

            # Check collision with chariots
            for chariot in self.chariots[:]:
                if bullet_rect.colliderect(chariot.get_rect()):
                    bullet.active = False
                    for _ in range(8):
                        self.dust_particles.append(DustParticle(bullet.x, bullet.y))
                    break

    def update(self, dt):
        """Update game state."""
        if self.state != "playing":
            return

        # Handle continuous key input for movement and shooting
        keys = pygame.key.get_pressed()

        # Player 1 (left) - WASD + Space
        p1 = self.cowboys[0]
        if p1.alive:
            dx1, dy1 = 0, 0
            if keys[pygame.K_a]:
                dx1 = -MOVE_SPEED
            if keys[pygame.K_d]:
                dx1 = MOVE_SPEED
            if keys[pygame.K_w]:
                dy1 = -MOVE_SPEED
            if keys[pygame.K_s]:
                dy1 = MOVE_SPEED
            if dx1 != 0 or dy1 != 0:
                p1.move(dx1 * dt, dy1 * dt, self.obstacles, self.chariots)
                # Face the direction of movement
                if dx1 > 0:
                    p1.facing_right = True
                elif dx1 < 0:
                    p1.facing_right = False

            if keys[pygame.K_SPACE]:
                bullet = p1.shoot()
                if bullet:
                    self.bullets.append(bullet)
                    for _ in range(5):
                        gx, gy = p1.get_gun_position()
                        self.dust_particles.append(DustParticle(gx, gy))

        # Player 2 (right) - Arrow keys + Enter
        p2 = self.cowboys[1]
        if p2.alive:
            dx2, dy2 = 0, 0
            if keys[pygame.K_LEFT]:
                dx2 = -MOVE_SPEED
            if keys[pygame.K_RIGHT]:
                dx2 = MOVE_SPEED
            if keys[pygame.K_UP]:
                dy2 = -MOVE_SPEED
            if keys[pygame.K_DOWN]:
                dy2 = MOVE_SPEED
            if dx2 != 0 or dy2 != 0:
                p2.move(dx2 * dt, dy2 * dt, self.obstacles, self.chariots)
                # Face the direction of movement
                if dx2 > 0:
                    p2.facing_right = True
                elif dx2 < 0:
                    p2.facing_right = False

            if keys[pygame.K_RETURN] or keys[pygame.K_KP_ENTER]:
                bullet = p2.shoot()
                if bullet:
                    self.bullets.append(bullet)
                    for _ in range(5):
                        gx, gy = p2.get_gun_position()
                        self.dust_particles.append(DustParticle(gx, gy))

        # Update cowboys cooldowns
        for cowboy in self.cowboys:
            if cowboy.shoot_cooldown > 0:
                cowboy.shoot_cooldown -= 1

        # Update bullets
        for bullet in self.bullets[:]:
            bullet.update(dt)
            if not bullet.active:
                self.bullets.remove(bullet)

        # Check collisions
        self.check_collisions()

        # Update chariots
        self.chariot_spawn_timer += 1
        if self.chariot_spawn_timer >= self.chariot_spawn_interval:
            self.chariot_spawn_timer = 0
            if len(self.chariots) < 4:
                self.chariots.append(Chariot())

        for chariot in self.chariots[:]:
            chariot.update(dt)
            if not chariot.active:
                self.chariots.remove(chariot)
                continue

            # Check if chariot hits a cowboy
            chariot_rect = chariot.get_rect()
            for i, cowboy in enumerate(self.cowboys):
                if not cowboy.alive:
                    continue
                if chariot_rect.colliderect(cowboy.get_hitbox()):
                    # Chariot hit! Cowboy dies, other player wins
                    cowboy.alive = False
                    chariot.active = False
                    winner = 1 - i
                    self.scores[winner] += 1
                    self.message = f"Player {winner + 1} wins the round!"
                    self.message_timer = pygame.time.get_ticks()
                    self.state = "round_over"
                    break

        # Update dust particles
        self.dust_particles = [p for p in self.dust_particles if p.life > 0]
        for particle in self.dust_particles:
            particle.update(dt)

    def draw_menu(self):
        """Draw the main menu."""
        for y in range(WINDOW_HEIGHT):
            color_ratio = y / WINDOW_HEIGHT
            r = int(135 + color_ratio * 45)
            g = int(180 + color_ratio * 20)
            b = int(220 - color_ratio * 100)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (WINDOW_WIDTH, y))

        pygame.draw.circle(self.screen, SUN_COLOR, (WINDOW_WIDTH // 2, 80), 40)
        pygame.draw.circle(self.screen, (255, 240, 150), (WINDOW_WIDTH // 2, 80), 30)

        ground_rect = pygame.Rect(0, WINDOW_HEIGHT - GROUND_HEIGHT, WINDOW_WIDTH, GROUND_HEIGHT)
        pygame.draw.rect(self.screen, GROUND_COLOR, ground_rect)
        pygame.draw.rect(self.screen, GROUND_DARK, ground_rect, 2)

        for deco_cactus in self.menu_cactuses:
            deco_cactus.draw(self.screen)

        title = self.font_large.render("HIGH NOON", True, (200, 50, 50))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 160))
        shadow = self.font_large.render("HIGH NOON", True, (80, 20, 20))
        shadow_rect = shadow.get_rect(center=(WINDOW_WIDTH // 2 + 3, 163))
        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(title, title_rect)

        subtitle = self.font_medium.render("A Wild West Duel", True, WHITE)
        sub_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 210))
        self.screen.blit(subtitle, sub_rect)

        instr = [
            "Player 1 (Blue): WASD to move, Space to shoot",
            "Player 2 (Red): Arrow keys to move, Enter to shoot",
            "Rocks, cactuses, and chariots provide cover",
            "First to hit the other wins the round!"
        ]
        for i, line in enumerate(instr):
            t = self.font_small.render(line, True, (220, 220, 200))
            t_rect = t.get_rect(center=(WINDOW_WIDTH // 2, 250 + i * 25))
            self.screen.blit(t, t_rect)

        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 370, 200, 50)
        mouse_pos = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mouse_pos)
        color = (180, 60, 60) if hover else (140, 40, 40)
        pygame.draw.rect(self.screen, color, btn_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, btn_rect, 2, border_radius=10)
        text = self.font_medium.render("Play Game", True, WHITE)
        text_rect = text.get_rect(center=btn_rect.center)
        self.screen.blit(text, text_rect)

    def draw_game(self):
        """Draw the game screen."""
        self.screen.fill(SKY_COLOR)

        pygame.draw.circle(self.screen, SUN_COLOR, (WINDOW_WIDTH - 80, 60), 35)
        pygame.draw.circle(self.screen, (255, 240, 150), (WINDOW_WIDTH - 80, 60), 25)

        ground_rect = pygame.Rect(0, WINDOW_HEIGHT - GROUND_HEIGHT, WINDOW_WIDTH, GROUND_HEIGHT)
        pygame.draw.rect(self.screen, GROUND_COLOR, ground_rect)
        pygame.draw.rect(self.screen, GROUND_DARK, ground_rect, 2)

        for gx in range(0, WINDOW_WIDTH, 30):
            pygame.draw.line(self.screen, GROUND_DARK,
                           (gx, WINDOW_HEIGHT - GROUND_HEIGHT),
                           (gx, WINDOW_HEIGHT), 1)

        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        for chariot in self.chariots:
            chariot.draw(self.screen)

        for cowboy in self.cowboys:
            cowboy.draw(self.screen)

        for bullet in self.bullets:
            bullet.draw(self.screen)

        for particle in self.dust_particles:
            particle.draw(self.screen)

        self.draw_ui()

        if self.message and pygame.time.get_ticks() - self.message_timer < 2000:
            msg = self.font_medium.render(self.message, True, WHITE)
            msg_rect = msg.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100))
            bg_rect = msg_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect, border_radius=5)
            self.screen.blit(msg, msg_rect)

    def draw_ui(self):
        """Draw the game UI overlay."""
        pygame.draw.rect(self.screen, (0, 0, 0, 160), (0, 0, WINDOW_WIDTH, 50))
        pygame.draw.rect(self.screen, WHITE, (0, 0, WINDOW_WIDTH, 50), 2)

        round_text = self.font_medium.render(f"Round {self.round_num}", True, WHITE)
        round_rect = round_text.get_rect(center=(WINDOW_WIDTH // 2, 12))
        self.screen.blit(round_text, round_rect)

        score_text = self.font_small.render(
            f"P1: {self.scores[0]}  |  P2: {self.scores[1]}", True, WHITE)
        score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 35))
        self.screen.blit(score_text, score_rect)

        p1_label = self.font_small.render("Player 1 (WASD+Space)", True, COWBOY_BODY_P1)
        p1_rect = p1_label.get_rect(topleft=(10, 5))
        self.screen.blit(p1_label, p1_rect)

        p2_label = self.font_small.render("Player 2 (Arrows+Enter)", True, COWBOY_BODY_P2)
        p2_rect = p2_label.get_rect(topright=(WINDOW_WIDTH - 10, 5))
        self.screen.blit(p2_label, p2_rect)

    def draw_round_over(self):
        """Draw the round over screen."""
        self.draw_game()

        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(160)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        winner = None
        for i, cowboy in enumerate(self.cowboys):
            if not cowboy.alive:
                winner = 1 - i
                break

        if winner is not None:
            result_text = f"Player {winner + 1} Wins the Round!"
        else:
            result_text = "Draw!"

        congrats = self.font_large.render(result_text, True, (255, 215, 0))
        congrats_rect = congrats.get_rect(center=(WINDOW_WIDTH // 2, 200))
        self.screen.blit(congrats, congrats_rect)

        score_text = self.font_medium.render(
            f"Score - Player 1: {self.scores[0]}  |  Player 2: {self.scores[1]}", True, WHITE)
        score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 260))
        self.screen.blit(score_text, score_rect)

        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 320, 200, 50)
        mouse_pos = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mouse_pos)
        color = (100, 200, 100) if hover else (60, 150, 60)
        pygame.draw.rect(self.screen, color, btn_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, btn_rect, 2, border_radius=10)
        next_text = self.font_medium.render("Next Round", True, WHITE)
        next_rect = next_text.get_rect(center=btn_rect.center)
        self.screen.blit(next_text, next_rect)

        menu_btn = pygame.Rect(WINDOW_WIDTH // 2 - 100, 390, 200, 50)
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
                        self.handle_menu_click(event.pos)
                    elif self.state == "round_over":
                        btn_rect, menu_btn = self.draw_round_over()
                        if btn_rect.collidepoint(event.pos):
                            self.round_num += 1
                            self.setup_round()
                            self.state = "playing"
                        elif menu_btn.collidepoint(event.pos):
                            self.state = "menu"

            # Update game state (handles continuous key input)
            self.update(dt)

            # Draw based on state
            if self.state == "menu":
                self.draw_menu()
            elif self.state == "playing":
                self.draw_game()
            elif self.state == "round_over":
                self.draw_round_over()

            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = HighNoonGame()
    game.run()
