"""
Joust - Classic 1982 arcade game implemented in Pygame.

Ride your ostrich and joust against enemy knights on buzzards!
Flap to stay airborne, collide with enemies from above to defeat them.

Controls:
- Left/Right arrows: Move
- Up arrow / Space / Z: Flap (fly upward)
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
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
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
SKY_BLUE = (50, 100, 200)
LAVA_COLOR = (200, 80, 20)
PLATFORM_COLOR = (80, 60, 40)

# Physics
GRAVITY = 0.3
FLAP_SPEED = -5.0
MAX_FALL_SPEED = 8.0
PLAYER_SPEED = 3.0
ENEMY_SPEED = 1.5

# Player
PLAYER_WIDTH = 24
PLAYER_HEIGHT = 28

# Enemy
ENEMY_WIDTH = 24
ENEMY_HEIGHT = 28

# Platforms
PLATFORM_HEIGHT = 12

# Scoring
SCORE_KNIGHT = 500
SCORE_BOUNTY = 1000
SCORE_EGG = 250

# Fonts
font_small = pygame.font.Font(None, 20)
font_medium = pygame.font.Font(None, 32)
font_large = pygame.font.Font(None, 48)
font_huge = pygame.font.Font(None, 64)

# Game states
MENU = 0
PLAYING = 1
GAME_OVER = 2
WAVE_CLEAR = 3

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


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


class Platform:
    """A floating platform in the jousting arena."""
    def __init__(self, x, y, width):
        self.x = x
        self.y = y
        self.width = width
        self.height = PLATFORM_HEIGHT
        self.rect = pygame.Rect(x, y, width, PLATFORM_HEIGHT)

    def draw(self, screen):
        # Platform body
        pygame.draw.rect(screen, PLATFORM_COLOR, self.rect)
        # Top edge
        pygame.draw.rect(screen, (120, 100, 80), (self.x, self.y, self.width, 3))
        # Bottom edge
        pygame.draw.rect(screen, (50, 35, 20), (self.x, self.y + self.height - 3, self.width, 3))
        # Support pillars
        pillar_w = 4
        for px in [self.x + 2, self.x + self.width - 2 - pillar_w]:
            pygame.draw.rect(screen, (60, 45, 30), (px, self.y + self.height, pillar_w, 8))


class Player:
    """The player's knight riding an ostrich."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.facing = 1  # 1 = right, -1 = left
        self.alive = True
        self.flap_timer = 0
        self.wing_frame = 0
        self.anim_timer = 0
        self.on_ground = False
        self.score = 0
        self.lives = 3
        self.respawn_timer = 0
        self.invincible_timer = 0

    def update(self, keys, platforms):
        if not self.alive:
            self.respawn_timer -= 1
            if self.respawn_timer <= 0 and self.lives > 0:
                self.alive = True
                self.x = SCREEN_WIDTH // 2
                self.y = 100
                self.vx = 0
                self.vy = 0
                self.invincible_timer = 90
            return

        if self.invincible_timer > 0:
            self.invincible_timer -= 1

        # Movement
        self.vx = 0
        if keys[pygame.K_LEFT]:
            self.vx = -PLAYER_SPEED
            self.facing = -1
        elif keys[pygame.K_RIGHT]:
            self.vx = PLAYER_SPEED
            self.facing = 1

        # Flapping
        if keys[pygame.K_UP] or keys[pygame.K_SPACE] or keys[pygame.K_z]:
            self.vy = FLAP_SPEED
            self.flap_timer = 8
            self.wing_frame = 1
        else:
            if self.flap_timer > 0:
                self.flap_timer -= 1
            else:
                self.wing_frame = 0

        # Gravity
        self.vy += GRAVITY
        if self.vy > MAX_FALL_SPEED:
            self.vy = MAX_FALL_SPEED

        # Move horizontally
        self.x += self.vx
        self.x = clamp(self.x, self.width // 2, SCREEN_WIDTH - self.width // 2)

        # Move vertically
        self.y += self.vy

        # Platform collision
        self.on_ground = False
        player_rect = self.get_rect()
        for plat in platforms:
            if self.vy >= 0:  # Falling
                # Check if landing on platform
                foot_rect = pygame.Rect(self.x - self.width // 2 + 2,
                                        self.y + self.height // 2 - 4,
                                        self.width - 4, 8)
                if foot_rect.colliderect(plat.rect):
                    self.y = plat.y - self.height // 2
                    self.vy = 0
                    self.on_ground = True
                    break

        # Screen top/bottom bounds
        if self.y < self.height // 2:
            self.y = self.height // 2
            self.vy = 0
        if self.y > SCREEN_HEIGHT + 50:
            self.die()

        # Animation
        self.anim_timer += 1
        if self.anim_timer > 10:
            self.anim_timer = 0

    def die(self):
        if self.invincible_timer > 0:
            return
        self.alive = False
        self.lives -= 1
        self.respawn_timer = 60

    def get_rect(self):
        return pygame.Rect(self.x - self.width // 2, self.y - self.height // 2,
                           self.width, self.height)

    def draw(self, screen):
        if not self.alive:
            return

        # Flash when invincible
        if self.invincible_timer > 0 and (self.invincible_timer // 4) % 2 == 0:
            return

        cx, cy = self.x, self.y
        f = self.facing

        # --- Ostrich body ---
        # Body
        body_color = (180, 140, 100)
        pygame.draw.ellipse(screen, body_color,
                            (cx - 10, cy + 2, 20, 14))

        # Neck
        neck_color = (160, 200, 120)
        pygame.draw.ellipse(screen, neck_color,
                            (cx + f * 4, cy - 10, 8, 16))

        # Head
        head_color = (160, 200, 120)
        pygame.draw.circle(screen, head_color, (cx + f * 8, cy - 12), 6)
        # Beak
        beak_color = (255, 200, 0)
        pygame.draw.polygon(screen, beak_color, [
            (cx + f * 13, cy - 12),
            (cx + f * 18, cy - 10),
            (cx + f * 13, cy - 8)
        ])
        # Eye
        pygame.draw.circle(screen, BLACK, (cx + f * 9, cy - 13), 2)
        pygame.draw.circle(screen, WHITE, (cx + f * 9, cy - 13), 1)

        # Legs
        leg_color = (200, 160, 80)
        leg_offset = 3 if self.anim_timer < 5 else -3
        pygame.draw.line(screen, leg_color,
                         (cx - 4, cy + 14), (cx - 6 + leg_offset, cy + 22), 3)
        pygame.draw.line(screen, leg_color,
                         (cx + 4, cy + 14), (cx + 6 - leg_offset, cy + 22), 3)

        # Wings
        wing_color = (200, 170, 120)
        if self.wing_frame == 1:
            # Wings up (flapping)
            pygame.draw.ellipse(screen, wing_color,
                                (cx - 14, cy - 6, 12, 8))
            pygame.draw.ellipse(screen, wing_color,
                                (cx + 2, cy - 6, 12, 8))
        else:
            # Wings down (gliding)
            pygame.draw.ellipse(screen, wing_color,
                                (cx - 14, cy + 2, 12, 8))
            pygame.draw.ellipse(screen, wing_color,
                                (cx + 2, cy + 2, 12, 8))

        # --- Knight on top ---
        # Body
        knight_color = (200, 50, 50)
        pygame.draw.ellipse(screen, knight_color,
                            (cx - 6, cy - 8, 12, 14))

        # Helmet
        helmet_color = (150, 150, 150)
        pygame.draw.ellipse(screen, helmet_color,
                            (cx - 5, cy - 14, 10, 10))
        # Helmet visor
        visor_color = (100, 100, 100)
        pygame.draw.rect(screen, visor_color,
                         (cx + f * 2, cy - 12, 6, 4))

        # Lance
        lance_color = (180, 160, 120)
        lance_y = cy - 4
        if f > 0:
            pygame.draw.line(screen, lance_color,
                             (cx + 8, lance_y), (cx + 30, lance_y - 2), 3)
            # Lance tip
            pygame.draw.polygon(screen, WHITE, [
                (cx + 30, lance_y - 2),
                (cx + 36, lance_y - 1),
                (cx + 30, lance_y)
            ])
        else:
            pygame.draw.line(screen, lance_color,
                             (cx - 8, lance_y), (cx - 30, lance_y - 2), 3)
            # Lance tip
            pygame.draw.polygon(screen, WHITE, [
                (cx - 30, lance_y - 2),
                (cx - 36, lance_y - 1),
                (cx - 30, lance_y)
            ])


class Enemy:
    """Enemy knight riding a buzzard."""
    def __init__(self, x, y, etype=0):
        self.x = x
        self.y = y
        self.vx = ENEMY_SPEED * random.choice([-1, 1])
        self.vy = 0
        self.width = ENEMY_WIDTH
        self.height = ENEMY_HEIGHT
        self.type = etype  # 0 = normal, 1 = shadow lord (faster)
        self.alive = True
        self.facing = 1 if self.vx > 0 else -1
        self.wing_frame = 0
        self.anim_timer = 0
        self.flap_timer = random.randint(0, 30)
        self.popping = False
        self.pop_timer = 0
        self.score_value = SCORE_KNIGHT if etype == 0 else SCORE_BOUNTY
        self.target_y = random.randint(60, 200)

        # Color based on type
        if etype == 0:
            self.body_color = (100, 180, 100)  # Green buzzard
            self.knight_color = (50, 50, 200)  # Blue knight
        else:
            self.body_color = (80, 60, 100)  # Purple buzzard
            self.knight_color = (200, 50, 200)  # Purple knight
            self.vx *= 1.5

    def update(self, platforms, player_x, player_y):
        if not self.alive:
            if self.popping:
                self.pop_timer -= 1
                return self.pop_timer > 0
            return False

        # AI: move toward player
        dx = player_x - self.x
        dy = player_y - self.y

        # Horizontal movement
        if abs(dx) > 20:
            self.vx = ENEMY_SPEED * (1 if dx > 0 else -1)
            if self.type == 1:
                self.vx *= 1.5
        else:
            self.vx *= 0.9

        self.facing = 1 if self.vx > 0 else -1

        # Vertical movement - try to match player's height
        self.flap_timer -= 1
        if self.flap_timer <= 0:
            if dy < -30:  # Player is above
                self.vy = FLAP_SPEED * 0.8
                self.flap_timer = random.randint(10, 25)
                self.wing_frame = 1
            elif dy > 50:  # Player is below
                self.vy = FLAP_SPEED * 0.5
                self.flap_timer = random.randint(20, 40)
                self.wing_frame = 1
            else:
                self.flap_timer = random.randint(15, 35)
                self.wing_frame = 1
        else:
            if self.flap_timer < 5:
                self.wing_frame = 0

        # Gravity
        self.vy += GRAVITY * 0.7
        if self.vy > MAX_FALL_SPEED:
            self.vy = MAX_FALL_SPEED

        # Move
        self.x += self.vx
        self.y += self.vy

        # Screen bounds
        if self.x < self.width // 2:
            self.x = self.width // 2
            self.vx = abs(self.vx)
        elif self.x > SCREEN_WIDTH - self.width // 2:
            self.x = SCREEN_WIDTH - self.width // 2
            self.vx = -abs(self.vx)

        # Platform collision
        enemy_rect = self.get_rect()
        for plat in platforms:
            if self.vy >= 0:
                foot_rect = pygame.Rect(self.x - self.width // 2 + 2,
                                        self.y + self.height // 2 - 4,
                                        self.width - 4, 8)
                if foot_rect.colliderect(plat.rect):
                    self.y = plat.y - self.height // 2
                    self.vy = 0
                    break

        # Fall off screen
        if self.y > SCREEN_HEIGHT + 50:
            self.alive = False
            return False

        # Animation
        self.anim_timer += 1
        if self.anim_timer > 10:
            self.anim_timer = 0

        return True

    def die(self):
        self.alive = False
        self.popping = True
        self.pop_timer = 30

    def get_rect(self):
        return pygame.Rect(self.x - self.width // 2, self.y - self.height // 2,
                           self.width, self.height)

    def draw(self, screen):
        if not self.alive:
            if self.popping:
                pop_size = int(16 + (30 - self.pop_timer) * 0.5)
                alpha = int((self.pop_timer / 30) * 255)
                color = (255, min(255, alpha), min(255, alpha))
                pygame.draw.circle(screen, color, (int(self.x), int(self.y)), pop_size)
                pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), pop_size, 2)
            return

        cx, cy = self.x, self.y
        f = self.facing

        # --- Buzzard body ---
        body_color = self.body_color
        pygame.draw.ellipse(screen, body_color,
                            (cx - 10, cy + 2, 20, 14))

        # Neck
        neck_color = (120, 160, 100) if self.type == 0 else (100, 80, 120)
        pygame.draw.ellipse(screen, neck_color,
                            (cx + f * 4, cy - 8, 7, 14))

        # Head
        head_color = (140, 180, 120) if self.type == 0 else (120, 100, 140)
        pygame.draw.circle(screen, head_color, (cx + f * 8, cy - 10), 5)
        # Beak
        beak_color = (200, 180, 50)
        pygame.draw.polygon(screen, beak_color, [
            (cx + f * 12, cy - 10),
            (cx + f * 16, cy - 8),
            (cx + f * 12, cy - 6)
        ])
        # Eye
        pygame.draw.circle(screen, RED, (cx + f * 8, cy - 11), 2)

        # Wings
        wing_color = (140, 200, 140) if self.type == 0 else (120, 100, 140)
        if self.wing_frame == 1:
            pygame.draw.ellipse(screen, wing_color,
                                (cx - 16, cy - 4, 14, 8))
            pygame.draw.ellipse(screen, wing_color,
                                (cx + 2, cy - 4, 14, 8))
        else:
            pygame.draw.ellipse(screen, wing_color,
                                (cx - 16, cy + 4, 14, 8))
            pygame.draw.ellipse(screen, wing_color,
                                (cx + 2, cy + 4, 14, 8))

        # --- Knight on top ---
        knight_color = self.knight_color
        pygame.draw.ellipse(screen, knight_color,
                            (cx - 6, cy - 6, 12, 12))

        # Helmet
        helmet_color = (180, 180, 180)
        pygame.draw.ellipse(screen, helmet_color,
                            (cx - 5, cy - 12, 10, 10))
        # Visor
        visor_color = (120, 120, 120)
        pygame.draw.rect(screen, visor_color,
                         (cx + f * 2, cy - 10, 5, 3))

        # Lance
        lance_color = (160, 140, 100)
        lance_y = cy - 2
        if f > 0:
            pygame.draw.line(screen, lance_color,
                             (cx + 8, lance_y), (cx + 28, lance_y - 1), 3)
            pygame.draw.polygon(screen, WHITE, [
                (cx + 28, lance_y - 1),
                (cx + 34, lance_y),
                (cx + 28, lance_y + 1)
            ])
        else:
            pygame.draw.line(screen, lance_color,
                             (cx - 8, lance_y), (cx - 28, lance_y - 1), 3)
            pygame.draw.polygon(screen, WHITE, [
                (cx - 28, lance_y - 1),
                (cx - 34, lance_y),
                (cx - 28, lance_y + 1)
            ])


class Egg:
    """An egg that appears when an enemy is defeated."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vy = -2
        self.alive = True
        self.collected = False
        self.collect_timer = 0
        self.size = 8

    def update(self, platforms):
        if not self.alive:
            return False

        # Gravity
        self.vy += 0.2
        self.y += self.vy

        # Platform collision
        egg_rect = self.get_rect()
        for plat in platforms:
            if self.vy >= 0:
                if egg_rect.colliderect(plat.rect):
                    self.y = plat.y - self.size
                    self.vy = 0

        # Fall off screen
        if self.y > SCREEN_HEIGHT + 20:
            self.alive = False

        return True

    def collect(self):
        self.collected = True
        self.collect_timer = 20

    def get_rect(self):
        return pygame.Rect(self.x - self.size, self.y - self.size,
                           self.size * 2, self.size * 2)

    def draw(self, screen):
        if not self.alive:
            return

        # Egg shape
        egg_color = (255, 255, 200)
        pygame.draw.ellipse(screen, egg_color,
                            (self.x - self.size, self.y - self.size * 0.7,
                             self.size * 2, self.size * 1.4))
        pygame.draw.ellipse(screen, (200, 200, 150),
                            (self.x - self.size, self.y - self.size * 0.7,
                             self.size * 2, self.size * 1.4), 1)

        # Sparkle
        pygame.draw.circle(screen, WHITE,
                           (self.x - 2, self.y - 3), 2)


class JoustGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Joust")
        self.clock = pygame.time.Clock()
        self.running = True

        # Sound
        self.sound_enabled = True
        try:
            pygame.mixer.init()
            self.flap_sound = self.create_sound(200, 0.1)
            self.hit_sound = self.create_sound(500, 0.15)
            self.collect_sound = self.create_sound(800, 0.1)
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

    def generate_platforms(self):
        """Generate platforms for the arena."""
        platforms = []

        # Bottom lava platform (full width)
        platforms.append(Platform(0, SCREEN_HEIGHT - 40, SCREEN_WIDTH))

        # Floating platforms
        plat_configs = [
            (50, 350, 120),
            (200, 300, 100),
            (350, 350, 120),
            (500, 300, 100),
            (100, 220, 80),
            (280, 200, 100),
            (460, 220, 80),
            (50, 140, 70),
            (300, 120, 80),
            (520, 140, 70),
        ]

        for px, py, pw in plat_configs:
            platforms.append(Platform(px, py, pw))

        return platforms

    def reset_game(self):
        self.state = MENU
        self.score = 0
        self.high_score = 0
        self.wave = 1
        self.player = Player(SCREEN_WIDTH // 2, 200)
        self.platforms = self.generate_platforms()
        self.enemies = []
        self.eggs = []
        self.particles = []
        self.spawn_timer = 0
        self.wave_clear_timer = 0
        self.game_over_timer = 0
        self.wave_transition = False

    def spawn_enemy(self):
        """Spawn a new enemy."""
        if len(self.enemies) >= min(3 + self.wave, 8):
            return

        # Spawn from the sides
        side = random.choice([-1, 1])
        x = self.player.width // 2 if side < 0 else SCREEN_WIDTH - self.player.width // 2
        y = random.randint(80, 200)

        etype = 0
        if self.wave >= 3 and random.random() < 0.2:
            etype = 1  # Shadow lord

        enemy = Enemy(x, y, etype)
        self.enemies.append(enemy)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
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

    def check_joust(self):
        """Check jousting collisions between player and enemies."""
        player_rect = self.player.get_rect()

        for enemy in self.enemies[:]:
            if not enemy.alive or enemy.popping:
                continue

            enemy_rect = enemy.get_rect()

            if player_rect.colliderect(enemy_rect):
                # Determine who wins based on vertical position
                # The one higher up wins (jousting rule)
                if self.player.y < enemy.y - 8:
                    # Player wins - enemy defeated
                    enemy.die()
                    self.score += enemy.score_value
                    if self.sound_enabled:
                        self.hit_sound.play()

                    # Explosion particles
                    for _ in range(15):
                        angle = random.uniform(0, math.pi * 2)
                        speed = random.uniform(1, 4)
                        self.particles.append(Particle(
                            enemy.x, enemy.y, enemy.knight_color, speed, angle
                        ))

                    # Spawn egg
                    egg = Egg(enemy.x, enemy.y)
                    self.eggs.append(egg)

                elif self.player.y > enemy.y + 8:
                    # Enemy wins - player defeated
                    self.player.die()
                    if self.sound_enabled:
                        self.hit_sound.play()

                    # Explosion particles
                    for _ in range(15):
                        angle = random.uniform(0, math.pi * 2)
                        speed = random.uniform(1, 4)
                        self.particles.append(Particle(
                            self.player.x, self.player.y, YELLOW, speed, angle
                        ))

                    if self.player.lives <= 0:
                        self.state = GAME_OVER
                        self.game_over_timer = 120
                        if self.score > self.high_score:
                            self.high_score = self.score
                else:
                    # Equal height - both bounce back
                    self.player.vy = -3
                    enemy.vy = -3

    def check_egg_collection(self):
        """Check if player collects eggs."""
        player_rect = self.player.get_rect()

        for egg in self.eggs[:]:
            if not egg.alive or egg.collected:
                continue

            if player_rect.colliderect(egg.get_rect()):
                egg.collect()
                self.score += SCORE_EGG
                if self.sound_enabled:
                    self.collect_sound.play()

                # Sparkle particles
                for _ in range(8):
                    angle = random.uniform(0, math.pi * 2)
                    speed = random.uniform(1, 3)
                    self.particles.append(Particle(
                        egg.x, egg.y, YELLOW, speed, angle
                    ))

    def check_wave_clear(self):
        """Check if all enemies are defeated."""
        alive_enemies = [e for e in self.enemies if e.alive or e.popping]
        if not alive_enemies and len(self.enemies) > 0 and self.state == PLAYING:
            self.state = WAVE_CLEAR
            self.wave_clear_timer = 120
            self.wave_transition = True

    def update_particles(self):
        """Update all particles."""
        self.particles = [p for p in self.particles if p.update()]

    def update(self):
        """Main update loop."""
        if self.state == PLAYING:
            keys = pygame.key.get_pressed()
            self.player.update(keys, self.platforms)

            # Flap sound
            if (keys[pygame.K_UP] or keys[pygame.K_SPACE] or keys[pygame.K_z]):
                if self.player.alive and self.player.flap_timer == 8:
                    if self.sound_enabled:
                        self.flap_sound.play()

            # Spawn enemies
            self.spawn_timer += 1
            spawn_delay = max(60, 180 - self.wave * 10)
            if self.spawn_timer >= spawn_delay:
                self.spawn_timer = 0
                self.spawn_enemy()

            # Update enemies
            for enemy in self.enemies[:]:
                alive = enemy.update(self.platforms, self.player.x, self.player.y)
                if not alive and not enemy.popping:
                    self.enemies.remove(enemy)
                elif not alive and enemy.popping and enemy.pop_timer <= 0:
                    self.enemies.remove(enemy)

            # Update eggs
            for egg in self.eggs[:]:
                alive = egg.update(self.platforms)
                if not alive:
                    self.eggs.remove(egg)
                elif egg.collected:
                    egg.collect_timer -= 1
                    if egg.collect_timer <= 0:
                        self.eggs.remove(egg)

            # Check collisions
            self.check_joust()
            self.check_egg_collection()
            self.check_wave_clear()
            self.update_particles()

            # Check game over
            if self.player.lives <= 0 and not self.player.alive:
                if self.player.respawn_timer <= 0:
                    self.state = GAME_OVER
                    self.game_over_timer = 120
                    if self.score > self.high_score:
                        self.high_score = self.score

        elif self.state == WAVE_CLEAR:
            self.wave_clear_timer -= 1
            self.update_particles()

            # Update eggs during wave clear
            for egg in self.eggs[:]:
                alive = egg.update(self.platforms)
                if not alive:
                    self.eggs.remove(egg)
                elif egg.collected:
                    egg.collect_timer -= 1
                    if egg.collect_timer <= 0:
                        self.eggs.remove(egg)

            if self.wave_clear_timer <= 0:
                self.wave += 1
                self.enemies.clear()
                self.spawn_timer = 0
                self.wave_transition = False
                self.state = PLAYING

        elif self.state == GAME_OVER:
            self.game_over_timer -= 1
            self.update_particles()

    def draw_background(self):
        """Draw the background."""
        # Sky gradient
        for y in range(SCREEN_HEIGHT):
            t = y / SCREEN_HEIGHT
            r = int(50 + t * 50)
            g = int(100 + t * 80)
            b = int(200 - t * 100)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))

        # Lava at bottom
        lava_rect = pygame.Rect(0, SCREEN_HEIGHT - 40, SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, LAVA_COLOR, lava_rect)

        # Lava glow - use pre-computed values to avoid per-frame random flickering
        if not hasattr(self, 'lava_glow_data'):
            self.lava_glow_data = []
            for x in range(0, SCREEN_WIDTH, 4):
                h = random.randint(5, 15)
                r = random.randint(100, 200)
                g = random.randint(20, 60)
                self.lava_glow_data.append((h, r, g))
        for i, x in enumerate(range(0, SCREEN_WIDTH, 4)):
            h, r, g = self.lava_glow_data[i]
            pygame.draw.line(self.screen, (255, r, g),
                             (x, SCREEN_HEIGHT - 40 - h),
                             (x, SCREEN_HEIGHT - 40), 2)

    def draw_hud(self):
        """Draw the HUD."""
        # Score
        score_text = font_medium.render(f"{self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # High score
        high_text = font_small.render(f"HIGH: {self.high_score}", True, WHITE)
        self.screen.blit(high_text, (SCREEN_WIDTH // 2 - high_text.get_width() // 2, 10))

        # Wave
        wave_text = font_small.render(f"WAVE {self.wave}", True, WHITE)
        self.screen.blit(wave_text, (SCREEN_WIDTH - 100, 10))

        # Lives
        for i in range(self.player.lives):
            life_x = 20 + i * 20
            life_y = SCREEN_HEIGHT - 20
            # Small knight icon
            pygame.draw.circle(self.screen, (200, 50, 50), (life_x + 6, life_y), 5)
            pygame.draw.ellipse(self.screen, (150, 150, 150),
                                (life_x + 2, life_y - 6, 8, 6))

    def draw(self):
        """Main draw loop."""
        self.draw_background()

        if self.state == MENU:
            # Title
            title = font_huge.render("JOUST", True, YELLOW)
            self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 80))

            subtitle = font_medium.render("~ 1982 WILLIAMS ~", True, CYAN)
            self.screen.blit(subtitle, (SCREEN_WIDTH // 2 - subtitle.get_width() // 2, 140))

            # Draw player example
            px = SCREEN_WIDTH // 2 - 60
            py = 220
            # Ostrich
            pygame.draw.ellipse(self.screen, (180, 140, 100), (px - 10, py + 2, 20, 14))
            pygame.draw.ellipse(self.screen, (160, 200, 120), (px + 4, py - 10, 8, 16))
            pygame.draw.circle(self.screen, (160, 200, 120), (px + 8, py - 12), 6)
            pygame.draw.polygon(self.screen, (255, 200, 0), [
                (px + 13, py - 12), (px + 18, py - 10), (px + 13, py - 8)
            ])
            # Knight
            pygame.draw.ellipse(self.screen, (200, 50, 50), (px - 6, py - 8, 12, 14))
            pygame.draw.ellipse(self.screen, (150, 150, 150), (px - 5, py - 14, 10, 10))
            # Lance
            pygame.draw.line(self.screen, (180, 160, 120), (px + 8, py - 4), (px + 30, py - 6), 3)

            player_label = font_small.render("YOU", True, WHITE)
            self.screen.blit(player_label, (px - player_label.get_width() // 2, py + 30))

            # Draw enemy example
            ex = SCREEN_WIDTH // 2 + 60
            ey = 220
            # Buzzard
            pygame.draw.ellipse(self.screen, (100, 180, 100), (ex - 10, ey + 2, 20, 14))
            pygame.draw.ellipse(self.screen, (120, 160, 100), (ex - 4, ey - 8, 7, 14))
            pygame.draw.circle(self.screen, (140, 180, 120), (ex - 8, ey - 10), 5)
            pygame.draw.polygon(self.screen, (200, 180, 50), [
                (ex - 12, ey - 10), (ex - 16, ey - 8), (ex - 12, ey - 6)
            ])
            # Knight
            pygame.draw.ellipse(self.screen, (50, 50, 200), (ex - 6, ey - 6, 12, 12))
            pygame.draw.ellipse(self.screen, (180, 180, 180), (ex - 5, ey - 12, 10, 10))
            # Lance
            pygame.draw.line(self.screen, (160, 140, 100), (ex - 8, ey - 2), (ex - 28, ey - 3), 3)

            enemy_label = font_small.render("ENEMY", True, RED)
            self.screen.blit(enemy_label, (ex - enemy_label.get_width() // 2, ey + 30))

            # Instructions
            start = font_medium.render("Press SPACE to Start", True, WHITE)
            self.screen.blit(start, (SCREEN_WIDTH // 2 - start.get_width() // 2, 310))

            controls = [
                "Left/Right - Move",
                "Up / Space / Z - Flap",
                "P - Pause",
                "",
                "Joust from ABOVE to defeat enemies!",
            ]
            for i, text in enumerate(controls):
                ctrl = font_small.render(text, True, GRAY)
                self.screen.blit(ctrl, (SCREEN_WIDTH // 2 - ctrl.get_width() // 2, 360 + i * 22))

        elif self.state == PLAYING or self.state == WAVE_CLEAR:
            # Draw platforms
            for plat in self.platforms:
                plat.draw(self.screen)

            # Draw eggs
            for egg in self.eggs:
                egg.draw(self.screen)

            # Draw enemies
            for enemy in self.enemies:
                enemy.draw(self.screen)

            # Draw player
            self.player.draw(self.screen)

            # Draw particles
            for particle in self.particles:
                particle.draw(self.screen)

            self.draw_hud()

            if self.state == WAVE_CLEAR:
                clear_text = font_large.render("WAVE CLEAR!", True, GREEN)
                self.screen.blit(clear_text,
                                 (SCREEN_WIDTH // 2 - clear_text.get_width() // 2,
                                  SCREEN_HEIGHT // 2 - 50))

        elif self.state == GAME_OVER:
            # Draw final state
            for plat in self.platforms:
                plat.draw(self.screen)
            for egg in self.eggs:
                egg.draw(self.screen)
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
    game = JoustGame()
    game.run()
