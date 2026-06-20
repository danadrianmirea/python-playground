"""
Galaga - Classic 1981 arcade game implemented in Pygame.

Defend your ship from waves of Galaga fighters!
Enemies fly in formation and dive-bomb your ship.
Shoot them down, rescue captured ships, and survive!

Controls:
- Arrow keys / A,D: Move left/right
- Space / Z: Shoot
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
DARK_BLUE = (0, 0, 20)
DARK_GREEN = (0, 80, 0)

# Player
PLAYER_WIDTH = 28
PLAYER_HEIGHT = 28
PLAYER_SPEED = 3
PLAYER_Y = SCREEN_HEIGHT - 48

# Bullets
PLAYER_BULLET_SPEED = -8
ENEMY_BULLET_SPEED = 4
BULLET_WIDTH = 3
BULLET_HEIGHT = 8
PLAYER_FIRE_COOLDOWN = 12  # frames

# Formation
FORMATION_COLS = 8
FORMATION_ROWS = 5
ENEMY_WIDTH = 24
ENEMY_HEIGHT = 24
FORMATION_SPACING_X = 32
FORMATION_SPACING_Y = 28
FORMATION_TOP = 60
FORMATION_CENTER_X = SCREEN_WIDTH // 2
FORMATION_AMPLITUDE = 40  # side-to-side sway
FORMATION_SPEED = 0.5

# Enemy types
BOSS_TYPE = 0  # Top row - larger, can capture
FIGHTER_TYPE = 1  # Middle rows
SCOUT_TYPE = 2  # Bottom rows

# Dive bomber
DIVE_SPEED = 3
DIVE_RECOVER_Y = 100  # Y to recover after dive

# Tractor beam
TRACTOR_BEAM_SPEED = 1.5
TRACTOR_BEAM_LENGTH = 200

# Scoring
SCORE_BOSS = 150
SCORE_FIGHTER = 100
SCORE_SCOUT = 80
SCORE_BONUS_SHIP = 1000

# Fonts
font_small = pygame.font.Font(None, 20)
font_medium = pygame.font.Font(None, 32)
font_large = pygame.font.Font(None, 48)
font_huge = pygame.font.Font(None, 64)

# ---------------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------------
MENU = 0
PLAYING = 1
GAME_OVER = 2
STAGE_CLEAR = 3
BONUS_STAGE = 4
CAPTURED = 5  # Player is being tractored

# ---------------------------------------------------------------------------
# Enemy formation patterns (relative positions in grid)
# ---------------------------------------------------------------------------
FORMATION_PATTERNS = [
    # Pattern 0: Standard diamond
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
     (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
     (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3),
     (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4)],
    # Pattern 1: Arrowhead
    [(3, 0), (4, 0),
     (2, 1), (3, 1), (4, 1), (5, 1),
     (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
     (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3),
     (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4)],
    # Pattern 2: V formation
    [(0, 0), (7, 0),
     (0, 1), (1, 1), (6, 1), (7, 1),
     (0, 2), (1, 2), (2, 2), (5, 2), (6, 2), (7, 2),
     (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3),
     (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4)],
    # Pattern 3: Full grid
    [(c, r) for r in range(5) for c in range(8)],
]


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
        self.size = random.randint(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # gravity
        self.life -= 1
        return self.life > 0

    def draw(self, screen):
        alpha = self.life / self.max_life
        size = max(1, int(self.size * alpha))
        color = tuple(int(c * alpha) for c in self.color)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), size)


class Star:
    """Background star."""
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)
        self.speed = random.uniform(0.2, 1.0)
        self.brightness = random.randint(50, 200)

    def update(self):
        self.y += self.speed
        if self.y > SCREEN_HEIGHT:
            self.y = 0
            self.x = random.randint(0, SCREEN_WIDTH)

    def draw(self, screen):
        pygame.draw.circle(screen, (self.brightness, self.brightness, self.brightness),
                           (int(self.x), int(self.y)), 1)


class Player:
    def __init__(self):
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.x = SCREEN_WIDTH // 2 - self.width // 2
        self.y = PLAYER_Y
        self.speed = PLAYER_SPEED
        self.fire_cooldown = 0
        self.alive = True
        self.dual = False  # Dual ship power-up
        self.animation_frame = 0
        self.animation_timer = 0

    def update(self, keys):
        if not self.alive:
            return

        # Movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.x -= self.speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.x += self.speed

        # Clamp to screen
        self.x = max(0, min(self.x, SCREEN_WIDTH - self.width))

        # Cooldown
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

        # Animation
        self.animation_timer += 1
        if self.animation_timer > 10:
            self.animation_timer = 0
            self.animation_frame = (self.animation_frame + 1) % 2

    def shoot(self):
        if self.fire_cooldown == 0 and self.alive:
            self.fire_cooldown = PLAYER_FIRE_COOLDOWN
            bullets = []
            # Main bullet
            bullets.append(Bullet(
                self.x + self.width // 2 - BULLET_WIDTH // 2,
                self.y - BULLET_HEIGHT,
                PLAYER_BULLET_SPEED, True
            ))
            # Dual ship fires two
            if self.dual:
                bullets.append(Bullet(
                    self.x + self.width // 2 - BULLET_WIDTH // 2 - 10,
                    self.y - BULLET_HEIGHT,
                    PLAYER_BULLET_SPEED, True
                ))
                bullets.append(Bullet(
                    self.x + self.width // 2 - BULLET_WIDTH // 2 + 10,
                    self.y - BULLET_HEIGHT,
                    PLAYER_BULLET_SPEED, True
                ))
            return bullets
        return []

    def draw(self, screen):
        if not self.alive:
            return

        cx = self.x + self.width // 2
        cy = self.y + self.height // 2

        if self.dual:
            # Draw two ships side by side
            for offset in [-12, 12]:
                ox = self.x + offset
                # Ship body
                points = [
                    (ox + self.width // 2, self.y),  # nose
                    (ox + self.width, self.y + self.height // 2 + 4),
                    (ox + self.width - 4, self.y + self.height),
                    (ox + 4, self.y + self.height),
                    (ox, self.y + self.height // 2 + 4),
                ]
                pygame.draw.polygon(screen, GREEN, points)
                pygame.draw.polygon(screen, (0, 150, 0), points, 2)
                # Cockpit
                pygame.draw.circle(screen, CYAN,
                                   (ox + self.width // 2, self.y + 8), 3)
        else:
            # Single ship - classic Galaga style
            # Main body
            points = [
                (cx, self.y),  # nose
                (cx + 12, self.y + 10),
                (cx + 14, self.y + 18),
                (cx + 10, self.y + self.height),
                (cx - 10, self.y + self.height),
                (cx - 14, self.y + 18),
                (cx - 12, self.y + 10),
            ]
            pygame.draw.polygon(screen, GREEN, points)
            pygame.draw.polygon(screen, (0, 150, 0), points, 2)

            # Wings
            pygame.draw.polygon(screen, GREEN, [
                (cx - 14, self.y + 18),
                (cx - 20, self.y + 22),
                (cx - 16, self.y + self.height - 2),
                (cx - 10, self.y + self.height),
            ])
            pygame.draw.polygon(screen, GREEN, [
                (cx + 14, self.y + 18),
                (cx + 20, self.y + 22),
                (cx + 16, self.y + self.height - 2),
                (cx + 10, self.y + self.height),
            ])

            # Cockpit
            pygame.draw.circle(screen, CYAN, (cx, self.y + 8), 4)
            pygame.draw.circle(screen, WHITE, (cx, self.y + 8), 2)

            # Engine glow
            glow_size = 3 if self.animation_frame == 0 else 5
            pygame.draw.circle(screen, YELLOW, (cx - 6, self.y + self.height + 2), glow_size)
            pygame.draw.circle(screen, YELLOW, (cx + 6, self.y + self.height + 2), glow_size)

    def get_rect(self):
        if self.dual:
            return pygame.Rect(self.x - 12, self.y, self.width + 24, self.height)
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def reset(self):
        self.x = SCREEN_WIDTH // 2 - self.width // 2
        self.y = PLAYER_Y
        self.alive = True
        self.fire_cooldown = 0
        self.dual = False


class Bullet:
    def __init__(self, x, y, speed, is_player):
        self.x = x
        self.y = y
        self.speed = speed
        self.is_player = is_player
        self.width = BULLET_WIDTH
        self.height = BULLET_HEIGHT
        self.active = True

    def update(self):
        self.y += self.speed
        if self.y < -self.height or self.y > SCREEN_HEIGHT:
            self.active = False

    def draw(self, screen):
        if self.is_player:
            # Bright white/yellow bullet
            pygame.draw.rect(screen, WHITE,
                             (self.x, self.y, self.width, self.height))
            pygame.draw.rect(screen, YELLOW,
                             (self.x + 1, self.y + 2, self.width - 2, self.height - 4))
        else:
            # Enemy bullet - red with glow
            pygame.draw.rect(screen, RED,
                             (self.x, self.y, self.width, self.height))
            pygame.draw.circle(screen, (255, 100, 100),
                               (self.x + self.width // 2, self.y + self.height // 2),
                               self.width + 2)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Enemy:
    def __init__(self, grid_x, grid_y, enemy_type):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.type = enemy_type
        self.width = ENEMY_WIDTH
        self.height = ENEMY_HEIGHT
        self.x = 0
        self.y = 0
        self.target_x = 0
        self.target_y = 0
        self.alive = True
        self.entering = True  # Flying into formation
        self.entry_timer = grid_x * 4 + grid_y * 6  # Staggered entry
        self.entry_speed = 2
        self.entry_x = SCREEN_WIDTH // 2
        self.entry_y = -50 - grid_y * 30

        # Dive bombing
        self.diving = False
        self.dive_timer = 0
        self.dive_angle = 0
        self.dive_speed = DIVE_SPEED
        self.dive_target_x = 0
        self.recovering = False

        # Animation
        self.animation_frame = 0
        self.animation_timer = 0
        self.wing_angle = 0

        # Tractor beam
        self.tractoring = False
        self.tractor_target = None

        # Set type-specific properties
        if self.type == BOSS_TYPE:
            self.width = 28
            self.height = 28
            self.score_value = SCORE_BOSS
            self.color = RED
            self.secondary = ORANGE
        elif self.type == FIGHTER_TYPE:
            self.score_value = SCORE_FIGHTER
            self.color = MAGENTA
            self.secondary = PURPLE
        else:  # SCOUT_TYPE
            self.score_value = SCORE_SCOUT
            self.color = CYAN
            self.secondary = BLUE

    def set_formation_position(self, center_x, center_y, col, row, total_cols, total_rows):
        """Set the target position in formation."""
        formation_width = (total_cols - 1) * FORMATION_SPACING_X
        self.target_x = center_x - formation_width // 2 + col * FORMATION_SPACING_X
        self.target_y = FORMATION_TOP + row * FORMATION_SPACING_Y

    def start_dive(self, player_x):
        """Start a dive bombing run."""
        self.diving = True
        self.dive_timer = 0
        self.dive_target_x = player_x
        self.dive_angle = math.atan2(
            SCREEN_HEIGHT - self.y,
            player_x - self.x
        )

    def update(self, formation_offset_x):
        if not self.alive:
            return

        # Animation
        self.animation_timer += 1
        if self.animation_timer > 8:
            self.animation_timer = 0
            self.animation_frame = (self.animation_frame + 1) % 2
            self.wing_angle = math.sin(self.animation_timer * 0.3) * 0.2

        if self.entering:
            # Entry animation - fly in from top
            self.entry_timer -= 1
            if self.entry_timer <= 0:
                # Move toward formation position
                dx = self.target_x - self.entry_x
                dy = self.target_y - self.entry_y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < self.entry_speed:
                    self.entering = False
                    self.x = self.target_x
                    self.y = self.target_y
                else:
                    self.entry_x += (dx / dist) * self.entry_speed
                    self.entry_y += (dy / dist) * self.entry_speed
                    self.x = self.entry_x
                    self.y = self.entry_y
            else:
                # Fly down from top
                self.entry_y += self.entry_speed * 0.5
                self.entry_x += math.sin(self.entry_timer * 0.1) * 0.5
                self.x = self.entry_x
                self.y = self.entry_y
        elif self.diving:
            self.update_dive()
        elif self.recovering:
            self.update_recovery()
        else:
            # In formation - sway side to side
            self.x = self.target_x + formation_offset_x
            self.y = self.target_y + math.sin(self.animation_timer * 0.05) * 2

    def update_dive(self):
        """Update dive bombing movement."""
        self.dive_timer += 1

        if self.dive_timer < 30:
            # Initial swoop
            self.x += math.sin(self.dive_timer * 0.1) * 1.5
            self.y += self.dive_speed * 0.5
        elif self.dive_timer < 60:
            # Steep dive
            dx = self.dive_target_x - self.x
            dy = SCREEN_HEIGHT + 50 - self.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                self.x += (dx / dist) * self.dive_speed * 1.2
                self.y += (dy / dist) * self.dive_speed * 1.2
        else:
            # Continue downward
            self.y += self.dive_speed

        # Check if off screen
        if self.y > SCREEN_HEIGHT + 50 or self.x < -50 or self.x > SCREEN_WIDTH + 50:
            self.diving = False
            self.recovering = True
            self.recovery_timer = 0

    def update_recovery(self):
        """Fly back to formation."""
        self.recovery_timer += 1

        if self.recovery_timer < 20:
            # Continue downward briefly
            self.y += self.dive_speed * 0.5
        else:
            # Fly back up to formation
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 3:
                self.recovering = False
                self.x = self.target_x
                self.y = self.target_y
            else:
                speed = min(self.dive_speed * 1.5, dist)
                self.x += (dx / dist) * speed
                self.y += (dy / dist) * speed

    def draw(self, screen):
        if not self.alive:
            return

        cx = self.x + self.width // 2
        cy = self.y + self.height // 2

        if self.type == BOSS_TYPE:
            # Boss - larger, more detailed
            # Body
            pygame.draw.ellipse(screen, self.color,
                                (self.x + 2, self.y + 4, self.width - 4, self.height - 8))
            # Wings
            wing_offset = 3 if self.animation_frame == 0 else -3
            pygame.draw.polygon(screen, self.secondary, [
                (self.x + 2, cy),
                (self.x - 6, cy - 6 + wing_offset),
                (self.x - 6, cy + 6 - wing_offset),
            ])
            pygame.draw.polygon(screen, self.secondary, [
                (self.x + self.width - 2, cy),
                (self.x + self.width + 6, cy - 6 + wing_offset),
                (self.x + self.width + 6, cy + 6 - wing_offset),
            ])
            # Eyes
            pygame.draw.circle(screen, WHITE, (cx - 5, cy - 3), 4)
            pygame.draw.circle(screen, WHITE, (cx + 5, cy - 3), 4)
            pygame.draw.circle(screen, BLACK, (cx - 5, cy - 3), 2)
            pygame.draw.circle(screen, BLACK, (cx + 5, cy - 3), 2)
            # Antennae
            pygame.draw.line(screen, self.secondary, (cx - 4, self.y + 4),
                             (cx - 8, self.y - 4), 2)
            pygame.draw.line(screen, self.secondary, (cx + 4, self.y + 4),
                             (cx + 8, self.y - 4), 2)

        elif self.type == FIGHTER_TYPE:
            # Fighter - standard Galaga ship
            # Body
            pygame.draw.ellipse(screen, self.color,
                                (self.x + 3, self.y + 3, self.width - 6, self.height - 6))
            # Wings
            wing_offset = 2 if self.animation_frame == 0 else -2
            pygame.draw.polygon(screen, self.secondary, [
                (self.x + 3, cy),
                (self.x - 4, cy - 5 + wing_offset),
                (self.x - 4, cy + 5 - wing_offset),
            ])
            pygame.draw.polygon(screen, self.secondary, [
                (self.x + self.width - 3, cy),
                (self.x + self.width + 4, cy - 5 + wing_offset),
                (self.x + self.width + 4, cy + 5 - wing_offset),
            ])
            # Eyes
            pygame.draw.circle(screen, WHITE, (cx - 4, cy - 2), 3)
            pygame.draw.circle(screen, WHITE, (cx + 4, cy - 2), 3)
            pygame.draw.circle(screen, BLACK, (cx - 4, cy - 2), 2)
            pygame.draw.circle(screen, BLACK, (cx + 4, cy - 2), 2)

        else:  # SCOUT_TYPE
            # Scout - smaller, simpler
            pygame.draw.ellipse(screen, self.color,
                                (self.x + 4, self.y + 4, self.width - 8, self.height - 8))
            # Small wings
            wing_offset = 2 if self.animation_frame == 0 else -2
            pygame.draw.polygon(screen, self.secondary, [
                (self.x + 4, cy),
                (self.x - 2, cy - 4 + wing_offset),
                (self.x - 2, cy + 4 - wing_offset),
            ])
            pygame.draw.polygon(screen, self.secondary, [
                (self.x + self.width - 4, cy),
                (self.x + self.width + 2, cy - 4 + wing_offset),
                (self.x + self.width + 2, cy + 4 - wing_offset),
            ])
            # Single eye
            pygame.draw.circle(screen, WHITE, (cx, cy - 1), 3)
            pygame.draw.circle(screen, BLACK, (cx, cy - 1), 2)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class TractorBeam:
    """Tractor beam effect for capturing the player."""
    def __init__(self, x, y, target_y):
        self.x = x
        self.y = y
        self.target_y = target_y
        self.active = True
        self.lines = []
        self.phase = 0

    def update(self):
        if not self.active:
            return
        self.phase += 0.1
        # Generate beam lines
        self.lines = []
        for i in range(10):
            t = i / 10.0
            by = self.y + (self.target_y - self.y) * t
            bx = self.x + math.sin(self.phase + t * 4) * (4 + t * 6)
            self.lines.append((bx, by))

    def draw(self, screen):
        if not self.active or len(self.lines) < 2:
            return
        # Draw beam as connected lines
        for i in range(len(self.lines) - 1):
            alpha = 255 - int((i / len(self.lines)) * 200)
            color = (alpha, alpha, 255)
            pygame.draw.line(screen, color, self.lines[i], self.lines[i + 1], 2)
            # Side glow
            pygame.draw.line(screen, (0, 0, alpha // 2),
                             (self.lines[i][0] - 2, self.lines[i][1]),
                             (self.lines[i + 1][0] - 2, self.lines[i + 1][1]), 1)
            pygame.draw.line(screen, (0, 0, alpha // 2),
                             (self.lines[i][0] + 2, self.lines[i][1]),
                             (self.lines[i + 1][0] + 2, self.lines[i + 1][1]), 1)


class Galaga:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Galaga")
        self.clock = pygame.time.Clock()
        self.running = True

        # Sound
        self.sound_enabled = True
        try:
            pygame.mixer.init()
            self.shoot_sound = self.create_sound(440, 0.1)
            self.explosion_sound = self.create_sound(100, 0.3)
            self.capture_sound = self.create_sound(220, 0.5)
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

    def reset_game(self):
        self.state = MENU
        self.score = 0
        self.high_score = 0
        self.lives = 3
        self.stage = 1
        self.player = Player()
        self.player_bullets = []
        self.enemy_bullets = []
        self.enemies = []
        self.particles = []
        self.stars = [Star() for _ in range(30)]
        self.formation_offset = 0
        self.formation_direction = 1
        self.dive_timer = 0
        self.dive_cooldown = 120  # frames between dive attacks
        self.enemies_diving = []
        self.tractor_beam = None
        self.captured_player = None
        self.capture_timer = 0
        self.stage_clear_timer = 0
        self.bonus_stage_active = False
        self.bonus_enemies = []
        self.bonus_timer = 0
        self.bonus_score = 0
        self.game_over_timer = 0
        self.life_lost_timer = 0
        self.respawn_timer = 0
        self.flash_timer = 0

        self.create_enemies()

    def create_enemies(self):
        """Create enemies for the current stage."""
        self.enemies = []
        pattern_idx = (self.stage - 1) % len(FORMATION_PATTERNS)
        pattern = FORMATION_PATTERNS[pattern_idx]

        # Determine how many enemies based on stage
        num_enemies = min(len(pattern), 30 + self.stage * 2)
        selected = random.sample(pattern, min(num_enemies, len(pattern)))

        for col, row in selected:
            # Determine type based on row
            if row == 0:
                etype = BOSS_TYPE
            elif row <= 2:
                etype = FIGHTER_TYPE
            else:
                etype = SCOUT_TYPE

            enemy = Enemy(col, row, etype)
            enemy.set_formation_position(
                FORMATION_CENTER_X, FORMATION_TOP,
                col, row, FORMATION_COLS, FORMATION_ROWS
            )
            self.enemies.append(enemy)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_z:
                    if self.state == PLAYING:
                        bullets = self.player.shoot()
                        for b in bullets:
                            self.player_bullets.append(b)
                        if bullets and self.sound_enabled:
                            self.shoot_sound.play()

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

    def update_enemies(self):
        if not self.enemies:
            return

        # Update formation sway
        self.formation_offset += FORMATION_SPEED * self.formation_direction
        if abs(self.formation_offset) > FORMATION_AMPLITUDE:
            self.formation_direction *= -1

        # Update all enemies
        for enemy in self.enemies[:]:
            enemy.update(self.formation_offset)

        # Dive bombing logic
        self.dive_timer += 1
        if self.dive_timer >= self.dive_cooldown and self.enemies:
            self.dive_timer = 0
            # Pick a random enemy to dive
            available = [e for e in self.enemies
                         if e.alive and not e.entering
                         and not e.diving and not e.recovering]
            if available:
                diver = random.choice(available)
                diver.start_dive(self.player.x)
                self.enemies_diving.append(diver)

        # Enemy shooting
        for enemy in self.enemies:
            if (enemy.alive and not enemy.entering
                    and not enemy.diving and not enemy.recovering
                    and random.random() < 0.005):
                self.enemy_bullets.append(Bullet(
                    enemy.x + enemy.width // 2 - BULLET_WIDTH // 2,
                    enemy.y + enemy.height,
                    ENEMY_BULLET_SPEED, False
                ))

        # Tractor beam logic (boss enemies can capture)
        if self.tractor_beam is None and self.player.alive:
            bosses = [e for e in self.enemies
                      if e.alive and e.type == BOSS_TYPE
                      and not e.entering and not e.diving
                      and not e.recovering and not e.tractoring]
            for boss in bosses:
                if random.random() < 0.002:  # Rare chance
                    if abs(boss.x - self.player.x) < 100:
                        self.start_tractor_beam(boss)
                        break

    def start_tractor_beam(self, boss):
        """Start tractor beam to capture player."""
        self.tractor_beam = TractorBeam(
            boss.x + boss.width // 2,
            boss.y + boss.height,
            self.player.y
        )
        boss.tractoring = True
        self.state = CAPTURED
        self.capture_timer = 0
        if self.sound_enabled:
            self.capture_sound.play()

    def update_tractor_beam(self):
        """Update tractor beam capture sequence."""
        if self.tractor_beam is None:
            return

        self.capture_timer += 1
        self.tractor_beam.update()

        # Move player toward the boss
        boss = None
        for e in self.enemies:
            if e.tractoring:
                boss = e
                break

        if boss is None:
            self.tractor_beam = None
            self.state = PLAYING
            return

        # Pull player up
        dx = boss.x + boss.width // 2 - self.player.x - self.player.width // 2
        dy = boss.y + boss.height - self.player.y

        self.player.x += dx * 0.02
        self.player.y += dy * 0.02

        # Check if player reached the boss (captured)
        if abs(dy) < 10:
            self.player.alive = False
            self.captured_player = True
            boss.tractoring = False
            self.tractor_beam = None
            self.state = PLAYING
            self.lives -= 1
            self.life_lost_timer = 60
            if self.lives <= 0:
                self.state = GAME_OVER
                self.game_over_timer = 120
                if self.score > self.high_score:
                    self.high_score = self.score

    def update_bullets(self):
        """Update all bullets."""
        for bullet in self.player_bullets[:]:
            bullet.update()
            if not bullet.active:
                self.player_bullets.remove(bullet)

        for bullet in self.enemy_bullets[:]:
            bullet.update()
            if not bullet.active:
                self.enemy_bullets.remove(bullet)

    def check_collisions(self):
        """Check all bullet-enemy and bullet-player collisions."""
        # Player bullets vs enemies
        for bullet in self.player_bullets[:]:
            bullet_rect = bullet.get_rect()
            bullet_hit = False

            for enemy in self.enemies[:]:
                if enemy.alive and bullet_rect.colliderect(enemy.get_rect()):
                    # Hit!
                    enemy.alive = False
                    self.score += enemy.score_value
                    bullet_hit = True

                    # Explosion particles
                    cx = enemy.x + enemy.width // 2
                    cy = enemy.y + enemy.height // 2
                    for _ in range(12):
                        angle = random.uniform(0, math.pi * 2)
                        speed = random.uniform(1, 4)
                        self.particles.append(Particle(
                            cx, cy, enemy.color, speed, angle
                        ))

                    if self.sound_enabled:
                        self.explosion_sound.play()

                    if bullet in self.player_bullets:
                        self.player_bullets.remove(bullet)
                    break

            if bullet_hit:
                continue

            # Check if bullet hit a diving enemy (extra check)
            for enemy in self.enemies_diving[:]:
                if enemy.alive and bullet_rect.colliderect(enemy.get_rect()):
                    enemy.alive = False
                    self.score += enemy.score_value
                    bullet_hit = True

                    cx = enemy.x + enemy.width // 2
                    cy = enemy.y + enemy.height // 2
                    for _ in range(12):
                        angle = random.uniform(0, math.pi * 2)
                        speed = random.uniform(1, 4)
                        self.particles.append(Particle(
                            cx, cy, enemy.color, speed, angle
                        ))

                    if self.sound_enabled:
                        self.explosion_sound.play()

                    if bullet in self.player_bullets:
                        self.player_bullets.remove(bullet)
                    break

        # Enemy bullets vs player
        player_rect = self.player.get_rect()
        for bullet in self.enemy_bullets[:]:
            bullet_rect = bullet.get_rect()
            if bullet_rect.colliderect(player_rect) and self.player.alive:
                self.player_hit()
                if bullet in self.enemy_bullets:
                    self.enemy_bullets.remove(bullet)
                break

        # Diving enemies vs player
        for enemy in self.enemies_diving[:]:
            if (enemy.alive and enemy.diving
                    and enemy.get_rect().colliderect(player_rect)
                    and self.player.alive):
                self.player_hit()
                break

        # Remove dead enemies
        self.enemies = [e for e in self.enemies if e.alive]
        self.enemies_diving = [e for e in self.enemies_diving if e.alive]

    def player_hit(self):
        """Handle player getting hit."""
        self.lives -= 1
        self.life_lost_timer = 60

        # Explosion particles
        cx = self.player.x + self.player.width // 2
        cy = self.player.y + self.player.height // 2
        for _ in range(20):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(1, 5)
            self.particles.append(Particle(cx, cy, GREEN, speed, angle))

        if self.sound_enabled:
            self.explosion_sound.play()

        if self.lives <= 0:
            self.state = GAME_OVER
            self.game_over_timer = 120
            if self.score > self.high_score:
                self.high_score = self.score
        else:
            self.player.alive = False
            self.respawn_timer = 90  # 1.5 second respawn delay

    def check_stage_clear(self):
        """Check if all enemies are destroyed."""
        if not self.enemies and not self.enemies_diving and self.state == PLAYING:
            self.state = STAGE_CLEAR
            self.stage_clear_timer = 120

    def update_particles(self):
        """Update all particles."""
        self.particles = [p for p in self.particles if p.update()]

    def update(self):
        """Main update loop."""
        if self.state == PLAYING:
            keys = pygame.key.get_pressed()
            self.player.update(keys)

            # Auto-fire with space held
            if keys[pygame.K_SPACE] or keys[pygame.K_z]:
                bullets = self.player.shoot()
                for b in bullets:
                    self.player_bullets.append(b)
                if bullets and self.sound_enabled:
                    self.shoot_sound.play()

            self.update_enemies()
            self.update_bullets()
            self.check_collisions()
            self.check_stage_clear()
            self.update_particles()

            # Respawn player after delay
            if not self.player.alive and self.lives > 0:
                self.respawn_timer -= 1
                if self.respawn_timer <= 0:
                    self.player.reset()

        elif self.state == CAPTURED:
            self.update_tractor_beam()
            self.update_enemies()
            self.update_bullets()
            self.update_particles()

        elif self.state == STAGE_CLEAR:
            self.stage_clear_timer -= 1
            self.update_particles()
            if self.stage_clear_timer <= 0:
                self.stage += 1
                self.dive_cooldown = max(60, 120 - (self.stage - 1) * 5)
                self.create_enemies()
                self.player.reset()
                self.player_bullets.clear()
                self.enemy_bullets.clear()
                self.state = PLAYING

        elif self.state == GAME_OVER:
            self.game_over_timer -= 1
            self.update_particles()

        if self.life_lost_timer > 0:
            self.life_lost_timer -= 1

        # Update stars
        for star in self.stars:
            star.update()

    def draw_background(self):
        """Draw the starfield background."""
        self.screen.fill(DARK_BLUE)
        for star in self.stars:
            star.draw(self.screen)

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
            ship_y = SCREEN_HEIGHT - 25
            points = [
                (ship_x + 8, ship_y),
                (ship_x + 16, ship_y + 10),
                (ship_x + 14, ship_y + 16),
                (ship_x + 2, ship_y + 16),
                (ship_x, ship_y + 10),
            ]
            pygame.draw.polygon(self.screen, GREEN, points)

    def draw(self):
        """Main draw loop."""
        self.draw_background()

        if self.state == MENU:
            # Title
            title = font_huge.render("GALAGA", True, GREEN)
            self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 150))

            subtitle = font_medium.render("~ 1981 NAMCO ~", True, CYAN)
            self.screen.blit(subtitle, (SCREEN_WIDTH // 2 - subtitle.get_width() // 2, 210))

            # Animated enemy display
            cx = SCREEN_WIDTH // 2
            for i, (color, y_off) in enumerate([(RED, 0), (MAGENTA, 30), (CYAN, 60)]):
                points = [
                    (cx, 280 + y_off),
                    (cx + 12, 290 + y_off),
                    (cx + 14, 298 + y_off),
                    (cx + 10, 310 + y_off),
                    (cx - 10, 310 + y_off),
                    (cx - 14, 298 + y_off),
                    (cx - 12, 290 + y_off),
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.polygon(self.screen, WHITE, points, 1)

            start = font_medium.render("Press SPACE to Start", True, WHITE)
            self.screen.blit(start, (SCREEN_WIDTH // 2 - start.get_width() // 2, 370))

            controls = [
                "Arrow Keys / A,D - Move",
                "Space / Z - Shoot",
                "P - Pause",
            ]
            for i, text in enumerate(controls):
                ctrl = font_small.render(text, True, GRAY)
                self.screen.blit(ctrl, (SCREEN_WIDTH // 2 - ctrl.get_width() // 2, 420 + i * 22))

        elif self.state == PLAYING or self.state == CAPTURED or self.state == STAGE_CLEAR:
            # Draw enemies
            for enemy in self.enemies:
                enemy.draw(self.screen)
            for enemy in self.enemies_diving:
                enemy.draw(self.screen)

            # Draw tractor beam
            if self.tractor_beam:
                self.tractor_beam.draw(self.screen)

            # Draw player
            self.player.draw(self.screen)

            # Draw bullets
            for bullet in self.player_bullets:
                bullet.draw(self.screen)
            for bullet in self.enemy_bullets:
                bullet.draw(self.screen)

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

            if self.state == CAPTURED:
                capture_text = font_medium.render("TRACTOR BEAM!", True, CYAN)
                self.screen.blit(capture_text,
                                 (SCREEN_WIDTH // 2 - capture_text.get_width() // 2,
                                  SCREEN_HEIGHT // 2 - 80))

        elif self.state == GAME_OVER:
            # Draw final state
            for enemy in self.enemies:
                enemy.draw(self.screen)
            for enemy in self.enemies_diving:
                enemy.draw(self.screen)
            self.player.draw(self.screen)
            for bullet in self.player_bullets:
                bullet.draw(self.screen)
            for bullet in self.enemy_bullets:
                bullet.draw(self.screen)
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
    game = Galaga()
    game.run()
