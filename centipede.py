"""
Centipede - Classic 1981 arcade game implemented in Pygame.

Shoot the centipede as it snakes down through a field of mushrooms!
Each segment splits into two when hit. Clear all segments to win.
Watch out for fleas, spiders, and scorpions!

Controls:
- Mouse: Move the ship
- Left click / Space: Shoot
- R: Restart after game over
- P: Pause
"""

import pygame
import random
import sys

# Initialize pygame
pygame.init()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 700
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (50, 255, 50)
DARK_GREEN = (0, 150, 0)
RED = (255, 50, 50)
DARK_RED = (150, 0, 0)
YELLOW = (255, 255, 50)
ORANGE = (255, 165, 0)
PURPLE = (180, 50, 255)
CYAN = (50, 255, 255)
BROWN = (139, 69, 19)
GRAY = (80, 80, 80)
DARK_GRAY = (40, 40, 40)
PINK = (255, 100, 180)
BLUE = (50, 100, 255)

# Play area
PLAY_AREA_LEFT = 0
PLAY_AREA_RIGHT = SCREEN_WIDTH
PLAY_AREA_TOP = 40
PLAY_AREA_BOTTOM = SCREEN_HEIGHT - 40

# Player
PLAYER_WIDTH = 20
PLAYER_HEIGHT = 16
PLAYER_SPEED = 4

# Bullets
BULLET_SPEED = -8
BULLET_WIDTH = 3
BULLET_HEIGHT = 8
FIRE_COOLDOWN = 12

# Centipede
SEGMENT_WIDTH = 16
SEGMENT_HEIGHT = 16
CENTIPEDE_BASE_SPEED = 2.0
CENTIPEDE_DROP = SEGMENT_HEIGHT

# Mushrooms
MUSHROOM_SIZE = 14
MUSHROOM_COUNT = 40

# Flea
FLEA_WIDTH = 14
FLEA_HEIGHT = 14
FLEA_SPEED = 3
FLEA_SPAWN_CHANCE = 0.003

# Spider
SPIDER_SIZE = 18
SPIDER_SPEED = 3
SPIDER_SPAWN_CHANCE = 0.002
SPIDER_BOUNCE_CHANCE = 0.02

# Scoring
SCORE_SEGMENT = 10
SCORE_MUSHROOM = 1
SCORE_FLEA = 200
SCORE_SPIDER = 300
SCORE_SCORPION = 500

# Fonts
font_small = pygame.font.Font(None, 28)
font_medium = pygame.font.Font(None, 48)
font_large = pygame.font.Font(None, 72)

# ---------------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------------
MENU = 0
PLAYING = 1
GAME_OVER = 2
WAVE_CLEAR = 3


class Player:
    def __init__(self):
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.x = SCREEN_WIDTH // 2 - self.width // 2
        self.y = SCREEN_HEIGHT - 60
        self.speed = PLAYER_SPEED
        self.fire_cooldown = 0
        self.alive = True

    def update(self):
        if not self.alive:
            return

        # Mouse movement
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.x = mouse_x - self.width // 2
        self.y = mouse_y - self.height // 2

        # Clamp to play area
        self.x = max(PLAY_AREA_LEFT, min(self.x, PLAY_AREA_RIGHT - self.width))
        self.y = max(PLAY_AREA_TOP, min(self.y, PLAY_AREA_BOTTOM - self.height))

        # Cooldown
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

    def shoot(self):
        if self.fire_cooldown == 0 and self.alive:
            self.fire_cooldown = FIRE_COOLDOWN
            return Bullet(self.x + self.width // 2 - BULLET_WIDTH // 2,
                          self.y - BULLET_HEIGHT, BULLET_SPEED)
        return None

    def draw(self, screen):
        if not self.alive:
            return
        # Draw a small ship/arrow shape
        points = [
            (self.x + self.width // 2, self.y),  # nose
            (self.x + self.width, self.y + self.height),  # right
            (self.x + self.width // 2, self.y + self.height - 4),  # center notch
            (self.x, self.y + self.height),  # left
        ]
        pygame.draw.polygon(screen, GREEN, points)
        pygame.draw.polygon(screen, DARK_GREEN, points, 2)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def reset(self):
        self.x = SCREEN_WIDTH // 2 - self.width // 2
        self.y = SCREEN_HEIGHT - 60
        self.alive = True
        self.fire_cooldown = 0


class Bullet:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed
        self.width = BULLET_WIDTH
        self.height = BULLET_HEIGHT
        self.active = True

    def update(self):
        self.y += self.speed
        if self.y < -self.height or self.y > SCREEN_HEIGHT:
            self.active = False

    def draw(self, screen):
        pygame.draw.rect(screen, YELLOW, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, WHITE, (self.x + 1, self.y + 1, self.width - 2, self.height - 2))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class CentipedeSegment:
    def __init__(self, x, y, is_head=False, direction=1):
        self.x = x
        self.y = y
        self.width = SEGMENT_WIDTH
        self.height = SEGMENT_HEIGHT
        self.is_head = is_head
        self.direction = direction  # 1 = right, -1 = left
        self.alive = True
        self.moving_down = False
        self.down_counter = 0

    def update(self, mushrooms, speed):
        if not self.alive:
            return

        if self.moving_down:
            self.down_counter += 1
            if self.down_counter >= SEGMENT_HEIGHT:
                self.moving_down = False
                self.down_counter = 0
                self.direction *= -1
            else:
                self.y += 1
            return

        # Move horizontally
        self.x += speed * self.direction

        # Check wall collisions
        if self.x + self.width > PLAY_AREA_RIGHT:
            self.x = PLAY_AREA_RIGHT - self.width
            self.moving_down = True
        elif self.x < PLAY_AREA_LEFT:
            self.x = PLAY_AREA_LEFT
            self.moving_down = True

        # Check mushroom collisions
        for mushroom in mushrooms:
            if mushroom.alive and self.get_rect().colliderect(mushroom.get_rect()):
                # Move back and go down
                self.x -= speed * self.direction
                self.moving_down = True
                break

        # Check bottom boundary
        if self.y + self.height > PLAY_AREA_BOTTOM:
            self.y = PLAY_AREA_BOTTOM - self.height
            self.moving_down = True

    def draw(self, screen):
        if not self.alive:
            return

        color = GREEN if self.is_head else DARK_GREEN
        # Body segment
        pygame.draw.ellipse(screen, color,
                            (self.x, self.y, self.width, self.height))
        pygame.draw.ellipse(screen, DARK_GREEN,
                            (self.x, self.y, self.width, self.height), 2)

        if self.is_head:
            # Eyes
            eye_offset = 3 * self.direction
            pygame.draw.circle(screen, WHITE,
                               (self.x + self.width // 2 + eye_offset - 2, self.y + 4), 3)
            pygame.draw.circle(screen, WHITE,
                               (self.x + self.width // 2 + eye_offset + 2, self.y + 4), 3)
            pygame.draw.circle(screen, BLACK,
                               (self.x + self.width // 2 + eye_offset - 2, self.y + 4), 1)
            pygame.draw.circle(screen, BLACK,
                               (self.x + self.width // 2 + eye_offset + 2, self.y + 4), 1)
            # Antennae
            pygame.draw.line(screen, color,
                             (self.x + self.width // 2 + eye_offset - 3, self.y),
                             (self.x + self.width // 2 + eye_offset - 5, self.y - 6), 2)
            pygame.draw.line(screen, color,
                             (self.x + self.width // 2 + eye_offset + 3, self.y),
                             (self.x + self.width // 2 + eye_offset + 5, self.y - 6), 2)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Mushroom:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = MUSHROOM_SIZE
        self.alive = True
        self.hits = 0  # Takes 4 hits to destroy

    def hit(self):
        self.hits += 1
        if self.hits >= 4:
            self.alive = False
            return True  # Destroyed
        return False

    def draw(self, screen):
        if not self.alive:
            return

        # Draw mushroom
        colors = [BROWN, (160, 82, 45), (180, 90, 50), (200, 100, 60)]
        color = colors[min(self.hits, len(colors) - 1)]

        # Stem
        pygame.draw.rect(screen, (210, 180, 140),
                         (self.x + 3, self.y + self.size // 2,
                          self.size - 6, self.size // 2))
        # Cap
        pygame.draw.ellipse(screen, color,
                            (self.x, self.y, self.size, self.size))
        # Spots
        if self.hits < 3:
            pygame.draw.circle(screen, WHITE,
                               (self.x + 4, self.y + 4), 2)
            pygame.draw.circle(screen, WHITE,
                               (self.x + self.size - 4, self.y + 4), 2)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.size, self.size)


class Flea:
    def __init__(self, x):
        self.x = x
        self.y = PLAY_AREA_TOP
        self.width = FLEA_WIDTH
        self.height = FLEA_HEIGHT
        self.speed = FLEA_SPEED
        self.active = True
        self.direction = random.choice([-1, 1])

    def update(self):
        self.y += self.speed
        self.x += self.direction * 0.5
        if self.y > PLAY_AREA_BOTTOM:
            self.active = False

    def draw(self, screen):
        if not self.active:
            return
        # Draw flea
        pygame.draw.ellipse(screen, RED,
                            (self.x, self.y, self.width, self.height))
        pygame.draw.ellipse(screen, DARK_RED,
                            (self.x, self.y, self.width, self.height), 2)
        # Legs
        for i in range(3):
            lx = self.x + 2 + i * 4
            pygame.draw.line(screen, RED, (lx, self.y + self.height),
                             (lx - 2, self.y + self.height + 4), 1)
            pygame.draw.line(screen, RED, (lx, self.y + self.height),
                             (lx + 2, self.y + self.height + 4), 1)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Spider:
    def __init__(self):
        self.size = SPIDER_SIZE
        self.x = random.randint(PLAY_AREA_LEFT, PLAY_AREA_RIGHT - self.size)
        self.y = random.randint(PLAY_AREA_TOP + 100, PLAY_AREA_BOTTOM - self.size)
        self.speed = SPIDER_SPEED
        self.active = True
        self.dx = random.choice([-1, 1]) * self.speed
        self.dy = random.choice([-1, 1]) * self.speed

    def update(self):
        self.x += self.dx
        self.y += self.dy

        # Bounce off walls
        if self.x + self.size > PLAY_AREA_RIGHT:
            self.x = PLAY_AREA_RIGHT - self.size
            self.dx = -self.dx
        elif self.x < PLAY_AREA_LEFT:
            self.x = PLAY_AREA_LEFT
            self.dx = -self.dx

        if self.y + self.size > PLAY_AREA_BOTTOM:
            self.y = PLAY_AREA_BOTTOM - self.size
            self.dy = -self.dy
        elif self.y < PLAY_AREA_TOP:
            self.y = PLAY_AREA_TOP
            self.dy = -self.dy

        # Random direction changes
        if random.random() < SPIDER_BOUNCE_CHANCE:
            self.dx = random.choice([-1, 1]) * self.speed
            self.dy = random.choice([-1, 1]) * self.speed

    def draw(self, screen):
        if not self.active:
            return
        # Draw spider body
        cx = self.x + self.size // 2
        cy = self.y + self.size // 2
        pygame.draw.circle(screen, PINK, (cx, cy), self.size // 2)
        pygame.draw.circle(screen, DARK_RED, (cx, cy), self.size // 2, 2)
        # Eyes
        pygame.draw.circle(screen, WHITE, (cx - 3, cy - 2), 3)
        pygame.draw.circle(screen, WHITE, (cx + 3, cy - 2), 3)
        pygame.draw.circle(screen, BLACK, (cx - 3, cy - 2), 1)
        pygame.draw.circle(screen, BLACK, (cx + 3, cy - 2), 1)
        # Legs
        for angle in [0.3, 0.6, 0.9, 1.2]:
            for sign in [-1, 1]:
                leg_x = cx + sign * (self.size // 2) * pygame.math.Vector2(1, 0).rotate(angle * 180 / 3.14159 * sign).x
                leg_y = cy + (self.size // 2) * pygame.math.Vector2(1, 0).rotate(angle * 180 / 3.14159 * sign).y
                pygame.draw.line(screen, PINK, (cx, cy), (leg_x, leg_y), 2)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.size, self.size)


class CentipedeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Centipede")
        self.clock = pygame.time.Clock()
        self.running = True

        self.reset_game()

    def reset_game(self):
        self.state = PLAYING
        self.score = 0
        self.high_score = 0
        self.lives = 3
        self.wave = 1
        self.player = Player()
        self.bullets = []
        self.segments = []
        self.mushrooms = []
        self.fleas = []
        self.spiders = []
        self.speed = CENTIPEDE_BASE_SPEED
        self.wave_clear_timer = 0
        self.game_over_timer = 0
        self.life_lost_timer = 0

        self.create_mushrooms()
        self.create_centipede()

    def create_mushrooms(self):
        self.mushrooms = []
        # Place mushrooms randomly in the upper 2/3 of the play area
        for _ in range(MUSHROOM_COUNT):
            x = random.randint(PLAY_AREA_LEFT, PLAY_AREA_RIGHT - MUSHROOM_SIZE)
            y = random.randint(PLAY_AREA_TOP + 20,
                               PLAY_AREA_BOTTOM - 150)
            # Avoid overlapping
            overlap = False
            new_rect = pygame.Rect(x, y, MUSHROOM_SIZE, MUSHROOM_SIZE)
            for m in self.mushrooms:
                if new_rect.colliderect(m.get_rect()):
                    overlap = True
                    break
            if not overlap:
                self.mushrooms.append(Mushroom(x, y))

    def create_centipede(self):
        self.segments = []
        # Start from top-left, going right
        start_x = PLAY_AREA_LEFT
        start_y = PLAY_AREA_TOP + 10
        length = 10 + self.wave * 2  # More segments per wave

        for i in range(length):
            x = start_x + i * SEGMENT_WIDTH
            y = start_y
            is_head = (i == length - 1)
            self.segments.append(CentipedeSegment(x, y, is_head, 1))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.state == PLAYING:
                    bullet = self.player.shoot()
                    if bullet:
                        self.bullets.append(bullet)

                if event.key == pygame.K_r and self.state == GAME_OVER:
                    self.reset_game()

                if event.key == pygame.K_p and self.state == PLAYING:
                    self.state = MENU
                elif event.key == pygame.K_p and self.state == MENU:
                    self.state = PLAYING

                if event.key == pygame.K_SPACE and self.state == MENU:
                    self.state = PLAYING

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.state == PLAYING:
                    bullet = self.player.shoot()
                    if bullet:
                        self.bullets.append(bullet)

    def update_centipede(self):
        for segment in self.segments:
            segment.update(self.mushrooms, self.speed)

    def update_bullets(self):
        for bullet in self.bullets[:]:
            bullet.update()
            if not bullet.active:
                self.bullets.remove(bullet)

    def update_enemies(self):
        # Fleas
        if random.random() < FLEA_SPAWN_CHANCE and len(self.fleas) < 3:
            fx = random.randint(PLAY_AREA_LEFT, PLAY_AREA_RIGHT - FLEA_WIDTH)
            self.fleas.append(Flea(fx))

        for flea in self.fleas[:]:
            flea.update()
            if not flea.active:
                self.fleas.remove(flea)

        # Spiders
        if random.random() < SPIDER_SPAWN_CHANCE and len(self.spiders) < 2:
            self.spiders.append(Spider())

        for spider in self.spiders[:]:
            spider.update()
            # Spiders have a limited lifetime
            if random.random() < 0.005:
                spider.active = False
            if not spider.active:
                self.spiders.remove(spider)

    def check_collisions(self):
        # Bullets vs centipede segments
        for bullet in self.bullets[:]:
            bullet_rect = bullet.get_rect()
            bullet_hit = False

            for segment in self.segments[:]:
                if segment.alive and bullet_rect.colliderect(segment.get_rect()):
                    segment.alive = False
                    self.score += SCORE_SEGMENT
                    bullet_hit = True

                    # Split the centipede: create two new segments
                    idx = self.segments.index(segment)
                    # Left part continues left, right part continues right
                    if idx > 0:
                        prev = self.segments[idx - 1]
                        prev.is_head = True
                        prev.direction = -1
                    if idx < len(self.segments) - 1:
                        next_seg = self.segments[idx + 1]
                        next_seg.is_head = True
                        next_seg.direction = 1

                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break

            if bullet_hit:
                continue

            # Bullets vs mushrooms
            for mushroom in self.mushrooms[:]:
                if mushroom.alive and bullet_rect.colliderect(mushroom.get_rect()):
                    if mushroom.hit():
                        self.score += SCORE_MUSHROOM
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break

            if bullet_hit:
                continue

            # Bullets vs fleas
            for flea in self.fleas[:]:
                if flea.active and bullet_rect.colliderect(flea.get_rect()):
                    flea.active = False
                    self.score += SCORE_FLEA
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break

            if bullet_hit:
                continue

            # Bullets vs spiders
            for spider in self.spiders[:]:
                if spider.active and bullet_rect.colliderect(spider.get_rect()):
                    spider.active = False
                    self.score += SCORE_SPIDER
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break

        # Player vs centipede
        player_rect = self.player.get_rect()
        for segment in self.segments:
            if segment.alive and segment.get_rect().colliderect(player_rect):
                self.player_hit()
                break

        # Player vs fleas
        for flea in self.fleas:
            if flea.active and flea.get_rect().colliderect(player_rect):
                self.player_hit()
                break

        # Player vs spiders
        for spider in self.spiders:
            if spider.active and spider.get_rect().colliderect(player_rect):
                self.player_hit()
                break

        # Remove dead segments
        self.segments = [s for s in self.segments if s.alive]

    def player_hit(self):
        self.lives -= 1
        self.life_lost_timer = 60
        if self.lives <= 0:
            self.state = GAME_OVER
            self.game_over_timer = 120
            if self.score > self.high_score:
                self.high_score = self.score
        else:
            self.player.alive = False
            self.game_over_timer = 60

    def check_wave_clear(self):
        if not self.segments and self.state == PLAYING:
            self.state = WAVE_CLEAR
            self.wave_clear_timer = 120

    def update(self):
        if self.state == PLAYING:
            self.player.update()

            # Auto-fire with space held
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                bullet = self.player.shoot()
                if bullet:
                    self.bullets.append(bullet)

            # Auto-fire with mouse held
            if pygame.mouse.get_pressed()[0]:
                bullet = self.player.shoot()
                if bullet:
                    self.bullets.append(bullet)

            self.update_centipede()
            self.update_bullets()
            self.update_enemies()
            self.check_collisions()
            self.check_wave_clear()

            if not self.player.alive and self.lives > 0:
                self.game_over_timer -= 1
                if self.game_over_timer <= 0:
                    self.player.reset()

        elif self.state == WAVE_CLEAR:
            self.wave_clear_timer -= 1
            if self.wave_clear_timer <= 0:
                self.wave += 1
                self.speed = CENTIPEDE_BASE_SPEED + (self.wave - 1) * 0.15
                self.create_mushrooms()
                self.create_centipede()
                self.player.reset()
                self.bullets.clear()
                self.fleas.clear()
                self.spiders.clear()
                self.state = PLAYING

        elif self.state == GAME_OVER:
            self.game_over_timer -= 1

        if self.life_lost_timer > 0:
            self.life_lost_timer -= 1

    def draw_background(self):
        # Play area border
        pygame.draw.rect(self.screen, DARK_GRAY,
                         (PLAY_AREA_LEFT, PLAY_AREA_TOP,
                          PLAY_AREA_RIGHT - PLAY_AREA_LEFT,
                          PLAY_AREA_BOTTOM - PLAY_AREA_TOP), 2)
        # Grass-like bottom area
        pygame.draw.rect(self.screen, (0, 40, 0),
                         (PLAY_AREA_LEFT, PLAY_AREA_BOTTOM - 20,
                          PLAY_AREA_RIGHT, 20))

    def draw_hud(self):
        # Score
        score_text = font_medium.render(f"{self.score}", True, WHITE)
        self.screen.blit(score_text, (20, 5))

        # High score
        if self.score > self.high_score:
            self.high_score = self.score
        high_text = font_small.render(f"HIGH SCORE: {self.high_score}", True, WHITE)
        self.screen.blit(high_text, (SCREEN_WIDTH // 2 - high_text.get_width() // 2, 5))

        # Wave
        wave_text = font_small.render(f"WAVE {self.wave}", True, WHITE)
        self.screen.blit(wave_text, (SCREEN_WIDTH - 120, 5))

        # Lives
        lives_text = font_small.render(f"LIVES: {self.lives}", True, GREEN)
        self.screen.blit(lives_text, (20, SCREEN_HEIGHT - 25))

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_background()

        if self.state == MENU:
            title = font_large.render("CENTIPEDE", True, GREEN)
            self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 200))

            start = font_medium.render("Press SPACE or Click to Start", True, WHITE)
            self.screen.blit(start, (SCREEN_WIDTH // 2 - start.get_width() // 2, 300))

            controls = [
                "Mouse - Move",
                "Left Click / Space - Shoot",
                "P - Pause",
            ]
            for i, text in enumerate(controls):
                ctrl = font_small.render(text, True, GRAY)
                self.screen.blit(ctrl, (SCREEN_WIDTH // 2 - ctrl.get_width() // 2, 370 + i * 30))

        elif self.state == PLAYING or self.state == WAVE_CLEAR:
            # Draw mushrooms
            for mushroom in self.mushrooms:
                mushroom.draw(self.screen)

            # Draw centipede
            for segment in self.segments:
                segment.draw(self.screen)

            # Draw fleas
            for flea in self.fleas:
                flea.draw(self.screen)

            # Draw spiders
            for spider in self.spiders:
                spider.draw(self.screen)

            # Draw player
            self.player.draw(self.screen)

            # Draw bullets
            for bullet in self.bullets:
                bullet.draw(self.screen)

            self.draw_hud()

            # Life lost popup
            if self.life_lost_timer > 0:
                life_lost_text = font_large.render("LIFE LOST!", True, RED)
                self.screen.blit(life_lost_text,
                                 (SCREEN_WIDTH // 2 - life_lost_text.get_width() // 2,
                                  SCREEN_HEIGHT // 2 - 50))

            if self.state == WAVE_CLEAR:
                clear_text = font_large.render("WAVE CLEAR!", True, GREEN)
                self.screen.blit(clear_text,
                                 (SCREEN_WIDTH // 2 - clear_text.get_width() // 2,
                                  SCREEN_HEIGHT // 2 - 50))

        elif self.state == GAME_OVER:
            # Draw final state
            for mushroom in self.mushrooms:
                mushroom.draw(self.screen)
            for segment in self.segments:
                segment.draw(self.screen)
            for flea in self.fleas:
                flea.draw(self.screen)
            for spider in self.spiders:
                spider.draw(self.screen)
            self.player.draw(self.screen)
            for bullet in self.bullets:
                bullet.draw(self.screen)
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
                              SCREEN_HEIGHT // 2 - 10))

            if self.score >= self.high_score and self.score > 0:
                new_high = font_medium.render("NEW HIGH SCORE!", True, YELLOW)
                self.screen.blit(new_high,
                                 (SCREEN_WIDTH // 2 - new_high.get_width() // 2,
                                  SCREEN_HEIGHT // 2 + 40))

            restart = font_medium.render("Press R to Restart", True, WHITE)
            self.screen.blit(restart,
                             (SCREEN_WIDTH // 2 - restart.get_width() // 2,
                              SCREEN_HEIGHT // 2 + 100))

        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = CentipedeGame()
    game.run()