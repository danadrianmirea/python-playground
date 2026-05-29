"""
Space Invaders - Classic 1978 arcade game implemented in Pygame.

Defend Earth from waves of descending aliens!
Move left/right with arrow keys, shoot with Space.
Destroy all aliens to advance to the next wave.
Don't let them reach the bottom!

Controls:
- Arrow keys / A,D: Move left/right
- Space: Shoot
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
YELLOW = (255, 255, 50)
ORANGE = (255, 165, 0)
PURPLE = (180, 50, 255)
CYAN = (50, 255, 255)
GRAY = (80, 80, 80)
DARK_GRAY = (40, 40, 40)

# Player
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 30
PLAYER_SPEED = 5
PLAYER_Y = SCREEN_HEIGHT - 60

# Bullets
PLAYER_BULLET_SPEED = -10
ALIEN_BULLET_SPEED = 5
BULLET_WIDTH = 4
BULLET_HEIGHT = 12
PLAYER_FIRE_COOLDOWN = 60  # frames (1 shot per second at 60 FPS)

# Aliens
ALIEN_COLS = 11
ALIEN_ROWS = 5
ALIEN_WIDTH = 40
ALIEN_HEIGHT = 30
ALIEN_PADDING = 10
ALIEN_OFFSET_Y = 60
ALIEN_BASE_SPEED = 0.4
ALIEN_DROP_DISTANCE = 20
ALIEN_SHOOT_COOLDOWN = 60  # frames between alien shots

# Mystery ship
MYSTERY_SHIP_WIDTH = 40
MYSTERY_SHIP_HEIGHT = 20
MYSTERY_SHIP_SPEED = 3
MYSTERY_SHIP_SCORE = [50, 100, 150, 200, 300]
MYSTERY_SPAWN_CHANCE = 0.005  # per frame

# Shields
SHIELD_BLOCKS_X = 4
SHIELD_BLOCKS_Y = 4
SHIELD_BLOCK_SIZE = 8
SHIELD_COUNT = 4
SHIELD_Y = PLAYER_Y - 100

# Scoring
SCORE_ALIEN_TIER = [10, 20, 30]  # bottom, middle, top rows

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
        self.y = PLAYER_Y
        self.speed = PLAYER_SPEED
        self.fire_cooldown = 0
        self.alive = True

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

    def shoot(self):
        if self.fire_cooldown == 0 and self.alive:
            self.fire_cooldown = PLAYER_FIRE_COOLDOWN
            return Bullet(self.x + self.width // 2 - BULLET_WIDTH // 2,
                          self.y - BULLET_HEIGHT, PLAYER_BULLET_SPEED, True)
        return None

    def draw(self, screen):
        if not self.alive:
            return
        # Draw a simple ship shape
        points = [
            (self.x + self.width // 2, self.y),  # nose
            (self.x + self.width, self.y + self.height),  # right wing
            (self.x + self.width - 10, self.y + self.height - 5),
            (self.x + self.width // 2 + 5, self.y + self.height - 5),
            (self.x + self.width // 2 + 5, self.y + self.height),
            (self.x + self.width // 2 - 5, self.y + self.height),
            (self.x + self.width // 2 - 5, self.y + self.height - 5),
            (self.x + 10, self.y + self.height - 5),
            (self.x, self.y + self.height),  # left wing
        ]
        pygame.draw.polygon(screen, GREEN, points)
        pygame.draw.polygon(screen, DARK_GREEN, points, 2)

        # Cockpit
        pygame.draw.circle(screen, CYAN,
                           (self.x + self.width // 2, self.y + 10), 5)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def reset(self):
        self.x = SCREEN_WIDTH // 2 - self.width // 2
        self.alive = True
        self.fire_cooldown = 0


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
            color = YELLOW
            # Draw a bright bullet with trail
            pygame.draw.rect(screen, WHITE,
                             (self.x, self.y, self.width, self.height))
            pygame.draw.rect(screen, color,
                             (self.x + 1, self.y + 2, self.width - 2, self.height - 4))
        else:
            color = RED
            pygame.draw.rect(screen, color,
                             (self.x, self.y, self.width, self.height))
            # Glow effect
            pygame.draw.circle(screen, (255, 100, 100),
                               (self.x + self.width // 2, self.y + self.height // 2),
                               self.width)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Alien:
    def __init__(self, x, y, tier):
        self.x = x
        self.y = y
        self.width = ALIEN_WIDTH
        self.height = ALIEN_HEIGHT
        self.tier = tier  # 0=bottom, 1=middle, 2=top
        self.alive = True
        self.animation_frame = 0
        self.animation_timer = 0

    def update(self):
        self.animation_timer += 1
        if self.animation_timer > 20:
            self.animation_timer = 0
            self.animation_frame = (self.animation_frame + 1) % 2

    def draw(self, screen):
        if not self.alive:
            return

        cx = self.x + self.width // 2
        cy = self.y + self.height // 2

        if self.tier == 0:  # Bottom tier - squid-like (10 pts)
            color = PURPLE
            # Body
            pygame.draw.ellipse(screen, color,
                                (self.x + 5, self.y + 5, self.width - 10, self.height - 10))
            # Eyes
            pygame.draw.circle(screen, WHITE, (cx - 7, cy - 3), 4)
            pygame.draw.circle(screen, WHITE, (cx + 7, cy - 3), 4)
            pygame.draw.circle(screen, BLACK, (cx - 7, cy - 3), 2)
            pygame.draw.circle(screen, BLACK, (cx + 7, cy - 3), 2)
            # Legs
            leg_offset = 3 if self.animation_frame == 0 else -3
            for i in range(-2, 3):
                pygame.draw.line(screen, color,
                                 (cx + i * 6, self.y + self.height - 8),
                                 (cx + i * 6, self.y + self.height - 5 + leg_offset), 2)

        elif self.tier == 1:  # Middle tier - crab-like (20 pts)
            color = ORANGE
            # Body
            pygame.draw.ellipse(screen, color,
                                (self.x + 3, self.y + 5, self.width - 6, self.height - 10))
            # Eyes
            pygame.draw.circle(screen, WHITE, (cx - 6, cy - 5), 4)
            pygame.draw.circle(screen, WHITE, (cx + 6, cy - 5), 4)
            pygame.draw.circle(screen, BLACK, (cx - 6, cy - 5), 2)
            pygame.draw.circle(screen, BLACK, (cx + 6, cy - 5), 2)
            # Claws
            claw_offset = 3 if self.animation_frame == 0 else -3
            pygame.draw.rect(screen, color,
                             (self.x - 3, self.y + 5 + claw_offset, 6, 8))
            pygame.draw.rect(screen, color,
                             (self.x + self.width - 3, self.y + 5 + claw_offset, 6, 8))

        else:  # Top tier - bug-like (30 pts)
            color = RED
            # Body
            pygame.draw.ellipse(screen, color,
                                (self.x + 2, self.y + 3, self.width - 4, self.height - 6))
            # Eyes
            pygame.draw.circle(screen, WHITE, (cx - 5, cy - 4), 3)
            pygame.draw.circle(screen, WHITE, (cx + 5, cy - 4), 3)
            pygame.draw.circle(screen, BLACK, (cx - 5, cy - 4), 2)
            pygame.draw.circle(screen, BLACK, (cx + 5, cy - 4), 2)
            # Antennae
            ant_offset = 3 if self.animation_frame == 0 else -3
            pygame.draw.line(screen, color, (cx - 6, self.y + 3),
                             (cx - 10, self.y - 5 + ant_offset), 2)
            pygame.draw.line(screen, color, (cx + 6, self.y + 3),
                             (cx + 10, self.y - 5 + ant_offset), 2)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class MysteryShip:
    def __init__(self):
        self.width = MYSTERY_SHIP_WIDTH
        self.height = MYSTERY_SHIP_HEIGHT
        self.x = -self.width
        self.y = 30
        self.speed = MYSTERY_SHIP_SPEED
        self.active = False
        self.direction = 1  # 1 = right, -1 = left
        self.score_value = random.choice(MYSTERY_SHIP_SCORE)

    def spawn(self):
        self.active = True
        self.direction = random.choice([-1, 1])
        if self.direction == 1:
            self.x = -self.width
        else:
            self.x = SCREEN_WIDTH
        self.score_value = random.choice(MYSTERY_SHIP_SCORE)

    def update(self):
        if not self.active:
            return

        self.x += self.speed * self.direction

        # Check if off screen
        if self.direction == 1 and self.x > SCREEN_WIDTH:
            self.active = False
        elif self.direction == -1 and self.x < -self.width:
            self.active = False

    def draw(self, screen):
        if not self.active:
            return

        # Draw a UFO shape
        cx = self.x + self.width // 2
        cy = self.y + self.height // 2

        # Dome
        pygame.draw.ellipse(screen, CYAN,
                            (self.x + 10, self.y, self.width - 20, self.height // 2))
        # Body
        pygame.draw.ellipse(screen, GRAY,
                            (self.x, self.y + 5, self.width, self.height - 5))
        # Lights
        light_colors = [RED, GREEN, YELLOW, CYAN]
        for i, color in enumerate(light_colors):
            lx = self.x + 5 + i * 9
            pygame.draw.circle(screen, color, (lx, cy + 2), 3)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Shield:
    def __init__(self, x, y):
        self.blocks = []
        for row in range(SHIELD_BLOCKS_Y):
            for col in range(SHIELD_BLOCKS_X):
                # Create a pattern that looks like a shield
                if row == 0 and (col == 0 or col == SHIELD_BLOCKS_X - 1):
                    continue  # Missing corners
                bx = x + col * SHIELD_BLOCK_SIZE
                by = y + row * SHIELD_BLOCK_SIZE
                self.blocks.append(pygame.Rect(bx, by, SHIELD_BLOCK_SIZE, SHIELD_BLOCK_SIZE))

    def hit(self, bullet_rect):
        """Check if a bullet hits this shield. Returns True if hit."""
        for block in self.blocks[:]:
            if block.colliderect(bullet_rect):
                self.blocks.remove(block)
                return True
        return False

    def draw(self, screen):
        for block in self.blocks:
            pygame.draw.rect(screen, GREEN, block)
            pygame.draw.rect(screen, DARK_GREEN, block, 1)

    def is_destroyed(self):
        return len(self.blocks) == 0


class SpaceInvaders:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Space Invaders")
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
        self.player_bullets = []
        self.alien_bullets = []
        self.aliens = []
        self.mystery_ship = MysteryShip()
        self.shields = []
        self.alien_direction = 1
        self.alien_speed = ALIEN_BASE_SPEED
        self.alien_shoot_timer = 0
        self.alien_shoot_cooldown = ALIEN_SHOOT_COOLDOWN
        self.wave_clear_timer = 0
        self.game_over_timer = 0
        self.life_lost_timer = 0

        self.create_aliens()
        self.create_shields()

    def create_aliens(self):
        self.aliens = []
        total_width = ALIEN_COLS * (ALIEN_WIDTH + ALIEN_PADDING) - ALIEN_PADDING
        start_x = (SCREEN_WIDTH - total_width) // 2

        for row in range(ALIEN_ROWS):
            for col in range(ALIEN_COLS):
                x = start_x + col * (ALIEN_WIDTH + ALIEN_PADDING)
                y = ALIEN_OFFSET_Y + row * (ALIEN_HEIGHT + ALIEN_PADDING)
                tier = 2 - (row // 2)  # 0=bottom, 1=middle, 2=top
                if tier < 0:
                    tier = 0
                self.aliens.append(Alien(x, y, tier))

    def create_shields(self):
        self.shields = []
        shield_spacing = SCREEN_WIDTH // (SHIELD_COUNT + 1)
        for i in range(SHIELD_COUNT):
            sx = shield_spacing * (i + 1) - (SHIELD_BLOCKS_X * SHIELD_BLOCK_SIZE) // 2
            self.shields.append(Shield(sx, SHIELD_Y))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.state == PLAYING:
                    bullet = self.player.shoot()
                    if bullet:
                        self.player_bullets.append(bullet)

                if event.key == pygame.K_r and self.state == GAME_OVER:
                    self.reset_game()

                if event.key == pygame.K_p and self.state == PLAYING:
                    self.state = MENU
                elif event.key == pygame.K_p and self.state == MENU:
                    self.state = PLAYING

                if event.key == pygame.K_SPACE and self.state == MENU:
                    self.state = PLAYING

    def update_aliens(self):
        if not self.aliens:
            return

        # Calculate speed based on remaining aliens
        remaining = len(self.aliens)
        total_initial = ALIEN_COLS * ALIEN_ROWS
        speed_mult = 1.0 + (1.0 - remaining / total_initial) * 1.0
        current_speed = self.alien_speed * speed_mult

        # Move aliens
        move_down = False
        for alien in self.aliens:
            alien.x += current_speed * self.alien_direction
            alien.update()

            # Check if any alien hits the edge
            if alien.x + alien.width >= SCREEN_WIDTH:
                self.alien_direction = -1
                move_down = True
            elif alien.x <= 0:
                self.alien_direction = 1
                move_down = True

        if move_down:
            for alien in self.aliens:
                alien.y += ALIEN_DROP_DISTANCE

        # Alien shooting
        self.alien_shoot_timer += 1
        if self.alien_shoot_timer >= self.alien_shoot_cooldown and self.aliens:
            self.alien_shoot_timer = 0
            # Pick a random alien to shoot
            shooter = random.choice(self.aliens)
            self.alien_bullets.append(
                Bullet(shooter.x + shooter.width // 2 - BULLET_WIDTH // 2,
                       shooter.y + shooter.height,
                       ALIEN_BULLET_SPEED, False)
            )

        # Mystery ship
        if not self.mystery_ship.active and random.random() < MYSTERY_SPAWN_CHANCE:
            self.mystery_ship.spawn()
        self.mystery_ship.update()

    def update_bullets(self):
        # Player bullets
        for bullet in self.player_bullets[:]:
            bullet.update()
            if not bullet.active:
                self.player_bullets.remove(bullet)

        # Alien bullets
        for bullet in self.alien_bullets[:]:
            bullet.update()
            if not bullet.active:
                self.alien_bullets.remove(bullet)

    def check_collisions(self):
        # Player bullets vs aliens
        for bullet in self.player_bullets[:]:
            bullet_rect = bullet.get_rect()
            bullet_hit = False

            # Check shields
            for shield in self.shields:
                if shield.hit(bullet_rect):
                    bullet_hit = True
                    break
            if bullet_hit:
                self.player.fire_cooldown = 0  # Reset cooldown on hit
                if bullet in self.player_bullets:
                    self.player_bullets.remove(bullet)
                continue

            # Check aliens
            for alien in self.aliens[:]:
                if alien.alive and bullet_rect.colliderect(alien.get_rect()):
                    alien.alive = False
                    self.score += SCORE_ALIEN_TIER[alien.tier]
                    bullet_hit = True
                    if bullet in self.player_bullets:
                        self.player_bullets.remove(bullet)
                    break

            if bullet_hit:
                self.player.fire_cooldown = 0  # Reset cooldown on hit
                continue

            # Check mystery ship
            if self.mystery_ship.active and bullet_rect.colliderect(self.mystery_ship.get_rect()):
                self.score += self.mystery_ship.score_value
                self.mystery_ship.active = False
                bullet_hit = True
                if bullet in self.player_bullets:
                    self.player_bullets.remove(bullet)

            if bullet_hit:
                self.player.fire_cooldown = 0  # Reset cooldown on hit
                continue

        # Alien bullets vs player
        player_rect = self.player.get_rect()
        for bullet in self.alien_bullets[:]:
            bullet_rect = bullet.get_rect()

            # Check shields
            shield_hit = False
            for shield in self.shields:
                if shield.hit(bullet_rect):
                    shield_hit = True
                    break
            if shield_hit:
                if bullet in self.alien_bullets:
                    self.alien_bullets.remove(bullet)
                continue

            # Check player
            if bullet_rect.colliderect(player_rect):
                self.player_hit()
                if bullet in self.alien_bullets:
                    self.alien_bullets.remove(bullet)
                break

        # Aliens vs player (if they reach the player)
        for alien in self.aliens:
            if alien.alive and alien.get_rect().colliderect(player_rect):
                self.player_hit()
                break

        # Aliens vs shields
        for alien in self.aliens[:]:
            if not alien.alive:
                continue
            alien_rect = alien.get_rect()
            for shield in self.shields[:]:
                for block in shield.blocks[:]:
                    if block.colliderect(alien_rect):
                        shield.blocks.remove(block)

        # Remove dead aliens
        self.aliens = [a for a in self.aliens if a.alive]

        # Remove destroyed shields
        self.shields = [s for s in self.shields if not s.is_destroyed()]

    def player_hit(self):
        self.lives -= 1
        self.life_lost_timer = 60  # Show "LIFE LOST" for 1 second
        if self.lives <= 0:
            self.state = GAME_OVER
            self.game_over_timer = 120
            if self.score > self.high_score:
                self.high_score = self.score
        else:
            self.player.alive = False
            # Brief respawn delay
            self.game_over_timer = 60

    def check_wave_clear(self):
        if not self.aliens and self.state == PLAYING:
            self.state = WAVE_CLEAR
            self.wave_clear_timer = 120

    def update(self):
        if self.state == PLAYING:
            keys = pygame.key.get_pressed()
            self.player.update(keys)

            # Auto-fire with space held
            if keys[pygame.K_SPACE]:
                bullet = self.player.shoot()
                if bullet:
                    self.player_bullets.append(bullet)

            self.update_aliens()
            self.update_bullets()
            self.check_collisions()
            self.check_wave_clear()

            # Respawn player after delay
            if not self.player.alive and self.lives > 0:
                self.game_over_timer -= 1
                if self.game_over_timer <= 0:
                    self.player.reset()

        elif self.state == WAVE_CLEAR:
            self.wave_clear_timer -= 1
            if self.wave_clear_timer <= 0:
                self.wave += 1
                self.alien_speed = ALIEN_BASE_SPEED + (self.wave - 1) * 0.08
                self.alien_shoot_cooldown = max(30, ALIEN_SHOOT_COOLDOWN - (self.wave - 1) * 5)
                self.create_aliens()
                self.create_shields()
                self.player.reset()
                self.player_bullets.clear()
                self.alien_bullets.clear()
                self.state = PLAYING

        elif self.state == GAME_OVER:
            self.game_over_timer -= 1

        if self.life_lost_timer > 0:
            self.life_lost_timer -= 1

    def draw_background(self):
        # Starfield
        for _ in range(50):
            sx = random.randint(0, SCREEN_WIDTH)
            sy = random.randint(0, SCREEN_HEIGHT)
            brightness = random.randint(50, 200)
            pygame.draw.circle(self.screen, (brightness, brightness, brightness), (sx, sy), 1)

    def draw_hud(self):
        # Score
        score_text = font_medium.render(f"{self.score}", True, WHITE)
        self.screen.blit(score_text, (20, 10))

        # High score
        if self.score > self.high_score:
            self.high_score = self.score
        high_text = font_small.render(f"HIGH SCORE: {self.high_score}", True, WHITE)
        self.screen.blit(high_text, (SCREEN_WIDTH // 2 - high_text.get_width() // 2, 10))

        # Wave
        wave_text = font_small.render(f"WAVE {self.wave}", True, WHITE)
        self.screen.blit(wave_text, (SCREEN_WIDTH - 120, 10))

        # Lives
        lives_text = font_small.render(f"LIVES: {self.lives}", True, GREEN)
        self.screen.blit(lives_text, (20, SCREEN_HEIGHT - 30))

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_background()

        if self.state == MENU:
            title = font_large.render("SPACE INVADERS", True, GREEN)
            self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 200))

            start = font_medium.render("Press SPACE to Start", True, WHITE)
            self.screen.blit(start, (SCREEN_WIDTH // 2 - start.get_width() // 2, 300))

            controls = [
                "Arrow Keys / A,D - Move",
                "Space - Shoot",
                "P - Pause",
            ]
            for i, text in enumerate(controls):
                ctrl = font_small.render(text, True, GRAY)
                self.screen.blit(ctrl, (SCREEN_WIDTH // 2 - ctrl.get_width() // 2, 370 + i * 30))

        elif self.state == PLAYING or self.state == WAVE_CLEAR:
            # Draw shields
            for shield in self.shields:
                shield.draw(self.screen)

            # Draw aliens
            for alien in self.aliens:
                alien.draw(self.screen)

            # Draw mystery ship
            self.mystery_ship.draw(self.screen)

            # Draw player
            self.player.draw(self.screen)

            # Draw bullets
            for bullet in self.player_bullets:
                bullet.draw(self.screen)
            for bullet in self.alien_bullets:
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
            for shield in self.shields:
                shield.draw(self.screen)
            for alien in self.aliens:
                alien.draw(self.screen)
            self.mystery_ship.draw(self.screen)
            self.player.draw(self.screen)
            for bullet in self.player_bullets:
                bullet.draw(self.screen)
            for bullet in self.alien_bullets:
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
    game = SpaceInvaders()
    game.run()