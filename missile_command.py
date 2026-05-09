# missile_command.py - Missile Command / DEFCON-style defense game
#
# Defend your cities from incoming enemy missiles.
# Click to launch interceptor missiles. Each click creates an explosion
# at the clicked position that destroys any enemy missiles passing through it.
# You have limited ammo per wave. Survive as many waves as possible!

import pygame
import random
import math

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Missile Command")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 150, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
DARK_GREEN = (0, 100, 0)
DARK_RED = (100, 0, 0)
GREY = (100, 100, 100)
LIGHT_BLUE = (100, 200, 255)

# Game constants
FPS = 60
GROUND_Y = SCREEN_HEIGHT - 50
NUM_CITIES = 6
CITY_WIDTH = 40
CITY_HEIGHT = 25
AMMO_PER_WAVE = 30
INTERCEPTOR_SPEED = 14
ENEMY_MISSILE_SPEED_BASE = 1.2
EXPLOSION_RADIUS = 50
EXPLOSION_DURATION = 35  # frames

# Fonts
font = pygame.font.Font(None, 36)
big_font = pygame.font.Font(None, 72)
small_font = pygame.font.Font(None, 24)


class City:
    def __init__(self, x):
        self.x = x
        self.y = GROUND_Y - CITY_HEIGHT
        self.width = CITY_WIDTH
        self.height = CITY_HEIGHT
        self.alive = True

    def draw(self, surface):
        if not self.alive:
            return
        # Draw building
        points = [
            (self.x, self.y + self.height),
            (self.x + 5, self.y + self.height - 10),
            (self.x + 5, self.y + 5),
            (self.x + 10, self.y),
            (self.x + 15, self.y + 5),
            (self.x + 15, self.y + self.height - 10),
            (self.x + 20, self.y + self.height - 5),
            (self.x + 25, self.y + self.height - 10),
            (self.x + 25, self.y + 5),
            (self.x + 30, self.y),
            (self.x + 35, self.y + 5),
            (self.x + 35, self.y + self.height - 10),
            (self.x + self.width, self.y + self.height),
        ]
        pygame.draw.polygon(surface, BLUE, points)
        pygame.draw.polygon(surface, LIGHT_BLUE, points, 2)
        # Windows
        for wx in [self.x + 8, self.x + 18, self.x + 28]:
            pygame.draw.rect(surface, YELLOW, (wx, self.y + 8, 4, 4))
            pygame.draw.rect(surface, YELLOW, (wx, self.y + 14, 4, 4))


class Missile:
    def __init__(self, start_x, start_y, target_x, target_y, speed, color, is_enemy=True):
        self.x = start_x
        self.y = start_y
        self.target_x = target_x
        self.target_y = target_y
        self.speed = speed
        self.color = color
        self.is_enemy = is_enemy
        self.alive = True
        self.trail = []

        # Calculate direction
        dx = target_x - start_x
        dy = target_y - start_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0:
            self.vx = dx / dist * speed
            self.vy = dy / dist * speed
        else:
            self.vx = 0
            self.vy = speed

    def update(self):
        if not self.alive:
            return

        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 20:
            self.trail.pop(0)

        self.x += self.vx
        self.y += self.vy

        # Check if reached target
        dx = self.x - self.target_x
        dy = self.y - self.target_y
        if dx * dx + dy * dy < self.speed * self.speed:
            self.alive = False
            return True  # Signal that missile reached target

        # Off screen check
        if self.x < -50 or self.x > SCREEN_WIDTH + 50 or self.y < -50 or self.y > SCREEN_HEIGHT + 50:
            self.alive = False

        return False

    def draw(self, surface):
        if not self.alive:
            return
        # Draw trail
        for i, (tx, ty) in enumerate(self.trail):
            alpha = i / len(self.trail) if self.trail else 0
            size = max(1, int(3 * alpha))
            color = tuple(int(c * alpha) for c in self.color)
            pygame.draw.circle(surface, color, (tx, ty), size)

        # Draw missile head
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), 4)
        if self.is_enemy:
            pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), 2)


class Explosion:
    def __init__(self, x, y, radius=EXPLOSION_RADIUS, is_interceptor=True):
        self.x = x
        self.y = y
        self.max_radius = radius
        self.current_radius = 0
        self.life = 0
        self.max_life = EXPLOSION_DURATION if is_interceptor else 20
        self.alive = True
        self.is_interceptor = is_interceptor

    def update(self):
        self.life += 1
        if self.life >= self.max_life:
            self.alive = False
            return

        progress = self.life / self.max_life
        self.current_radius = self.max_radius * (1 - (1 - progress) ** 2)

    def draw(self, surface):
        if not self.alive:
            return

        progress = self.life / self.max_life
        alpha = int(255 * (1 - progress))

        if self.is_interceptor:
            # Outer ring
            color = (255, int(200 * (1 - progress)), int(50 * (1 - progress)))
            pygame.draw.circle(surface, color, (int(self.x), int(self.y)), int(self.current_radius), max(1, int(3 * (1 - progress))))
            # Inner glow
            if progress < 0.5:
                inner_progress = progress * 2
                inner_radius = int(self.current_radius * 0.6 * inner_progress)
                glow_color = (255, int(255 * (1 - inner_progress)), int(200 * (1 - inner_progress)))
                pygame.draw.circle(surface, glow_color, (int(self.x), int(self.y)), inner_radius)
        else:
            # Enemy explosion (when missile hits ground/city)
            color = (255, int(100 * (1 - progress)), 0)
            pygame.draw.circle(surface, color, (int(self.x), int(self.y)), int(self.current_radius), max(1, int(2 * (1 - progress))))

    def check_missile_collision(self, missile):
        """Check if a missile is within the explosion radius."""
        if not self.alive:
            return False
        dx = missile.x - self.x
        dy = missile.y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        return dist < self.current_radius


class Game:
    def __init__(self):
        self.score = 0
        self.wave = 0
        self.ammo = AMMO_PER_WAVE
        self.cities = []
        self.enemy_missiles = []
        self.interceptor_missiles = []
        self.explosions = []
        self.game_over = False
        self.wave_active = False
        self.wave_delay = 0
        self.missiles_this_wave = 0
        self.missiles_spawned = 0
        self.missiles_destroyed = 0
        self.target_pos = None  # For showing aim point
        self.create_cities()

    def create_cities(self):
        self.cities = []
        spacing = SCREEN_WIDTH // (NUM_CITIES + 1)
        for i in range(NUM_CITIES):
            x = spacing * (i + 1) - CITY_WIDTH // 2
            self.cities.append(City(x))

    def start_wave(self):
        self.wave += 1
        self.wave_active = True
        self.ammo = AMMO_PER_WAVE + self.wave * 8
        self.missiles_this_wave = 3 + self.wave * 2
        self.missiles_spawned = 0
        self.missiles_destroyed = 0
        self.wave_delay = 0

    def spawn_enemy_missile(self):
        # Launch from random position along top and sides
        side = random.choice(['top', 'left', 'right'])
        if side == 'top':
            start_x = random.randint(50, SCREEN_WIDTH - 50)
            start_y = -20
        elif side == 'left':
            start_x = -20
            start_y = random.randint(50, SCREEN_HEIGHT // 2)
        else:
            start_x = SCREEN_WIDTH + 20
            start_y = random.randint(50, SCREEN_HEIGHT // 2)

        # Target a random city or random ground position
        alive_cities = [c for c in self.cities if c.alive]
        if alive_cities and random.random() < 0.7:
            target = random.choice(alive_cities)
            target_x = target.x + target.width // 2 + random.randint(-10, 10)
            target_y = target.y + target.height
        else:
            target_x = random.randint(50, SCREEN_WIDTH - 50)
            target_y = GROUND_Y + random.randint(-10, 10)

        speed = ENEMY_MISSILE_SPEED_BASE + random.uniform(0, self.wave * 0.15)
        color = random.choice([RED, ORANGE, (255, 100, 100)])
        self.enemy_missiles.append(Missile(start_x, start_y, target_x, target_y, speed, color, is_enemy=True))
        self.missiles_spawned += 1

    def launch_interceptor(self, target_x, target_y):
        if self.ammo <= 0:
            return
        self.ammo -= 1

        # Launch from bottom-center (defense base)
        start_x = SCREEN_WIDTH // 2
        start_y = GROUND_Y

        self.interceptor_missiles.append(Missile(start_x, start_y, target_x, target_y, INTERCEPTOR_SPEED, YELLOW, is_enemy=False))

    def update(self):
        if self.game_over:
            return

        # Wave management
        if not self.wave_active:
            self.wave_delay += 1
            if self.wave_delay > 120:  # 2 second delay between waves
                self.start_wave()
            return

        # Spawn enemy missiles
        if self.missiles_spawned < self.missiles_this_wave:
            self.wave_delay += 1
            spawn_rate = max(15, 60 - self.wave * 3)
            if self.wave_delay >= spawn_rate:
                self.wave_delay = 0
                self.spawn_enemy_missile()
                # Sometimes spawn multiple at once in later waves
                if self.wave > 5 and random.random() < 0.2 + self.wave * 0.02:
                    self.spawn_enemy_missile()
                if self.wave > 10 and random.random() < 0.1 + self.wave * 0.01:
                    self.spawn_enemy_missile()

        # Update enemy missiles
        for missile in self.enemy_missiles[:]:
            reached_target = missile.update()
            if reached_target:
                # Create ground explosion
                self.explosions.append(Explosion(missile.x, missile.y, 30, is_interceptor=False))
                # Check if it hit a city
                for city in self.cities:
                    if city.alive:
                        cx = city.x + city.width // 2
                        cy = city.y + city.height // 2
                        dx = missile.x - cx
                        dy = missile.y - cy
                        if abs(dx) < city.width and abs(dy) < city.height:
                            city.alive = False
                            self.score = max(0, self.score - 50)
                self.enemy_missiles.remove(missile)

        # Update interceptor missiles
        for missile in self.interceptor_missiles[:]:
            reached_target = missile.update()
            if reached_target or not missile.alive:
                # Create explosion at target
                self.explosions.append(Explosion(missile.target_x, missile.target_y, EXPLOSION_RADIUS, is_interceptor=True))
                self.interceptor_missiles.remove(missile)

        # Update explosions and check for enemy missile kills
        for explosion in self.explosions[:]:
            explosion.update()
            if explosion.alive and explosion.is_interceptor:
                # Check if any enemy missile is caught in this explosion
                for missile in self.enemy_missiles[:]:
                    if explosion.check_missile_collision(missile):
                        missile.alive = False
                        self.enemy_missiles.remove(missile)
                        self.missiles_destroyed += 1
                        self.score += 10
                        # Small bonus explosion
                        self.explosions.append(Explosion(missile.x, missile.y, 15, is_interceptor=False))
            if not explosion.alive:
                self.explosions.remove(explosion)

        # Check wave completion
        if (self.missiles_spawned >= self.missiles_this_wave and
                len(self.enemy_missiles) == 0 and
                len(self.interceptor_missiles) == 0):
            self.wave_active = False
            self.wave_delay = 0
            # Bonus for surviving cities
            alive_count = sum(1 for c in self.cities if c.alive)
            self.score += alive_count * 25

        # Check game over
        alive_cities = sum(1 for c in self.cities if c.alive)
        if alive_cities == 0:
            self.game_over = True

    def draw(self, surface):
        surface.fill(BLACK)

        # Draw ground
        pygame.draw.rect(surface, DARK_GREEN, (0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y))
        pygame.draw.line(surface, GREEN, (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)

        # Draw cities
        for city in self.cities:
            city.draw(surface)

        # Draw defense base
        base_x = SCREEN_WIDTH // 2
        base_y = GROUND_Y
        pygame.draw.rect(surface, GREY, (base_x - 15, base_y - 10, 30, 10))
        pygame.draw.rect(surface, WHITE, (base_x - 10, base_y - 15, 20, 5))

        # Draw enemy missiles
        for missile in self.enemy_missiles:
            missile.draw(surface)

        # Draw interceptor missiles
        for missile in self.interceptor_missiles:
            missile.draw(surface)

        # Draw explosions
        for explosion in self.explosions:
            explosion.draw(surface)

        # Draw aim point
        if self.target_pos:
            mx, my = self.target_pos
            pygame.draw.circle(surface, YELLOW, (mx, my), 5, 1)
            pygame.draw.line(surface, YELLOW, (mx - 10, my), (mx + 10, my), 1)
            pygame.draw.line(surface, YELLOW, (mx, my - 10), (mx, my + 10), 1)

        # Draw HUD
        # Score
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        surface.blit(score_text, (10, 10))

        # Wave
        wave_text = font.render(f"Wave: {self.wave}", True, WHITE)
        surface.blit(wave_text, (SCREEN_WIDTH // 2 - 40, 10))

        # Ammo
        ammo_text = font.render(f"Ammo: {self.ammo}", True, YELLOW)
        surface.blit(ammo_text, (SCREEN_WIDTH - 150, 10))

        # Cities alive
        alive = sum(1 for c in self.cities if c.alive)
        cities_text = font.render(f"Cities: {alive}/{NUM_CITIES}", True, BLUE)
        surface.blit(cities_text, (SCREEN_WIDTH - 150, 40))

        # Wave info
        if self.wave_active:
            remaining = self.missiles_this_wave - self.missiles_spawned
            info_text = small_font.render(f"Missiles remaining: {remaining}", True, WHITE)
            surface.blit(info_text, (10, 40))
        elif not self.game_over:
            if self.wave == 0:
                info_text = font.render("Click to launch interceptor missiles. Defend your cities!", True, WHITE)
                surface.blit(info_text, (SCREEN_WIDTH // 2 - 250, SCREEN_HEIGHT // 2))
            else:
                info_text = font.render("Wave incoming...", True, RED)
                surface.blit(info_text, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2))

        # Game over screen
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            surface.blit(overlay, (0, 0))

            game_over_text = big_font.render("GAME OVER", True, RED)
            surface.blit(game_over_text, (SCREEN_WIDTH // 2 - 180, SCREEN_HEIGHT // 2 - 80))

            final_score = font.render(f"Final Score: {self.score}", True, WHITE)
            surface.blit(final_score, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2))

            waves_survived = font.render(f"Waves Survived: {self.wave}", True, WHITE)
            surface.blit(waves_survived, (SCREEN_WIDTH // 2 - 110, SCREEN_HEIGHT // 2 + 40))

            restart_text = font.render("Press SPACE to restart", True, WHITE)
            surface.blit(restart_text, (SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT // 2 + 100))

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self.game_over:
                        mx, my = event.pos
                        # Don't launch if clicking on HUD area
                        if my > 60:
                            self.launch_interceptor(mx, my)
                elif event.type == pygame.MOUSEMOTION:
                    self.target_pos = event.pos
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.game_over:
                        self.__init__()
                        return self.run()

            self.update()
            self.draw(screen)
            pygame.display.flip()

        pygame.quit()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()