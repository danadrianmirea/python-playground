# frogger.py - Classic Frogger game

import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 650
GRID_SIZE = 40
ROWS = 15  # 600 / 40 = 15 rows for the play area
COLS = 15  # 600 / 40 = 15 columns

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Frogger")

# Game constants
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 100, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
DARK_BLUE = (0, 50, 150)
BROWN = (139, 69, 19)
GRAY = (100, 100, 100)
LIGHT_GRAY = (180, 180, 180)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Fonts
font = pygame.font.Font(None, 36)
score_font = pygame.font.Font(None, 48)
game_over_font = pygame.font.Font(None, 60)
title_font = pygame.font.Font(None, 22)

# Clock
clock = pygame.time.Clock()

# Lane definitions
# Row 0: Goal (top)
# Row 1-2: Grass (safe)
# Row 3-7: River (logs/turtles)
# Row 8: Grass (safe)
# Row 9-13: Road (cars/trucks)
# Row 14: Grass (start)

LANE_TYPES = {
    0: 'goal',
    1: 'grass',
    2: 'grass',
    3: 'river',
    4: 'river',
    5: 'river',
    6: 'river',
    7: 'river',
    8: 'grass',
    9: 'road',
    10: 'road',
    11: 'road',
    12: 'road',
    13: 'road',
    14: 'grass',
}


class Log:
    """A floating log in the river."""

    def __init__(self, col, row, length, speed, direction):
        self.col = col
        self.row = row
        self.length = length  # in tiles
        self.speed = speed
        self.direction = direction  # 1 = right, -1 = left
        self.width = length * GRID_SIZE
        self.x = col * GRID_SIZE
        self.y = row * GRID_SIZE

    def update(self):
        self.x += self.speed * self.direction
        self.col = int(self.x // GRID_SIZE)

        # Wrap around
        if self.direction == 1 and self.x > SCREEN_WIDTH:
            self.x = -self.width
        elif self.direction == -1 and self.x + self.width < 0:
            self.x = SCREEN_WIDTH

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, GRID_SIZE)

    def draw(self, surface):
        rect = self.get_rect()
        # Log body
        pygame.draw.rect(surface, BROWN, rect, border_radius=8)
        # Wood grain lines
        for i in range(self.length):
            gx = self.x + i * GRID_SIZE + 5
            pygame.draw.line(surface, (100, 50, 0), (gx, self.y + 5), (gx, self.y + GRID_SIZE - 5), 2)
            pygame.draw.line(surface, (100, 50, 0), (gx + 10, self.y + 5), (gx + 10, self.y + GRID_SIZE - 5), 2)
        # Outline
        pygame.draw.rect(surface, (80, 40, 0), rect, 2, border_radius=8)


class Car:
    """A car on the road."""

    def __init__(self, col, row, speed, direction, color, width=1):
        self.col = col
        self.row = row
        self.speed = speed
        self.direction = direction  # 1 = right, -1 = left
        self.color = color
        self.width_tiles = width
        self.width = width * GRID_SIZE
        self.height = GRID_SIZE - 6
        self.x = col * GRID_SIZE
        self.y = row * GRID_SIZE + 3

    def update(self):
        self.x += self.speed * self.direction
        self.col = int(self.x // GRID_SIZE)

        # Wrap around
        if self.direction == 1 and self.x > SCREEN_WIDTH:
            self.x = -self.width
        elif self.direction == -1 and self.x + self.width < 0:
            self.x = SCREEN_WIDTH

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, surface):
        rect = self.get_rect()
        # Car body
        pygame.draw.rect(surface, self.color, rect, border_radius=6)

        # Windshield
        if self.direction == 1:  # moving right
            wx = rect.right - 12
        else:  # moving left
            wx = rect.left + 2
        windshield = pygame.Rect(wx, rect.top + 6, 10, rect.height - 12)
        pygame.draw.rect(surface, (200, 230, 255), windshield, border_radius=3)

        # Wheels
        wheel_color = (30, 30, 30)
        pygame.draw.circle(surface, wheel_color, (rect.left + 6, rect.top - 1), 4)
        pygame.draw.circle(surface, wheel_color, (rect.right - 6, rect.top - 1), 4)
        pygame.draw.circle(surface, wheel_color, (rect.left + 6, rect.bottom + 1), 4)
        pygame.draw.circle(surface, wheel_color, (rect.right - 6, rect.bottom + 1), 4)

        # Headlights
        if self.direction == 1:
            pygame.draw.circle(surface, YELLOW, (rect.right - 3, rect.top + 5), 3)
            pygame.draw.circle(surface, YELLOW, (rect.right - 3, rect.bottom - 5), 3)
        else:
            pygame.draw.circle(surface, RED, (rect.left + 3, rect.top + 5), 3)
            pygame.draw.circle(surface, RED, (rect.left + 3, rect.bottom - 5), 3)


class Frog:
    """The player-controlled frog."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.col = 7
        self.row = 14  # Starting row (bottom)
        self.x = self.col * GRID_SIZE
        self.y = self.row * GRID_SIZE
        self.target_x = self.x
        self.target_y = self.y
        self.moving = False
        self.move_timer = 0
        self.move_duration = 8  # frames for hop animation
        self.alive = True
        self.on_log = None
        self.facing = 'up'

    def hop(self, dc, dr):
        """Start a hop in the given direction."""
        if self.moving or not self.alive:
            return

        new_col = self.col + dc
        new_row = self.row + dr

        # Boundary check
        if new_col < 0 or new_col >= COLS:
            return
        if new_row < 0 or new_row >= ROWS:
            return

        self.target_x = new_col * GRID_SIZE
        self.target_y = new_row * GRID_SIZE
        self.col = new_col
        self.row = new_row
        self.moving = True
        self.move_timer = 0

        if dc == 1:
            self.facing = 'right'
        elif dc == -1:
            self.facing = 'left'
        elif dr == -1:
            self.facing = 'up'
        elif dr == 1:
            self.facing = 'down'

    def update(self):
        if not self.alive:
            return

        if self.moving:
            self.move_timer += 1
            t = self.move_timer / self.move_duration
            # Ease out quad
            t = t * (2 - t)
            self.x = self.x + (self.target_x - self.x) * 0.3
            self.y = self.y + (self.target_y - self.y) * 0.3

            if self.move_timer >= self.move_duration:
                self.x = self.target_x
                self.y = self.target_y
                self.moving = False
                self.move_timer = 0

        # If on a log in the river, move with it
        if self.on_log and not self.moving:
            self.x += self.on_log.speed * self.on_log.direction
            self.col = int((self.x + GRID_SIZE // 2) // GRID_SIZE)

            # Check if frog fell off the log (off screen sides)
            if self.x < -GRID_SIZE or self.x > SCREEN_WIDTH:
                self.die()

    def die(self):
        self.alive = False

    def get_rect(self):
        return pygame.Rect(self.x + 4, self.y + 4, GRID_SIZE - 8, GRID_SIZE - 8)

    def draw(self, surface):
        if not self.alive:
            return

        cx = self.x + GRID_SIZE // 2
        cy = self.y + GRID_SIZE // 2

        # Body (green oval)
        body_rect = pygame.Rect(self.x + 4, self.y + 8, GRID_SIZE - 8, GRID_SIZE - 14)
        pygame.draw.ellipse(surface, GREEN, body_rect)

        # Darker green belly
        belly_rect = pygame.Rect(self.x + 8, self.y + 12, GRID_SIZE - 16, GRID_SIZE - 20)
        pygame.draw.ellipse(surface, DARK_GREEN, belly_rect)

        # Eyes (two white circles with black pupils)
        eye_offset_x = 6
        eye_offset_y = 4
        if self.facing == 'left':
            eye_offset_x = 4
        elif self.facing == 'right':
            eye_offset_x = 8

        # Left eye
        pygame.draw.circle(surface, WHITE, (cx - eye_offset_x, cy - 8), 5)
        pygame.draw.circle(surface, BLACK, (cx - eye_offset_x, cy - 8), 3)
        # Right eye
        pygame.draw.circle(surface, WHITE, (cx + eye_offset_x, cy - 8), 5)
        pygame.draw.circle(surface, BLACK, (cx + eye_offset_x, cy - 8), 3)

        # Mouth
        if self.facing == 'up':
            pygame.draw.arc(surface, BLACK, (cx - 5, cy + 2, 10, 6), 0, 3.14, 2)
        elif self.facing == 'down':
            pygame.draw.arc(surface, BLACK, (cx - 5, cy - 2, 10, 6), 3.14, 6.28, 2)
        elif self.facing == 'left':
            pygame.draw.arc(surface, BLACK, (cx - 8, cy - 2, 10, 6), 3.14, 6.28, 2)
        else:
            pygame.draw.arc(surface, BLACK, (cx - 2, cy - 2, 10, 6), 0, 3.14, 2)

        # Back legs
        leg_color = (0, 150, 0)
        if self.facing == 'up' or self.facing == 'down':
            pygame.draw.ellipse(surface, leg_color, (self.x + 4, self.y + GRID_SIZE - 12, 10, 8))
            pygame.draw.ellipse(surface, leg_color, (self.x + GRID_SIZE - 14, self.y + GRID_SIZE - 12, 10, 8))
        else:
            pygame.draw.ellipse(surface, leg_color, (self.x + 2, self.y + GRID_SIZE - 12, 10, 8))
            pygame.draw.ellipse(surface, leg_color, (self.x + GRID_SIZE - 12, self.y + GRID_SIZE - 12, 10, 8))


class Game:
    """Main game class."""

    def __init__(self):
        self.frog = Frog()
        self.cars = []
        self.logs = []
        self.score = 0
        self.lives = 5
        self.level = 1
        self.game_over = False
        self.won = False
        self.time_left = 3000  # frames (50 seconds at 60fps)
        self.spawn_timer = 0
        self.animation_timer = 0
        self.level_transition_timer = 0

        self._spawn_vehicles()
        self._spawn_logs()

    def _spawn_vehicles(self):
        """Spawn cars and trucks on road lanes."""
        self.cars = []

        # Gentle speed multiplier - increases slowly with level
        speed_mult = 1.0 + (self.level - 1) * 0.08

        # Road lane configurations: (row, speed, direction, color, width, count)
        road_configs = [
            (9, 2, 1, RED, 1, 3),        # red cars right
            (10, 3, -1, BLUE, 1, 3),      # blue cars left
            (11, 2, 1, ORANGE, 2, 2),     # orange trucks right (2 tiles wide)
            (12, 4, -1, YELLOW, 1, 3),    # yellow cars left (fast)
            (13, 1, 1, PURPLE, 1, 3),     # purple cars right (slow)
        ]

        # Add extra cars sparingly: +1 at level 5, +2 at level 10, +3 at level 15
        extra_count = max(0, (self.level - 4) // 5)
        extra_count = min(extra_count, 3)

        # On higher levels, add some speed variation within lanes
        speed_variation = 0
        if self.level >= 4:
            speed_variation = 1  # Some cars get +1 speed

        for row, speed, direction, color, width, count in road_configs:
            count += extra_count
            spacing = COLS // count
            for i in range(count):
                col = i * spacing + random.randint(0, spacing - 2)
                # Add slight speed variation to some cars for unpredictability
                car_speed = int(speed * speed_mult)
                if speed_variation > 0 and random.random() < 0.3:
                    car_speed += random.choice([-1, 1])
                car_speed = max(1, car_speed)
                self.cars.append(Car(col, row, car_speed, direction, color, width))

    def _spawn_logs(self):
        """Spawn logs on river lanes."""
        self.logs = []

        # Gentle speed multiplier for logs
        speed_mult = 1.0 + (self.level - 1) * 0.06

        # River lane configurations: (row, speed, direction, length, count)
        river_configs = [
            (3, 1, 1, 3, 3),    # long logs right
            (4, 2, -1, 2, 4),   # medium logs left
            (5, 1, 1, 4, 2),    # very long logs right
            (6, 3, -1, 2, 4),   # medium logs left (fast)
            (7, 1, 1, 3, 3),    # long logs right
        ]

        # Reduce log count very sparingly: -1 at level 8, -2 at level 16
        count_reduction = max(0, (self.level - 7) // 8)
        count_reduction = min(count_reduction, 2)

        # On higher levels, occasionally use shorter logs to create gaps
        length_reduction = 0
        if self.level >= 6:
            length_reduction = 1  # Some logs are 1 tile shorter

        for row, speed, direction, length, count in river_configs:
            count = max(count - count_reduction, 2)
            spacing = COLS // count
            for i in range(count):
                col = i * spacing + random.randint(0, spacing - length)
                log_length = length
                if length_reduction > 0 and random.random() < 0.3:
                    log_length = max(length - 1, 2)
                self.logs.append(Log(col, row, log_length, int(speed * speed_mult), direction))

    def reset_frog(self):
        """Reset frog to starting position."""
        self.frog = Frog()
        self.time_left = 3000

    def handle_event(self, event):
        """Handle keyboard input."""
        if event.type == pygame.KEYDOWN:
            if self.game_over or self.won:
                if event.key == pygame.K_SPACE:
                    self.__init__()
                return

            if not self.frog.alive:
                if event.key == pygame.K_SPACE:
                    self.reset_frog()
                return

            if event.key == pygame.K_UP or event.key == pygame.K_w:
                self.frog.hop(0, -1)
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                self.frog.hop(0, 1)
            elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                self.frog.hop(-1, 0)
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                self.frog.hop(1, 0)

    def update(self):
        """Update game state."""
        if self.game_over or self.won:
            return

        # Level transition countdown
        if self.level_transition_timer > 0:
            self.level_transition_timer -= 1
            return

        # Update timer
        self.time_left -= 1
        if self.time_left <= 0:
            self.frog.die()

        # Update cars
        for car in self.cars:
            car.update()

        # Update logs
        for log in self.logs:
            log.update()

        # Update frog
        self.frog.update()

        if not self.frog.alive:
            return

        # Check if frog is on a log
        self.frog.on_log = None
        if self.frog.row in [3, 4, 5, 6, 7]:  # River rows
            frog_rect = self.frog.get_rect()
            on_log = False
            for log in self.logs:
                if log.row == self.frog.row and log.get_rect().colliderect(frog_rect):
                    self.frog.on_log = log
                    on_log = True
                    break

            if not on_log:
                # Frog fell in the water
                self.frog.die()
                self.lives -= 1
                if self.lives <= 0:
                    self.game_over = True
                return

        # Check car collisions
        if 9 <= self.frog.row <= 13:  # Road rows
            frog_rect = self.frog.get_rect()
            for car in self.cars:
                if car.row == self.frog.row and car.get_rect().colliderect(frog_rect):
                    self.frog.die()
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True
                    return

        # Check if frog reached the top (advance to next level)
        if self.frog.row == 0:
            self.score += 100 + (self.time_left // 10)
            self.level += 1
            self._spawn_vehicles()
            self._spawn_logs()
            self.reset_frog()
            self.level_transition_timer = 60  # Show level transition for 1 second

    def draw(self, surface):
        """Draw the entire game."""
        surface.fill(BLACK)

        # Draw lanes
        for row in range(ROWS):
            y = row * GRID_SIZE
            lane_type = LANE_TYPES.get(row, 'grass')

            if lane_type == 'grass':
                if row == 0:
                    # Goal row - draw water background with goal spots
                    pygame.draw.rect(surface, DARK_BLUE, (0, y, SCREEN_WIDTH, GRID_SIZE))
                    # Draw goal spots (lily pads)
                    for gc in range(COLS):
                        gx = gc * GRID_SIZE
                        if gc % 3 == 0:  # Every 3rd column is a goal
                            # Draw lily pad
                            pygame.draw.ellipse(surface, (0, 150, 0), (gx + 6, y + 10, GRID_SIZE - 12, GRID_SIZE - 16))
                            pygame.draw.ellipse(surface, (0, 100, 0), (gx + 8, y + 12, GRID_SIZE - 16, GRID_SIZE - 20))
                            # Flower
                            pygame.draw.circle(surface, (255, 100, 100), (gx + GRID_SIZE // 2, y + 8), 4)
                            pygame.draw.circle(surface, YELLOW, (gx + GRID_SIZE // 2, y + 8), 2)
                else:
                    # Regular grass
                    shade = DARK_GREEN if (row + (row % 2)) % 2 == 0 else GREEN
                    pygame.draw.rect(surface, shade, (0, y, SCREEN_WIDTH, GRID_SIZE))
                    # Grass texture dots
                    for _ in range(5):
                        gx = random.randint(0, SCREEN_WIDTH - 1)
                        gy = y + random.randint(2, GRID_SIZE - 2)
                        pygame.draw.circle(surface, (0, random.randint(120, 180), 0), (gx, gy), 2)

            elif lane_type == 'river':
                # Water
                shade = DARK_BLUE if (row % 2) == 0 else BLUE
                pygame.draw.rect(surface, shade, (0, y, SCREEN_WIDTH, GRID_SIZE))
                # Water ripple lines
                for wx in range(0, SCREEN_WIDTH, 30):
                    offset = (self.animation_timer + wx * 2) % 60
                    ripple_y = y + 10 + (offset / 60) * 20
                    pygame.draw.line(surface, (100, 150, 255), (wx, ripple_y), (wx + 15, ripple_y + 3), 1)

            elif lane_type == 'road':
                # Road
                shade = (50, 50, 50) if (row % 2) == 0 else (60, 60, 60)
                pygame.draw.rect(surface, shade, (0, y, SCREEN_WIDTH, GRID_SIZE))
                # Road markings (dashed center line)
                for lx in range(0, SCREEN_WIDTH, 40):
                    pygame.draw.rect(surface, YELLOW, (lx, y + GRID_SIZE // 2 - 1, 20, 2))

        # Draw logs
        for log in self.logs:
            log.draw(surface)

        # Draw cars
        for car in self.cars:
            car.draw(surface)

        # Draw frog
        self.frog.draw(surface)

        # Draw death overlay if frog is dead but game isn't over
        if not self.frog.alive and not self.game_over and not self.won:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            surface.blit(overlay, (0, 0))

            death_text = font.render("You died! Press SPACE to continue", True, WHITE)
            surface.blit(death_text, (SCREEN_WIDTH // 2 - death_text.get_width() // 2, SCREEN_HEIGHT // 2 - 20))

        # Draw controls hint at top
        controls_text = title_font.render("Arrow Keys / WASD - Move | SPACE - Continue", True, WHITE)
        surface.blit(controls_text, (SCREEN_WIDTH // 2 - controls_text.get_width() // 2, 20))

        # Draw UI
        # Score
        score_text = score_font.render(f"Score: {self.score}", True, WHITE)
        surface.blit(score_text, (10, SCREEN_HEIGHT - 45))

        # Lives
        lives_text = font.render(f"Lives: {self.lives}", True, WHITE)
        surface.blit(lives_text, (SCREEN_WIDTH - 120, SCREEN_HEIGHT - 45))

        # Level
        level_text = font.render(f"Level: {self.level}", True, WHITE)
        surface.blit(level_text, (SCREEN_WIDTH // 2 - 40, SCREEN_HEIGHT - 45))

        # Timer bar
        timer_width = 200
        timer_height = 10
        timer_x = SCREEN_WIDTH // 2 - timer_width // 2
        timer_y = 5
        timer_pct = self.time_left / 3000
        pygame.draw.rect(surface, GRAY, (timer_x, timer_y, timer_width, timer_height))
        timer_color = RED if timer_pct < 0.3 else YELLOW if timer_pct < 0.6 else GREEN
        pygame.draw.rect(surface, timer_color, (timer_x, timer_y, int(timer_width * timer_pct), timer_height))

        # Level transition
        if self.level_transition_timer > 0:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            surface.blit(overlay, (0, 0))

            level_up_text = game_over_font.render(f"LEVEL {self.level}!", True, YELLOW)
            surface.blit(level_up_text, (SCREEN_WIDTH // 2 - level_up_text.get_width() // 2, SCREEN_HEIGHT // 2 - 40))

            level_info_text = font.render("Get ready...", True, WHITE)
            surface.blit(level_info_text, (SCREEN_WIDTH // 2 - level_info_text.get_width() // 2, SCREEN_HEIGHT // 2 + 20))

        # Game over / win
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            surface.blit(overlay, (0, 0))

            text = game_over_font.render("GAME OVER", True, RED)
            surface.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - 60))

            restart_text = font.render("Press SPACE to restart", True, WHITE)
            surface.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 10))

        if self.won:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            surface.blit(overlay, (0, 0))

            text = game_over_font.render("YOU WIN!", True, YELLOW)
            surface.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - 60))

            restart_text = font.render("Press SPACE to play again", True, WHITE)
            surface.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 10))


def main():
    """Main game loop."""
    global game
    game = Game()
    running = True

    while running:
        dt = clock.tick(FPS)
        game.animation_timer += 1

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            game.handle_event(event)

        # Update
        game.update()

        # Draw
        game.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()