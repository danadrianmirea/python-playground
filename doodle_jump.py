# doodle_jump.py - Doodle Jump clone

import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 700
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Doodle Jump")

# Game constants
GRAVITY = 0.4
JUMP_STRENGTH = -11
MOVE_SPEED = 5
PLATFORM_WIDTH = 60
PLATFORM_HEIGHT = 14
PLAYER_WIDTH = 30
PLAYER_HEIGHT = 40
FPS = 60
SCROLL_THRESHOLD = SCREEN_HEIGHT // 3
RISING_SPEED = 0.3  # How fast the death zone rises
RISING_ACCEL = 0.0005  # How much the rising speed increases per frame
MAX_RISING_SPEED = 1.5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (135, 206, 235)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
RED = (255, 50, 50)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
PURPLE = (180, 50, 200)
GRAY = (180, 180, 180)
BROWN = (139, 69, 19)

# Fonts
font_large = pygame.font.SysFont("Arial", 48, bold=True)
font_medium = pygame.font.SysFont("Arial", 28)
font_small = pygame.font.SysFont("Arial", 20)

clock = pygame.time.Clock()


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.vel_y = 0
        self.vel_x = 0
        self.on_platform = False
        # Animation
        self.eye_offset = 0
        self.eye_dir = 1

    def update(self):
        self.vel_y += GRAVITY
        self.x += self.vel_x
        self.y += self.vel_y

        # Wrap around screen horizontally
        if self.x + self.width < 0:
            self.x = SCREEN_WIDTH
        elif self.x > SCREEN_WIDTH:
            self.x = -self.width

        # Eye animation
        self.eye_offset += 0.05 * self.eye_dir
        if abs(self.eye_offset) > 2:
            self.eye_dir *= -1

    def jump(self):
        self.vel_y = JUMP_STRENGTH
        self.on_platform = False

    def draw(self, surface):
        # Body
        body_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(surface, ORANGE, body_rect)
        pygame.draw.rect(surface, (200, 120, 0), body_rect, 2)

        # Head (slightly wider at top)
        head_rect = pygame.Rect(self.x - 3, self.y - 5, self.width + 6, 20)
        pygame.draw.ellipse(surface, ORANGE, head_rect)
        pygame.draw.ellipse(surface, (200, 120, 0), head_rect, 2)

        # Eyes
        eye_y = self.y - 2
        pygame.draw.circle(surface, WHITE, (self.x + 8, eye_y), 5)
        pygame.draw.circle(surface, WHITE, (self.x + self.width - 8, eye_y), 5)
        pygame.draw.circle(
            surface, BLACK, (self.x + 8 + int(self.eye_offset), eye_y), 2
        )
        pygame.draw.circle(
            surface,
            BLACK,
            (self.x + self.width - 8 + int(self.eye_offset), eye_y),
            2,
        )

        # Feet
        foot_y = self.y + self.height
        pygame.draw.ellipse(surface, BLACK, (self.x - 2, foot_y - 4, 14, 8))
        pygame.draw.ellipse(
            surface, BLACK, (self.x + self.width - 12, foot_y - 4, 14, 8)
        )

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Platform:
    def __init__(self, x, y, platform_type="normal"):
        self.x = x
        self.y = y
        self.width = PLATFORM_WIDTH
        self.height = PLATFORM_HEIGHT
        self.type = platform_type  # "normal", "moving", "breaking", "disappearing"
        self.broken = False
        self.break_timer = 0
        self.move_dir = 1
        self.move_speed = 2
        self.move_range = 80
        self.original_x = x

    def update(self):
        if self.type == "moving":
            self.x += self.move_speed * self.move_dir
            if abs(self.x - self.original_x) > self.move_range:
                self.move_dir *= -1

        if self.type == "breaking" and self.broken:
            self.break_timer += 1
            if self.break_timer > 20:
                self.y += 10

    def draw(self, surface):
        if self.type == "normal":
            color = GREEN
            border = DARK_GREEN
        elif self.type == "moving":
            color = BLUE
            border = (0, 100, 200)
        elif self.type == "breaking":
            if self.broken:
                color = GRAY
                border = (100, 100, 100)
            else:
                color = BROWN
                border = (100, 50, 0)
        elif self.type == "disappearing":
            color = PURPLE
            border = (120, 0, 150)
        else:
            color = GREEN
            border = DARK_GREEN

        # Platform body
        rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, border, rect, 2)

        # Platform top highlight
        pygame.draw.line(
            surface, WHITE, (self.x + 2, self.y + 1), (self.x + self.width - 2, self.y + 1), 2
        )

        # Visual indicator for special platforms
        if self.type == "moving":
            arrow_x = self.x + self.width // 2
            pygame.draw.polygon(
                surface,
                WHITE,
                [
                    (arrow_x, self.y + 3),
                    (arrow_x - 5, self.y + self.height - 3),
                    (arrow_x + 5, self.y + self.height - 3),
                ],
            )
        elif self.type == "breaking":
            # Crack marks
            cx = self.x + self.width // 2
            cy = self.y + self.height // 2
            pygame.draw.line(surface, BLACK, (cx - 5, cy - 3), (cx + 3, cy + 3), 2)
            pygame.draw.line(surface, BLACK, (cx + 3, cy + 3), (cx + 8, cy - 2), 2)
        elif self.type == "disappearing":
            # Dots pattern
            for dx in range(10, self.width - 10, 15):
                pygame.draw.circle(surface, WHITE, (self.x + dx, self.y + self.height // 2), 2)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


def generate_platforms(count, start_y):
    """Generate a set of platforms."""
    platforms = []
    y = start_y

    for i in range(count):
        # Random x position
        x = random.randint(0, SCREEN_WIDTH - PLATFORM_WIDTH)

        # Random platform type (weighted)
        rand = random.random()
        if i == 0:
            ptype = "normal"
        elif rand < 0.6:
            ptype = "normal"
        elif rand < 0.75:
            ptype = "moving"
        elif rand < 0.9:
            ptype = "breaking"
        else:
            ptype = "disappearing"

        platforms.append(Platform(x, y, ptype))
        y -= random.randint(60, 90)

    return platforms


def draw_background(surface, scroll_y):
    """Draw scrolling background with gradient."""
    # Sky gradient
    for i in range(SCREEN_HEIGHT):
        r = int(135 + (i / SCREEN_HEIGHT) * 50)
        g = int(206 - (i / SCREEN_HEIGHT) * 30)
        b = int(235)
        pygame.draw.line(surface, (r, g, b), (0, i), (SCREEN_WIDTH, i))

    # Stars (parallax)
    star_seed = 42
    random.seed(star_seed)
    for _ in range(30):
        sx = random.randint(0, SCREEN_WIDTH)
        sy = (random.randint(0, SCREEN_HEIGHT) + scroll_y * 0.1) % SCREEN_HEIGHT
        brightness = random.randint(150, 255)
        pygame.draw.circle(surface, (brightness, brightness, brightness), (sx, int(sy)), 1)
    random.seed()


def show_score(surface, score, high_score):
    """Display score and high score."""
    score_text = font_medium.render(f"Score: {score}", True, BLACK)
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 30))
    surface.blit(score_text, score_rect)

    hs_text = font_small.render(f"Best: {high_score}", True, (80, 80, 80))
    hs_rect = hs_text.get_rect(center=(SCREEN_WIDTH // 2, 60))
    surface.blit(hs_text, hs_rect)


def show_game_over(surface, score, high_score):
    """Display game over screen."""
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill(BLACK)
    surface.blit(overlay, (0, 0))

    go_text = font_large.render("GAME OVER", True, RED)
    go_shadow = font_large.render("GAME OVER", True, (100, 0, 0))
    go_rect = go_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80))
    surface.blit(go_shadow, (go_rect.x + 3, go_rect.y + 3))
    surface.blit(go_text, go_rect)

    score_text = font_medium.render(f"Score: {score}", True, WHITE)
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
    surface.blit(score_text, score_rect)

    hs_text = font_medium.render(f"Best: {high_score}", True, YELLOW)
    hs_rect = hs_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
    surface.blit(hs_text, hs_rect)

    restart_text = font_small.render("Press SPACE or Click to Restart", True, WHITE)
    restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
    surface.blit(restart_text, restart_rect)

    quit_text = font_small.render("Press ESC to Quit", True, GRAY)
    quit_rect = quit_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 110))
    surface.blit(quit_text, quit_rect)


def show_start_screen(surface):
    """Display start screen."""
    draw_background(surface, 0)

    title_text = font_large.render("Doodle Jump", True, ORANGE)
    title_shadow = font_large.render("Doodle Jump", True, (150, 80, 0))
    title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 150))
    surface.blit(title_shadow, (title_rect.x + 3, title_rect.y + 3))
    surface.blit(title_text, title_rect)

    # Draw a sample platform
    sample = Platform(SCREEN_WIDTH // 2 - 30, 250, "normal")
    sample.draw(surface)

    # Draw sample player
    sample_player = Player(SCREEN_WIDTH // 2 - 15, 210)
    sample_player.draw(surface)

    inst1 = font_medium.render("Press SPACE or Click to Jump", True, BLACK)
    inst1_rect = inst1.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
    surface.blit(inst1, inst1_rect)

    inst2 = font_small.render("Land on platforms to go higher!", True, (80, 80, 80))
    inst2_rect = inst2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 120))
    surface.blit(inst2, inst2_rect)

    # Legend
    legend_y = SCREEN_HEIGHT // 2 + 170
    legend_items = [
        (GREEN, "Normal"),
        (BLUE, "Moving"),
        (BROWN, "Breaking"),
        (PURPLE, "Disappearing"),
    ]
    for i, (color, label) in enumerate(legend_items):
        lx = 60 + i * 90
        pygame.draw.rect(surface, color, (lx, legend_y, 20, 10))
        pygame.draw.rect(surface, BLACK, (lx, legend_y, 20, 10), 1)
        label_text = font_small.render(label, True, BLACK)
        surface.blit(label_text, (lx + 25, legend_y - 3))

    pygame.display.flip()


def reset_game():
    """Reset game state."""
    player = Player(SCREEN_WIDTH // 2 - PLAYER_WIDTH // 2, SCREEN_HEIGHT // 2)
    platforms = generate_platforms(12, SCREEN_HEIGHT - 50)
    score = 0
    scroll_y = 0
    return player, platforms, score, scroll_y


def draw_rising_zone(surface, rising_y):
    """Draw the rising death zone at the bottom."""
    if rising_y >= SCREEN_HEIGHT:
        return

    # Red danger zone
    zone_height = int(SCREEN_HEIGHT - rising_y)
    zone_rect = pygame.Rect(0, int(rising_y), SCREEN_WIDTH, zone_height)
    danger_surf = pygame.Surface((SCREEN_WIDTH, zone_height))
    for i in range(zone_height):
        alpha = min(200, 80 + int((i / zone_height) * 120))
        color = (180, 0, 0)
        danger_surf.set_at((0, i), color)
        pygame.draw.line(danger_surf, color, (0, i), (SCREEN_WIDTH, i))
    danger_surf.set_alpha(160)
    surface.blit(danger_surf, (0, rising_y))

    # Top edge line
    pygame.draw.line(surface, BLUE, (0, rising_y), (SCREEN_WIDTH, rising_y), 3)

    # Warning label
    #warn_text = font_small.render("RISING!", True, RED)
    #surface.blit(warn_text, (SCREEN_WIDTH // 2 - 30, rising_y + 10))


def check_collision(player, platforms):
    """Check if player lands on any platform."""
    player_rect = player.get_rect()

    for platform in platforms:
        if platform.type == "disappearing" and platform.broken:
            continue

        plat_rect = platform.get_rect()

        # Only check if player is falling and above the platform
        if player.vel_y > 0 and player_rect.colliderect(plat_rect):
            # Check if player's bottom is near the platform top
            player_bottom = player.y + player.height
            platform_top = platform.y

            if player_bottom - player.vel_y <= platform_top + 10:
                player.y = platform.y - player.height
                player.vel_y = 0
                player.on_platform = True

                # Handle special platform types
                if platform.type == "breaking" and not platform.broken:
                    platform.broken = True
                elif platform.type == "disappearing" and not platform.broken:
                    platform.broken = True

                return True

    return False


def main():
    """Main game loop."""
    player, platforms, score, scroll_y = reset_game()
    high_score = 0
    game_state = "start"
    running = True
    rising_y = SCREEN_HEIGHT  # Top of the rising death zone (starts off-screen)
    rising_speed = RISING_SPEED

    while running:
        clock.tick(FPS)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_SPACE:
                    if game_state == "start":
                        game_state = "playing"
                        player, platforms, score, scroll_y = reset_game()
                    elif game_state == "playing":
                        player.jump()
                    elif game_state == "game_over":
                        game_state = "playing"
                        player, platforms, score, scroll_y = reset_game()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_state == "start":
                    game_state = "playing"
                    player, platforms, score, scroll_y = reset_game()
                elif game_state == "game_over":
                    game_state = "playing"
                    player, platforms, score, scroll_y = reset_game()

        # Update
        if game_state == "playing":
            # Horizontal movement (auto-move or keyboard)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                player.vel_x = -MOVE_SPEED
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                player.vel_x = MOVE_SPEED
            else:
                player.vel_x = 0

            player.update()

            # Check platform collisions
            check_collision(player, platforms)

            # Scrolling
            if player.y < SCROLL_THRESHOLD:
                diff = SCROLL_THRESHOLD - player.y
                player.y = SCROLL_THRESHOLD
                scroll_y += diff

                # Move platforms and rising zone down
                for platform in platforms:
                    platform.y += diff
                rising_y += diff

                # Remove off-screen platforms and add new ones
                platforms = [p for p in platforms if p.y < SCREEN_HEIGHT + 50]

                while len(platforms) < 12:
                    last_y = min(p.y for p in platforms) if platforms else 0
                    new_platforms = generate_platforms(3, last_y - random.randint(60, 90))
                    platforms.extend(new_platforms)

                # Update score
                score += int(diff // 2)
                if score > high_score:
                    high_score = score

            # Check if player fell off screen
            if player.y > SCREEN_HEIGHT + 50:
                game_state = "game_over"

            # Update rising death zone
            rising_speed = min(rising_speed + RISING_ACCEL, MAX_RISING_SPEED)
            rising_y -= rising_speed

            # Check collision with rising zone
            if player.y + player.height > rising_y:
                game_state = "game_over"

            # Update special platforms
            for platform in platforms:
                platform.update()

        # Draw
        draw_background(screen, scroll_y)

        # Draw rising zone (behind platforms)
        if game_state == "playing":
            draw_rising_zone(screen, rising_y)

        # Draw platforms
        for platform in platforms:
            platform.draw(screen)

        # Draw player
        player.draw(screen)

        # Draw UI
        if game_state == "playing":
            show_score(screen, score, high_score)
        elif game_state == "start":
            show_start_screen(screen)
            continue
        elif game_state == "game_over":
            show_score(screen, score, high_score)
            show_game_over(screen, score, high_score)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
