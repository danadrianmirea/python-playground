# flappyBird.py - Flappy Bird clone

import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Game constants
GRAVITY = 0.5
FLAP_STRENGTH = -9
PIPE_WIDTH = 60
PIPE_GAP = 180
PIPE_SPEED = 4
PIPE_FREQUENCY = 1500  # milliseconds
BIRD_RADIUS = 15
GROUND_HEIGHT = 80
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
BLUE = (135, 206, 235)
BROWN = (139, 69, 19)
GRAY = (200, 200, 200)
RED = (255, 0, 0)

# Fonts
font_large = pygame.font.SysFont("Arial", 48, bold=True)
font_medium = pygame.font.SysFont("Arial", 28)
font_small = pygame.font.SysFont("Arial", 20)

clock = pygame.time.Clock()


class Bird:
    def __init__(self):
        self.x = SCREEN_WIDTH // 4
        self.y = SCREEN_HEIGHT // 2
        self.vel_y = 0
        self.radius = BIRD_RADIUS
        self.angle = 0

    def flap(self):
        self.vel_y = FLAP_STRENGTH

    def update(self):
        self.vel_y += GRAVITY
        self.y += self.vel_y

        # Calculate angle based on velocity
        self.angle = max(-30, min(90, self.vel_y * 3))

    def draw(self, surface):
        # Draw bird body (circle)
        pygame.draw.circle(surface, YELLOW, (int(self.x), int(self.y)), self.radius)

        # Draw wing (triangle)
        wing_offset_x = int(5 * pygame.math.Vector2(1, 0).rotate(self.angle).x)
        wing_offset_y = int(5 * pygame.math.Vector2(1, 0).rotate(self.angle).y)
        wing_points = [
            (int(self.x) - 5 + wing_offset_x, int(self.y) + wing_offset_y),
            (int(self.x) - 15, int(self.y) + 5),
            (int(self.x) - 15, int(self.y) - 5),
        ]
        pygame.draw.polygon(surface, (255, 200, 0), wing_points)

        # Draw eye
        eye_x = int(self.x + 8)
        eye_y = int(self.y - 4)
        pygame.draw.circle(surface, WHITE, (eye_x, eye_y), 5)
        pygame.draw.circle(surface, BLACK, (eye_x + 2, eye_y), 2)

        # Draw beak
        beak_points = [
            (int(self.x) + self.radius, int(self.y)),
            (int(self.x) + self.radius + 10, int(self.y) - 3),
            (int(self.x) + self.radius + 10, int(self.y) + 3),
        ]
        pygame.draw.polygon(surface, (255, 140, 0), beak_points)

    def get_rect(self):
        return pygame.Rect(
            self.x - self.radius,
            self.y - self.radius,
            self.radius * 2,
            self.radius * 2,
        )


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.gap = PIPE_GAP
        self.gap_y = random.randint(150, SCREEN_HEIGHT - GROUND_HEIGHT - 150)
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self, surface):
        # Top pipe
        top_height = self.gap_y - self.gap // 2
        top_rect = pygame.Rect(self.x, 0, self.width, top_height)
        pygame.draw.rect(surface, GREEN, top_rect)
        pygame.draw.rect(surface, DARK_GREEN, top_rect, 3)

        # Top pipe cap
        cap_rect = pygame.Rect(
            self.x - 5, top_height - 30, self.width + 10, 30
        )
        pygame.draw.rect(surface, GREEN, cap_rect)
        pygame.draw.rect(surface, DARK_GREEN, cap_rect, 3)

        # Bottom pipe
        bottom_y = self.gap_y + self.gap // 2
        bottom_height = SCREEN_HEIGHT - GROUND_HEIGHT - bottom_y
        bottom_rect = pygame.Rect(self.x, bottom_y, self.width, bottom_height)
        pygame.draw.rect(surface, GREEN, bottom_rect)
        pygame.draw.rect(surface, DARK_GREEN, bottom_rect, 3)

        # Bottom pipe cap
        cap_rect = pygame.Rect(
            self.x - 5, bottom_y, self.width + 10, 30
        )
        pygame.draw.rect(surface, GREEN, cap_rect)
        pygame.draw.rect(surface, DARK_GREEN, cap_rect, 3)

    def get_top_rect(self):
        return pygame.Rect(
            self.x,
            0,
            self.width,
            self.gap_y - self.gap // 2,
        )

    def get_bottom_rect(self):
        return pygame.Rect(
            self.x,
            self.gap_y + self.gap // 2,
            self.width,
            SCREEN_HEIGHT - GROUND_HEIGHT - (self.gap_y + self.gap // 2),
        )

    def off_screen(self):
        return self.x + self.width < 0


class Ground:
    def __init__(self):
        self.y = SCREEN_HEIGHT - GROUND_HEIGHT
        self.x1 = 0
        self.x2 = SCREEN_WIDTH

    def update(self):
        self.x1 -= PIPE_SPEED
        self.x2 -= PIPE_SPEED
        if self.x1 + SCREEN_WIDTH <= 0:
            self.x1 = self.x2 + SCREEN_WIDTH
        if self.x2 + SCREEN_WIDTH <= 0:
            self.x2 = self.x1 + SCREEN_WIDTH

    def draw(self, surface):
        # Draw ground
        pygame.draw.rect(
            surface, BROWN, (0, self.y, SCREEN_WIDTH, GROUND_HEIGHT)
        )
        # Draw grass on top
        pygame.draw.rect(
            surface, GREEN, (0, self.y, SCREEN_WIDTH, 10)
        )
        # Draw ground lines for scrolling effect
        for x in [self.x1, self.x2]:
            for i in range(0, SCREEN_WIDTH, 40):
                line_x = x + i
                if 0 <= line_x <= SCREEN_WIDTH:
                    pygame.draw.line(
                        surface,
                        (100, 50, 0),
                        (line_x, self.y + 20),
                        (line_x + 20, self.y + 20),
                        2,
                    )


def draw_background(surface):
    """Draw the sky background with clouds."""
    surface.fill(BLUE)
    # Draw some simple clouds
    cloud_positions = [(50, 80), (200, 120), (320, 60), (100, 200)]
    for cx, cy in cloud_positions:
        pygame.draw.ellipse(surface, WHITE, (cx, cy, 60, 30))
        pygame.draw.ellipse(surface, WHITE, (cx + 20, cy - 10, 40, 25))
        pygame.draw.ellipse(surface, WHITE, (cx + 40, cy, 30, 20))


def show_score(surface, score, high_score):
    """Display the current score and high score."""
    score_text = font_large.render(str(score), True, WHITE)
    score_shadow = font_large.render(str(score), True, BLACK)
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 60))
    surface.blit(score_shadow, (score_rect.x + 2, score_rect.y + 2))
    surface.blit(score_text, score_rect)

    # Show high score
    hs_text = font_small.render(f"Best: {high_score}", True, WHITE)
    hs_shadow = font_small.render(f"Best: {high_score}", True, BLACK)
    hs_rect = hs_text.get_rect(center=(SCREEN_WIDTH // 2, 100))
    surface.blit(hs_shadow, (hs_rect.x + 1, hs_rect.y + 1))
    surface.blit(hs_text, hs_rect)


def show_game_over(surface, score, high_score):
    """Display the game over screen."""
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill(BLACK)
    surface.blit(overlay, (0, 0))

    # Game Over text
    go_text = font_large.render("GAME OVER", True, RED)
    go_shadow = font_large.render("GAME OVER", True, (100, 0, 0))
    go_rect = go_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80))
    surface.blit(go_shadow, (go_rect.x + 3, go_rect.y + 3))
    surface.blit(go_text, go_rect)

    # Score
    score_text = font_medium.render(f"Score: {score}", True, WHITE)
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
    surface.blit(score_text, score_rect)

    # High score
    hs_text = font_medium.render(f"Best: {high_score}", True, YELLOW)
    hs_rect = hs_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
    surface.blit(hs_text, hs_rect)

    # Restart instruction
    restart_text = font_small.render("Press SPACE or Click to Restart", True, WHITE)
    restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
    surface.blit(restart_text, restart_rect)

    # Quit instruction
    quit_text = font_small.render("Press ESC to Quit", True, GRAY)
    quit_rect = quit_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 110))
    surface.blit(quit_text, quit_rect)


def show_start_screen(surface):
    """Display the start screen."""
    draw_background(surface)

    # Title
    title_text = font_large.render("Flappy Bird", True, YELLOW)
    title_shadow = font_large.render("Flappy Bird", True, (150, 100, 0))
    title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 150))
    surface.blit(title_shadow, (title_rect.x + 3, title_rect.y + 3))
    surface.blit(title_text, title_rect)

    # Bird preview
    preview_bird = Bird()
    preview_bird.y = SCREEN_HEIGHT // 2 - 30
    preview_bird.draw(surface)

    # Instructions
    inst1 = font_medium.render("Press SPACE or Click to Flap", True, WHITE)
    inst1_rect = inst1.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60))
    surface.blit(inst1, inst1_rect)

    inst2 = font_small.render("Avoid the pipes and ground!", True, WHITE)
    inst2_rect = inst2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100))
    surface.blit(inst2, inst2_rect)

    # Draw ground
    ground = Ground()
    ground.draw(surface)

    pygame.display.flip()


def reset_game():
    """Reset all game objects to initial state."""
    bird = Bird()
    pipes = []
    score = 0
    return bird, pipes, score


def check_collision(bird, pipes):
    """Check if the bird collides with pipes, ground, or goes off screen."""
    bird_rect = bird.get_rect()

    # Check ground collision
    if bird.y + bird.radius >= SCREEN_HEIGHT - GROUND_HEIGHT:
        return True

    # Check ceiling collision
    if bird.y - bird.radius <= 0:
        return True

    # Check pipe collisions
    for pipe in pipes:
        if bird_rect.colliderect(pipe.get_top_rect()) or bird_rect.colliderect(
            pipe.get_bottom_rect()
        ):
            return True

    return False


def main():
    """Main game loop."""
    bird, pipes, score = reset_game()
    high_score = 0
    game_state = "start"  # "start", "playing", "game_over"
    last_pipe_time = 0
    running = True

    while running:
        dt = clock.tick(FPS)
        current_time = pygame.time.get_ticks()

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
                        bird, pipes, score = reset_game()
                        last_pipe_time = current_time
                    elif game_state == "playing":
                        bird.flap()
                    elif game_state == "game_over":
                        game_state = "playing"
                        bird, pipes, score = reset_game()
                        last_pipe_time = current_time

            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_state == "start":
                    game_state = "playing"
                    bird, pipes, score = reset_game()
                    last_pipe_time = current_time
                elif game_state == "playing":
                    bird.flap()
                elif game_state == "game_over":
                    game_state = "playing"
                    bird, pipes, score = reset_game()
                    last_pipe_time = current_time

        # Update
        if game_state == "playing":
            bird.update()

            # Spawn pipes
            if current_time - last_pipe_time > PIPE_FREQUENCY:
                pipes.append(Pipe(SCREEN_WIDTH))
                last_pipe_time = current_time

            # Update pipes
            for pipe in pipes[:]:
                pipe.update()
                if pipe.off_screen():
                    pipes.remove(pipe)

            # Check score
            for pipe in pipes:
                if not pipe.passed and pipe.x + pipe.width < bird.x:
                    pipe.passed = True
                    score += 1
                    if score > high_score:
                        high_score = score

            # Check collisions
            if check_collision(bird, pipes):
                game_state = "game_over"

        # Draw
        draw_background(screen)

        # Draw pipes
        for pipe in pipes:
            pipe.draw(screen)

        # Draw ground
        ground = Ground()
        if game_state == "playing":
            ground.update()
        ground.draw(screen)

        # Draw bird
        bird.draw(screen)

        # Draw UI
        if game_state == "playing":
            show_score(screen, score, high_score)
        elif game_state == "start":
            show_start_screen(screen)
            continue  # Skip the flip since show_start_screen already does it
        elif game_state == "game_over":
            show_score(screen, score, high_score)
            show_game_over(screen, score, high_score)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
