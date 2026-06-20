"""
Pong - A retro table tennis arcade game inspired by the 1972 classic.

Two players battle it out in a digital ping-pong match.
First to 7 points wins!

Controls:
  Player 1 (Left):  W - Up, S - Down
  Player 2 (Right): UP arrow - Up, DOWN arrow - Down
  SPACE - Start / Pause
  R - Restart after game over
  ESC - Quit
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
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DIM_WHITE = (180, 180, 180)
GREEN = (50, 200, 50)
RED = (200, 50, 50)
YELLOW = (255, 255, 100)

# Game constants
PADDLE_WIDTH = 12
PADDLE_HEIGHT = 90
PADDLE_SPEED = 6
BALL_SIZE = 14
BALL_SPEED_INITIAL = 5
BALL_SPEED_MAX = 12
BALL_ACCELERATION = 0.3
WINNING_SCORE = 7
PADDLE_MARGIN = 30

# AI settings (for single-player mode)
AI_SPEED = 4.5
AI_REACTION_MARGIN = 40  # How close to the paddle the ball needs to be before AI reacts


class Paddle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
        self.speed = PADDLE_SPEED
        self.score = 0
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def move_up(self):
        self.y -= self.speed
        if self.y < 0:
            self.y = 0
        self.rect.y = self.y

    def move_down(self):
        self.y += self.speed
        if self.y + self.height > WINDOW_HEIGHT:
            self.y = WINDOW_HEIGHT - self.height
        self.rect.y = self.y

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.rect.x = self.x
        self.rect.y = self.y

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)
        # Add a subtle glow effect
        pygame.draw.rect(screen, DIM_WHITE, self.rect, 2)


class Ball:
    def __init__(self):
        self.reset()

    def reset(self, direction=None):
        self.x = WINDOW_WIDTH // 2 - BALL_SIZE // 2
        self.y = WINDOW_HEIGHT // 2 - BALL_SIZE // 2
        self.size = BALL_SIZE
        self.speed = BALL_SPEED_INITIAL

        # Randomize initial direction
        angle = random.uniform(-0.5, 0.5)
        if direction is None:
            direction = random.choice([-1, 1])
        self.dx = direction * math.cos(angle) * self.speed
        self.dy = math.sin(angle) * self.speed
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.rect.x = self.x
        self.rect.y = self.y

        # Top/bottom wall bounce
        if self.y <= 0:
            self.y = 0
            self.dy = abs(self.dy)
        elif self.y + self.size >= WINDOW_HEIGHT:
            self.y = WINDOW_HEIGHT - self.size
            self.dy = -abs(self.dy)

    def draw(self, screen):
        pygame.draw.ellipse(screen, WHITE, self.rect)
        # Glow effect
        pygame.draw.ellipse(screen, DIM_WHITE, self.rect, 2)

    def get_center_y(self):
        return self.y + self.size // 2


def draw_center_line(screen):
    """Draw the dashed center line."""
    dash_length = 15
    gap_length = 10
    x = WINDOW_WIDTH // 2
    y = 0
    while y < WINDOW_HEIGHT:
        pygame.draw.rect(screen, DIM_WHITE, (x - 2, y, 4, dash_length))
        y += dash_length + gap_length


def draw_score(screen, font, left_score, right_score):
    """Draw the score for both players."""
    score_surf = font.render(str(left_score), True, WHITE)
    screen.blit(score_surf, (WINDOW_WIDTH // 2 - 80 - score_surf.get_width() // 2, 30))

    score_surf = font.render(str(right_score), True, WHITE)
    screen.blit(score_surf, (WINDOW_WIDTH // 2 + 80 - score_surf.get_width() // 2, 30))


def show_message(screen, font, text, sub_text=None, color=WHITE):
    """Display a centered message on screen."""
    text_surf = font.render(text, True, color)
    text_rect = text_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 30))
    screen.blit(text_surf, text_rect)

    if sub_text:
        small_font = pygame.font.Font(None, 28)
        sub_surf = small_font.render(sub_text, True, DIM_WHITE)
        sub_rect = sub_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
        screen.blit(sub_surf, sub_rect)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("PONG")
    clock = pygame.time.Clock()

    # Fonts
    score_font = pygame.font.Font(None, 72)
    title_font = pygame.font.Font(None, 56)
    message_font = pygame.font.Font(None, 36)

    # Game objects
    paddle1 = Paddle(PADDLE_MARGIN, WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2)
    paddle2 = Paddle(WINDOW_WIDTH - PADDLE_MARGIN - PADDLE_WIDTH,
                     WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2)
    ball = Ball()

    # Game state
    state = "menu"  # menu, playing, paused, game_over
    mode = "1p"  # 1p (vs AI) or 2p
    winner = None
    serving_player = 1  # 1 = left serves, 2 = right serves

    # Sound effects (simple beeps using pygame mixer)
    pygame.mixer.init(frequency=22050, size=-16, channels=1)
    try:
        beep_hit = pygame.mixer.Sound(buffer=create_beep_sound(440, 0.08))
        beep_wall = pygame.mixer.Sound(buffer=create_beep_sound(330, 0.06))
        beep_score = pygame.mixer.Sound(buffer=create_beep_sound(220, 0.2))
    except Exception:
        beep_hit = beep_wall = beep_score = None

    running = True
    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if state == "menu":
                        state = "playing"
                        ball.reset(direction=1 if serving_player == 1 else -1)
                    elif state == "playing":
                        state = "paused"
                    elif state == "paused":
                        state = "playing"
                elif event.key == pygame.K_r and state == "game_over":
                    # Restart
                    paddle1.score = 0
                    paddle2.score = 0
                    paddle1.reset(PADDLE_MARGIN, WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2)
                    paddle2.reset(WINDOW_WIDTH - PADDLE_MARGIN - PADDLE_WIDTH,
                                  WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2)
                    state = "menu"
                    winner = None
                elif event.key == pygame.K_1 and state == "menu":
                    mode = "1p"
                elif event.key == pygame.K_2 and state == "menu":
                    mode = "2p"

        # --- Input handling ---
        keys = pygame.key.get_pressed()

        if state == "playing":
            # Player 1 controls (always human)
            if keys[pygame.K_w]:
                paddle1.move_up()
            if keys[pygame.K_s]:
                paddle1.move_down()

            if mode == "2p":
                # Player 2 controls (human)
                if keys[pygame.K_UP]:
                    paddle2.move_up()
                if keys[pygame.K_DOWN]:
                    paddle2.move_down()
            else:
                # AI for player 2
                ai_move_paddle(paddle2, ball)

            # --- Ball update ---
            ball.update()

            # --- Collision detection ---
            # Ball vs paddle 1 (left)
            if ball.rect.colliderect(paddle1.rect) and ball.dx < 0:
                handle_paddle_hit(ball, paddle1, 1)
                if beep_hit:
                    beep_hit.play()

            # Ball vs paddle 2 (right)
            if ball.rect.colliderect(paddle2.rect) and ball.dx > 0:
                handle_paddle_hit(ball, paddle2, -1)
                if beep_hit:
                    beep_hit.play()

            # --- Scoring ---
            if ball.x + ball.size < 0:
                # Player 2 scores
                paddle2.score += 1
                if beep_score:
                    beep_score.play()
                serving_player = 2
                if paddle2.score >= WINNING_SCORE:
                    winner = "Player 2" if mode == "2p" else "AI"
                    state = "game_over"
                else:
                    ball.reset(direction=-1)

            elif ball.x > WINDOW_WIDTH:
                # Player 1 scores
                paddle1.score += 1
                if beep_score:
                    beep_score.play()
                serving_player = 1
                if paddle1.score >= WINNING_SCORE:
                    winner = "Player 1"
                    state = "game_over"
                else:
                    ball.reset(direction=1)

        # --- Drawing ---
        screen.fill(BLACK)

        # Draw center line
        draw_center_line(screen)

        # Draw paddles and ball
        paddle1.draw(screen)
        paddle2.draw(screen)
        ball.draw(screen)

        # Draw scores
        draw_score(screen, score_font, paddle1.score, paddle2.score)

        # Draw state-specific overlays
        if state == "menu":
            # Semi-transparent overlay
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill(BLACK)
            screen.blit(overlay, (0, 0))

            title = title_font.render("PONG", True, WHITE)
            title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
            screen.blit(title, title_rect)

            show_message(screen, message_font,
                         "Press 1 for Single Player (vs AI)  |  Press 2 for Two Players",
                         "Press SPACE to start",
                         DIM_WHITE)

            # Draw controls
            controls_font = pygame.font.Font(None, 24)
            controls = [
                "Player 1: W / S",
                "Player 2: UP / DOWN",
                "First to 7 wins!"
            ]
            for i, text in enumerate(controls):
                ctrl_surf = controls_font.render(text, True, GRAY)
                ctrl_rect = ctrl_surf.get_rect(center=(WINDOW_WIDTH // 2,
                                                       WINDOW_HEIGHT // 2 + 80 + i * 25))
                screen.blit(ctrl_surf, ctrl_rect)

        elif state == "paused":
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(120)
            overlay.fill(BLACK)
            screen.blit(overlay, (0, 0))
            show_message(screen, message_font, "PAUSED", "Press SPACE to continue")

        elif state == "game_over":
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(160)
            overlay.fill(BLACK)
            screen.blit(overlay, (0, 0))

            winner_color = GREEN if winner == "Player 1" else RED
            show_message(screen, title_font, f"{winner} Wins!", "Press R to restart", winner_color)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


def handle_paddle_hit(ball, paddle, direction):
    """Handle ball collision with a paddle."""
    # Calculate where on the paddle the ball hit (normalized -1 to 1)
    paddle_center = paddle.y + paddle.height // 2
    ball_center = ball.get_center_y()
    relative_intersect = (ball_center - paddle_center) / (paddle.height // 2)
    # Clamp to [-1, 1]
    relative_intersect = max(-1.0, min(1.0, relative_intersect))

    # Calculate bounce angle (max 75 degrees from horizontal)
    max_angle = math.radians(75)
    bounce_angle = relative_intersect * max_angle

    # Increase speed slightly
    ball.speed = min(ball.speed + BALL_ACCELERATION, BALL_SPEED_MAX)

    # Set new velocity
    ball.dx = direction * math.cos(bounce_angle) * ball.speed
    ball.dy = math.sin(bounce_angle) * ball.speed

    # Push ball outside paddle to prevent sticking
    if direction == 1:
        ball.x = paddle.x + paddle.width + 1
    else:
        ball.x = paddle.x - ball.size - 1
    ball.rect.x = ball.x


def ai_move_paddle(paddle, ball):
    """Simple AI that follows the ball."""
    # Only react when ball is moving towards the AI paddle
    if ball.dx > 0:
        # Predict where the ball will be when it reaches the paddle
        ball_center_y = ball.get_center_y()
        paddle_center = paddle.y + paddle.height // 2

        # Add some reaction delay by only moving when ball is close enough
        if ball.x > WINDOW_WIDTH * 0.5:
            diff = ball_center_y - paddle_center
            if abs(diff) > AI_REACTION_MARGIN:
                if diff > 0:
                    paddle.move_down()
                else:
                    paddle.move_up()
            # Small random error to make it beatable
            elif random.random() < 0.02:
                if diff > 0:
                    paddle.move_down()
                else:
                    paddle.move_up()
    else:
        # Slowly return to center when ball is moving away
        paddle_center = paddle.y + paddle.height // 2
        center_diff = WINDOW_HEIGHT // 2 - paddle_center
        if abs(center_diff) > 20:
            if center_diff > 0:
                paddle.move_down()
            else:
                paddle.move_up()


def create_beep_sound(frequency, duration):
    """Generate a simple beep sound wave."""
    sample_rate = 22050
    n_samples = int(sample_rate * duration)
    buf = bytearray()
    for i in range(n_samples):
        sample = int(32767.0 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
        buf.extend(sample.to_bytes(2, 'little', signed=True))
    return bytes(buf)


if __name__ == "__main__":
    main()