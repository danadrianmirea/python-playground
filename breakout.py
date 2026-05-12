# breakout.py - Breakout game

import pygame
import random

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Breakout")

# Game constants
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
PADDLE_Y = SCREEN_HEIGHT - 50  # Near the bottom
PADDLE_SPEED = 7
BALL_RADIUS = 10
BRICK_WIDTH = 80
BRICK_HEIGHT = 30
BRICK_ROWS = 5
BRICK_COLS = 9
BRICK_PADDING = 10
BRICK_OFFSET_X = (SCREEN_WIDTH - (BRICK_COLS * BRICK_WIDTH) - (BRICK_COLS * BRICK_PADDING)) // 2
BRICK_OFFSET_Y = 60
BRICK_COLOR = (255, 0, 0)
BRICK_EMPTY_COLOR = (100, 100, 100)
BACKGROUND_COLOR = (0, 0, 50)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Player paddle color
PLAYER_COLOR = (50, 200, 50)

# Ball color
BALL_COLOR = (255, 215, 0)

# Load font
font = pygame.font.Font(None, 36)
score_font = pygame.font.Font(None, 48)
game_over_font = pygame.font.Font(None, 72)


class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.speed_x = random.choice([-4, 4])
        self.speed_y = -3
        self.launched = False

    def launch(self):
        if not self.launched:
            self.speed_x = random.choice([-4, 4])
            self.speed_y = -3
            self.launched = True

    def update(self):
        if not self.launched:
            # Stick to paddle before launch
            return

        self.x += self.speed_x
        self.y += self.speed_y

        # Bounce off top wall
        if self.y <= 0:
            self.speed_y *= -1

        # Bounce off left and right walls
        if self.x <= BALL_RADIUS or self.x >= SCREEN_WIDTH - BALL_RADIUS:
            self.speed_x *= -1
            # Keep ball within horizontal bounds
            if self.x < BALL_RADIUS:
                self.x = BALL_RADIUS
            if self.x > SCREEN_WIDTH - BALL_RADIUS:
                self.x = SCREEN_WIDTH - BALL_RADIUS

    def draw(self, surface):
        pygame.draw.circle(surface, BALL_COLOR, (int(self.x), int(self.y)), BALL_RADIUS)

    def hit_paddle(self, paddle):
        """Check if ball collides with paddle and bounce accordingly."""
        if (self.y + BALL_RADIUS >= paddle.y and
                self.y - BALL_RADIUS <= paddle.y + paddle.height and
                self.x + BALL_RADIUS >= paddle.x and
                self.x - BALL_RADIUS <= paddle.x + paddle.width):
            # Bounce upward
            self.speed_y = -abs(self.speed_y)
            # Add slight angle based on where ball hits paddle
            hit_pos = (self.x - paddle.x) / paddle.width  # 0 to 1
            self.speed_x = (hit_pos - 0.5) * 8  # Range: -4 to 4
            return True
        return False

    def is_below_screen(self):
        return self.y > SCREEN_HEIGHT + BALL_RADIUS


class Paddle:
    def __init__(self):
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
        self.x = SCREEN_WIDTH // 2 - self.width // 2
        self.y = PADDLE_Y
        self.speed = PADDLE_SPEED

    def move_right(self):
        self.x += self.speed
        if self.x > SCREEN_WIDTH - self.width:
            self.x = SCREEN_WIDTH - self.width

    def move_left(self):
        self.x -= self.speed
        if self.x < 0:
            self.x = 0

    def draw(self, surface):
        pygame.draw.rect(surface, PLAYER_COLOR, (self.x, self.y, self.width, self.height))


class Brick:
    def __init__(self, x, y, color=BRICK_COLOR):
        self.width = BRICK_WIDTH
        self.height = BRICK_HEIGHT
        self.x = x
        self.y = y
        self.color = color
        self.active = True

    def hit(self, ball):
        """Check if ball collides with this brick and bounce the ball."""
        if not self.active:
            return False

        # Check collision using AABB
        if (ball.x + BALL_RADIUS >= self.x and
                ball.x - BALL_RADIUS <= self.x + self.width and
                ball.y + BALL_RADIUS >= self.y and
                ball.y - BALL_RADIUS <= self.y + self.height):
            self.active = False

            # Determine bounce direction
            # Calculate overlap on each side
            overlap_left = (ball.x + BALL_RADIUS) - self.x
            overlap_right = (self.x + self.width) - (ball.x - BALL_RADIUS)
            overlap_top = (ball.y + BALL_RADIUS) - self.y
            overlap_bottom = (self.y + self.height) - (ball.y - BALL_RADIUS)

            # Find smallest overlap to determine bounce direction
            min_overlap_x = min(overlap_left, overlap_right)
            min_overlap_y = min(overlap_top, overlap_bottom)

            if min_overlap_x < min_overlap_y:
                ball.speed_x *= -1
            else:
                ball.speed_y *= -1

            return True
        return False

    def draw(self, surface):
        if self.active:
            pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height))


class Game:
    def __init__(self):
        self.score = 0
        self.lives = 3
        self.ball = Ball()
        self.paddle = Paddle()
        self.bricks = []
        self.create_bricks()
        self.running = True
        self.game_over = False
        self.won = False

    def create_bricks(self):
        self.bricks = []
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                brick_x = BRICK_OFFSET_X + col * (BRICK_WIDTH + BRICK_PADDING)
                brick_y = BRICK_OFFSET_Y + row * (BRICK_HEIGHT + BRICK_PADDING)
                brick_colors = [
                    (255, 0, 0),      # Red
                    (255, 165, 0),    # Orange
                    (255, 255, 0),    # Yellow
                    (0, 255, 0),      # Green
                    (0, 0, 255),      # Blue
                ]
                brick_color = brick_colors[row % len(brick_colors)]
                self.bricks.append(Brick(brick_x, brick_y, brick_color))

    def reset_game(self):
        self.score = 0
        self.lives = 3
        self.ball.reset()
        self.paddle = Paddle()
        self.create_bricks()
        self.game_over = False
        self.won = False
        self.running = True

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.paddle.move_left()
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.paddle.move_right()

        # Stick ball to paddle before launch
        if not self.ball.launched:
            self.ball.x = self.paddle.x + self.paddle.width // 2
            self.ball.y = self.paddle.y - BALL_RADIUS

        self.ball.update()

        # Check paddle collision
        if self.ball.launched:
            self.ball.hit_paddle(self.paddle)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.ball.launch()

    def check_collisions(self):
        # Check if ball hit brick
        for brick in self.bricks:
            if brick.active and brick.hit(self.ball):
                self.score += 10

    def handle_game_over(self):
        # Check if ball fell below screen
        if self.ball.is_below_screen():
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
                self.running = False
            else:
                # Reset ball position
                self.ball.reset()

        # Check if all bricks are destroyed
        if all(not brick.active for brick in self.bricks):
            self.won = True
            self.running = False

    def draw(self, surface):
        # Background
        surface.fill(BACKGROUND_COLOR)

        # Draw bricks
        for brick in self.bricks:
            brick.draw(surface)

        # Draw paddle
        self.paddle.draw(surface)

        # Draw ball
        self.ball.draw(surface)

        # Draw score
        score_text = score_font.render(f"Score: {self.score}", True, WHITE)
        surface.blit(score_text, (10, 10))

        # Draw lives
        lives_text = font.render(f"Lives: {self.lives}", True, WHITE)
        surface.blit(lives_text, (SCREEN_WIDTH // 2 - 30, SCREEN_HEIGHT - 30))

        # Draw game over / win screen
        if self.game_over or self.won:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(150)
            overlay.fill((0, 0, 0))
            surface.blit(overlay, (0, 0))

            if self.won:
                game_over_text = game_over_font.render("YOU WIN!", True, GREEN)
            else:
                game_over_text = game_over_font.render("GAME OVER", True, RED)

            surface.blit(game_over_text, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2))
            restart_text = font.render("Press SPACE to restart", True, WHITE)
            surface.blit(restart_text, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 60))

        # Draw instructions before ball is launched
        if not self.ball.launched:
            instructions = font.render("Press SPACE to launch ball", True, WHITE)
            surface.blit(instructions, (SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT // 2 + 100))

    def run(self):
        clock = pygame.time.Clock()
        self.running = True

        while self.running:
            clock.tick(60)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if self.game_over or self.won:
                            self.reset_game()
                        else:
                            self.ball.launch()
                    if event.key == pygame.K_r:
                        self.reset_game()

            if self.running:
                self.handle_input()
                self.update()
                self.check_collisions()
                self.handle_game_over()

            self.draw(screen)
            pygame.display.flip()

        pygame.quit()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()