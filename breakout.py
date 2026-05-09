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
PADDLE_Y = SCREEN_HEIGHT - 50
PADDLE_SPEED = 7
BALL_RADIUS = 10
BRICK_WIDTH = 80
BRICK_HEIGHT = 30
BRICK_ROWS = 5
BRICK_COLS = 9
BRICK_PADDING = 10
BRICK_OFFSET_X = (SCREEN_WIDTH - (BRICK_COLS * (BRICK_WIDTH + BRICK_PADDING))) // 2
BRICK_OFFSET_Y = 60
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

# Brick colors per row
BRICK_COLORS = [
    (255, 0, 0),      # Row 0 - Red
    (255, 165, 0),    # Row 1 - Orange
    (255, 255, 0),    # Row 2 - Yellow
    (0, 255, 0),      # Row 3 - Green
    (0, 0, 255),      # Row 4 - Blue
]

# Load font
font = pygame.font.Font(None, 36)
score_font = pygame.font.Font(None, 48)
game_over_font = pygame.font.Font(None, 72)


class Ball:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.speed_x = 0
        self.speed_y = 0
        self.launched = False

    def reset(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.speed_x = 0
        self.speed_y = 0
        self.launched = False

    def launch(self):
        self.speed_x = random.choice([-4, 4])
        self.speed_y = -3
        self.launched = True

    def update(self):
        if not self.launched:
            # Stick to paddle before launch
            self.x = SCREEN_WIDTH // 2
            self.y = SCREEN_HEIGHT // 2
            return

        self.x += self.speed_x
        self.y += self.speed_y

        # Bounce off top
        if self.y <= 0:
            self.speed_y *= -1
            self.y = 0

        # Bounce off left and right
        if self.x <= 0:
            self.speed_x *= -1
            self.x = 0
        elif self.x >= SCREEN_WIDTH - BALL_RADIUS:
            self.speed_x *= -1
            self.x = SCREEN_WIDTH - BALL_RADIUS

    def draw(self, surface):
        pygame.draw.circle(surface, BALL_COLOR, (int(self.x), int(self.y)), BALL_RADIUS)

    def hit_paddle(self, paddle):
        """Check if ball hits the paddle and bounce accordingly."""
        if (self.y + BALL_RADIUS >= paddle.y and
                self.y + BALL_RADIUS <= paddle.y + paddle.height + 5 and
                self.x >= paddle.x - BALL_RADIUS and
                self.x <= paddle.x + paddle.width + BALL_RADIUS):
            # Calculate bounce angle based on where ball hits the paddle
            hit_pos = (self.x - paddle.x) / paddle.width  # 0 to 1
            # Map hit position to angle: -60 to 60 degrees
            angle = (hit_pos - 0.5) * 120  # degrees
            import math
            speed = (self.speed_x ** 2 + self.speed_y ** 2) ** 0.5
            speed = min(speed * 1.02, 10)  # Slight speed increase, capped
            self.speed_x = speed * math.sin(math.radians(angle))
            self.speed_y = -speed * math.cos(math.radians(angle))
            return True
        return False


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
    def __init__(self, x, y, color):
        self.width = BRICK_WIDTH
        self.height = BRICK_HEIGHT
        self.x = x
        self.y = y
        self.color = color
        self.active = True

    def hit(self, ball):
        """Check collision with ball and bounce the ball off the brick."""
        if not self.active:
            return False

        # Find the closest point on the brick to the ball center
        closest_x = max(self.x, min(ball.x, self.x + self.width))
        closest_y = max(self.y, min(ball.y, self.y + self.height))

        # Calculate distance from ball center to closest point
        dx = ball.x - closest_x
        dy = ball.y - closest_y
        distance = (dx * dx + dy * dy) ** 0.5

        if distance < BALL_RADIUS:
            # Determine which side was hit
            overlap_x = BALL_RADIUS - abs(dx) if abs(dx) > 0 else 0
            overlap_y = BALL_RADIUS - abs(dy) if abs(dy) > 0 else 0

            if overlap_x < overlap_y:
                ball.speed_x *= -1
            else:
                ball.speed_y *= -1

            self.active = False
            return True
        return False

    def draw(self, surface):
        if self.active:
            pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height))
            # Draw a slight border for visual clarity
            pygame.draw.rect(surface, WHITE, (self.x, self.y, self.width, self.height), 1)


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
                brick_color = BRICK_COLORS[row % len(BRICK_COLORS)]
                self.bricks.append(Brick(brick_x, brick_y, brick_color))

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.paddle.move_left()
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.paddle.move_right()

        self.ball.update()

        # Check paddle collision
        if self.ball.launched and self.ball.hit_paddle(self.paddle):
            pass  # hit_paddle handles the bounce

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and not self.ball.launched:
            self.ball.launch()

    def check_collisions(self):
        # Check if ball hit brick
        for brick in self.bricks:
            if brick.active and brick.hit(self.ball):
                self.score += 10

    def handle_game_over(self):
        # Ball fell off the bottom
        if self.ball.y > SCREEN_HEIGHT:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
                self.running = False
            else:
                self.ball.reset()

        # Check win condition
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
        if not self.ball.launched and not self.game_over and not self.won:
            instructions = font.render("Press SPACE to launch the ball", True, WHITE)
            surface.blit(instructions, (SCREEN_WIDTH // 2 - 140, SCREEN_HEIGHT // 2 + 60))

    def run(self):
        clock = pygame.time.Clock()
        self.running = True
        self.game_over = False
        self.won = False

        while self.running:
            clock.tick(60)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if self.game_over or self.won:
                            # Restart game
                            self.__init__()
                            return self.run()
                        elif not self.ball.launched:
                            self.ball.launch()
                    if event.key == pygame.K_r:
                        self.__init__()
                        return self.run()

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