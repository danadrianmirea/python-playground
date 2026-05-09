# breakout.py - Breakout game

import pygame
import random

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Game constants
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
PADDLE_Y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
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
      self.x = SCREEN_WIDTH // 2
      self.y = SCREEN_HEIGHT // 2 - BALL_RADIUS
      self.speed_x = random.choice([-4, 4])
      self.speed_y = 0
      self.speed_x *= 1.5
      self.speed_y = -1
      self.reset()

  def reset(self):
      self.x = SCREEN_WIDTH // 2
      self.y = SCREEN_HEIGHT // 2 - BALL_RADIUS
      self.speed_x = random.choice([-4, 4])
      self.speed_y = -3

  def update(self):
      self.x += self.speed_x
      self.y += self.speed_y

      # Bounce off top and bottom
      if self.y <= 0 or self.y >= SCREEN_HEIGHT - BALL_RADIUS:
          self.speed_y *= -1

      # Bounce off left and right
      if self.x <= 0 or self.x >= SCREEN_WIDTH - BALL_RADIUS:
          self.speed_x *= -1

  def draw(self, surface):
      pygame.draw.circle(surface, BALL_COLOR, (int(self.x), int(self.y)), BALL_RADIUS)

  def is_in_paddle(self):
      return PADDLE_Y - BALL_RADIUS <= self.y <= PADDLE_Y + PADDLE_HEIGHT and \
             PADDLE_WIDTH // 2 <= self.x <= SCREEN_WIDTH - PADDLE_WIDTH // 2

  def hit_paddle(self):
      return self.y + BALL_RADIUS >= PADDLE_Y and self.x >= PADDLE_WIDTH // 2 and self.x <= SCREEN_WIDTH - PADDLE_WIDTH // 2


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

  def update(self, ball):
      if self.active:
          if self.hit(ball):
              self.active = False

  def hit(self, ball):
      return ball.x >= self.x and ball.x <= self.x + self.width and \
             ball.y + ball.speed_y >= self.y and ball.y + ball.speed_y <= self.y + self.height

  def draw(self, surface):
      if self.active:
          pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height))
      else:
          pygame.draw.rect(surface, BRICK_EMPTY_COLOR, (self.x, self.y, self.width, self.height))


class Game:
  def __init__(self):
      self.score = 0
      self.lives = 3
      self.ball = Ball()
      self.paddle = Paddle()
      self.bricks = []
      self.create_bricks()
      self.running = True

  def create_bricks(self):
      for row in range(BRICK_ROWS):
          for col in range(BRICK_COLS):
              brick_x = BRICK_OFFSET_X + col * (BRICK_WIDTH + BRICK_PADDING)
              brick_y = BRICK_OFFSET_Y + row * (BRICK_HEIGHT + BRICK_PADDING)
              brick_color = tuple([
                  255, 255, 0,
                  255, 100, 100,
                  200, 255, 100,
                  100, 255, 255,
                  100, 100, 255
              ][row])
              self.bricks.append(Brick(brick_x, brick_y, brick_color))

  def update(self):
      keys = pygame.key.get_pressed()
      if keys[pygame.K_LEFT] or keys[pygame.K_a]:
          self.paddle.move_left()
      elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
          self.paddle.move_right()

      self.ball.update()

      # Check paddle collision
      if self.ball.is_in_paddle():
          self.ball.speed_y *= -1
          self.ball.speed_y = abs(self.ball.speed_y)
          self.ball.speed_x *= 1.02  # Slight speed increase on each hit

  def handle_input(self):
      keys = pygame.key.get_pressed()
      if keys[pygame.K_SPACE] and self.paddle.active:
          self.ball.speed_x = random.choice([-4, 4])
          self.ball.speed_y = -3

  def check_collisions(self):
      # Check if ball hit brick
      for brick in self.bricks:
          if brick.active and brick.hit(self.ball):
              brick.active = False
              self.score += 10
              self.ball.speed_y = -self.ball.speed_y * 0.85  # Speed increases on hit

  def handle_game_over(self):
      if self.lives <= 0 or self.score >= 500:
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

      # Draw game over / start screen
      if not self.running and self.score < 500:
          overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
          overlay.set_alpha(150)
          overlay.fill((0, 0, 0))
          surface.blit(overlay, (0, 0))

          if self.score >= 500:
              game_over_text = game_over_font.render("YOU WIN!", True, GREEN)
          else:
              game_over_text = game_over_font.render("GAME OVER", True, RED)

          surface.blit(game_over_text, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2))
          start_text = font.render("Press SPACE to restart", True, WHITE)
          surface.blit(start_text, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 + 60))

      # Draw instructions
      if self.score < 100:
          instructions = font.render("Controls: Arrows to move | Space to launch", True, WHITE)
          surface.blit(instructions, (10, SCREEN_HEIGHT - 30))

  def run(self):
      clock = pygame.time.Clock()
      self.running = True

      while self.running:
          dt = clock.tick(60)

          # Handle events
          for event in pygame.event.get():
              if event.type == pygame.QUIT:
                  self.running = False
              if event.type == pygame.KEYDOWN:
                  if event.key == pygame.K_SPACE:
                      if self.score >= 500:
                          self.score = 0
                          self.lives = 3
                          self.running = True
                          self.ball.reset()
                      else:
                          self.running = False

                  if event.key == pygame.K_r:
                      self.running = True
                      self.score = 0
                      self.lives = 3
                      self.ball.reset()

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