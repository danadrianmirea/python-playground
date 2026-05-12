# snake.py - Classic Snake game

import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake")

# Game constants
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
FPS = 10

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (50, 200, 50)
DARK_GREEN = (30, 150, 30)
RED = (255, 50, 50)
DARK_RED = (180, 30, 30)
GRAY = (40, 40, 40)
LIGHT_GRAY = (60, 60, 60)
YELLOW = (255, 255, 50)

# Fonts
font = pygame.font.Font(None, 36)
score_font = pygame.font.Font(None, 48)
game_over_font = pygame.font.Font(None, 72)

# Clock
clock = pygame.time.Clock()


def draw_grid():
    """Draw the background grid lines."""
    for x in range(0, SCREEN_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (SCREEN_WIDTH, y))


def draw_snake(snake_body):
    """Draw the snake with rounded segments."""
    for i, segment in enumerate(snake_body):
        x, y = segment
        rect = pygame.Rect(x * GRID_SIZE + 1, y * GRID_SIZE + 1, GRID_SIZE - 2, GRID_SIZE - 2)
        if i == 0:
            # Head - brighter
            pygame.draw.rect(screen, GREEN, rect, border_radius=4)
            # Eyes
            eye_size = 3
            dx, dy = snake_body[0][0] - snake_body[1][0], snake_body[0][1] - snake_body[1][1]
            if dx == 1:  # moving right
                pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 14, y * GRID_SIZE + 6), eye_size)
                pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 14, y * GRID_SIZE + 14), eye_size)
                pygame.draw.circle(screen, BLACK, (x * GRID_SIZE + 15, y * GRID_SIZE + 6), 1)
                pygame.draw.circle(screen, BLACK, (x * GRID_SIZE + 15, y * GRID_SIZE + 14), 1)
            elif dx == -1:  # moving left
                pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 6, y * GRID_SIZE + 6), eye_size)
                pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 6, y * GRID_SIZE + 14), eye_size)
                pygame.draw.circle(screen, BLACK, (x * GRID_SIZE + 5, y * GRID_SIZE + 6), 1)
                pygame.draw.circle(screen, BLACK, (x * GRID_SIZE + 5, y * GRID_SIZE + 14), 1)
            elif dy == 1:  # moving down
                pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 6, y * GRID_SIZE + 14), eye_size)
                pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 14, y * GRID_SIZE + 14), eye_size)
                pygame.draw.circle(screen, BLACK, (x * GRID_SIZE + 6, y * GRID_SIZE + 15), 1)
                pygame.draw.circle(screen, BLACK, (x * GRID_SIZE + 14, y * GRID_SIZE + 15), 1)
            else:  # moving up
                pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 6, y * GRID_SIZE + 6), eye_size)
                pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 14, y * GRID_SIZE + 6), eye_size)
                pygame.draw.circle(screen, BLACK, (x * GRID_SIZE + 6, y * GRID_SIZE + 5), 1)
                pygame.draw.circle(screen, BLACK, (x * GRID_SIZE + 14, y * GRID_SIZE + 5), 1)
        else:
            # Body - alternating shades
            color = DARK_GREEN if i % 2 == 0 else GREEN
            pygame.draw.rect(screen, color, rect, border_radius=3)


def draw_food(food_pos):
    """Draw the food as a red apple-like circle."""
    x, y = food_pos
    center = (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2)
    pygame.draw.circle(screen, RED, center, GRID_SIZE // 2 - 2)
    pygame.draw.circle(screen, DARK_RED, center, GRID_SIZE // 4)
    # Highlight
    pygame.draw.circle(screen, (255, 150, 150), (center[0] - 3, center[1] - 3), 2)


def show_game_over(score):
    """Display the game over screen."""
    screen.fill(BLACK)
    game_over_text = game_over_font.render("GAME OVER", True, RED)
    score_text = score_font.render(f"Score: {score}", True, WHITE)
    restart_text = font.render("Press SPACE to play again or ESC to quit", True, WHITE)

    screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 80))
    screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, SCREEN_HEIGHT // 2 - 10))
    screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 50))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                    return True  # Restart
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
    return False


def get_random_food(snake_body):
    """Generate food at a random position not occupied by the snake."""
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake_body:
            return pos


def main():
    """Main game loop."""
    # Initial snake: 3 segments in the middle, moving right
    start_x = GRID_WIDTH // 2
    start_y = GRID_HEIGHT // 2
    snake = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
    direction = RIGHT
    next_direction = RIGHT
    food = get_random_food(snake)
    score = 0
    running = True

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != DOWN:
                    next_direction = UP
                elif event.key == pygame.K_DOWN and direction != UP:
                    next_direction = DOWN
                elif event.key == pygame.K_LEFT and direction != RIGHT:
                    next_direction = LEFT
                elif event.key == pygame.K_RIGHT and direction != LEFT:
                    next_direction = RIGHT

        # Apply direction
        direction = next_direction

        # Move snake
        head = snake[0]
        new_head = (head[0] + direction[0], head[1] + direction[1])

        # Wall collision - wrap around
        new_head = (new_head[0] % GRID_WIDTH, new_head[1] % GRID_HEIGHT)

        # Self collision check
        if new_head in snake:
            if show_game_over(score):
                # Restart
                start_x = GRID_WIDTH // 2
                start_y = GRID_HEIGHT // 2
                snake = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
                direction = RIGHT
                next_direction = RIGHT
                food = get_random_food(snake)
                score = 0
                continue
            else:
                running = False
                break

        # Insert new head
        snake.insert(0, new_head)

        # Check food collision
        if new_head == food:
            score += 10
            food = get_random_food(snake)
            # Don't remove tail - snake grows
        else:
            snake.pop()  # Remove tail

        # Draw everything
        screen.fill(BLACK)
        draw_grid()
        draw_food(food)
        draw_snake(snake)

        # Draw score
        score_text = score_font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()