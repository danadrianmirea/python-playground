"""
Color Lines - A classic puzzle game
Click a ball to select it, then click an empty cell to move it.
Line up 5 or more balls of the same color to score points.
"""

import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 9
CELL_SIZE = 55
MARGIN = 30
HEADER_HEIGHT = 100
BOARD_PIXEL_SIZE = BOARD_SIZE * CELL_SIZE
WINDOW_WIDTH = BOARD_PIXEL_SIZE + MARGIN * 2
WINDOW_HEIGHT = BOARD_PIXEL_SIZE + HEADER_HEIGHT + MARGIN
BALL_RADIUS = 20
FPS = 60
LINE_LENGTH = 5
NEW_BALLS_COUNT = 3

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
LIGHT_GRAY = (120, 120, 120)
DARK_GRAY = (25, 25, 25)
BOARD_BG = (35, 35, 55)
CELL_COLOR1 = (45, 45, 70)
CELL_COLOR2 = (55, 55, 85)
GOLD = (255, 215, 0)
RED = (220, 40, 40)
GREEN = (40, 200, 40)
BLUE = (40, 80, 220)
YELLOW = (220, 220, 40)
PURPLE = (180, 40, 220)
CYAN = (40, 220, 220)
ORANGE = (220, 140, 20)
PINK = (220, 40, 140)

# Ball colors (matching original game's palette)
BALL_COLORS = [
    (220, 40, 40),    # Red
    (40, 200, 40),    # Green
    (40, 80, 220),    # Blue
    (220, 220, 40),   # Yellow
    (180, 40, 220),   # Purple
    (40, 220, 220),   # Cyan
    (220, 140, 20),   # Orange
]

# Animation states
ANIM_NONE = 0
ANIM_MOVE = 1
ANIM_DESTROY = 2
ANIM_APPEAR = 3


class Ball:
    """Represents a single ball on the board."""

    def __init__(self, color_idx, row, col):
        self.color_idx = color_idx
        self.row = row
        self.col = col
        self.target_row = row
        self.target_col = col
        self.anim_offset_x = 0
        self.anim_offset_y = 0
        self.anim_scale = 1.0
        self.anim_alpha = 255
        self.anim_state = ANIM_NONE
        self.anim_progress = 0
        self.anim_duration = 8
        self.marked_for_destroy = False

    def get_pixel_pos(self):
        """Get the pixel position of the ball."""
        x = MARGIN + self.col * CELL_SIZE + CELL_SIZE // 2 + self.anim_offset_x
        y = HEADER_HEIGHT + self.row * CELL_SIZE + CELL_SIZE // 2 + self.anim_offset_y
        return x, y

    def update_animation(self):
        """Update animation state."""
        if self.anim_state == ANIM_NONE:
            return True

        self.anim_progress += 1
        t = self.anim_progress / self.anim_duration

        if self.anim_state == ANIM_MOVE:
            if t >= 1:
                self.row = self.target_row
                self.col = self.target_col
                self.anim_offset_x = 0
                self.anim_offset_y = 0
                self.anim_state = ANIM_NONE
                return True
            # Ease in-out
            t = t * t * (3 - 2 * t)
            dx = (self.target_col - self.col) * CELL_SIZE
            dy = (self.target_row - self.row) * CELL_SIZE
            self.anim_offset_x = dx * t
            self.anim_offset_y = dy * t
            return False

        elif self.anim_state == ANIM_DESTROY:
            if t >= 1:
                self.anim_scale = 0
                self.anim_alpha = 0
                self.anim_state = ANIM_NONE
                return True
            self.anim_scale = 1 - t * 0.8
            self.anim_alpha = int(255 * (1 - t))
            return False

        elif self.anim_state == ANIM_APPEAR:
            if t >= 1:
                self.anim_scale = 1.0
                self.anim_alpha = 255
                self.anim_state = ANIM_NONE
                return True
            # Pop-in effect
            if t < 0.5:
                self.anim_scale = t * 2 * 1.2  # Grow slightly larger
            else:
                self.anim_scale = 1.2 - (t - 0.5) * 0.4  # Shrink back
            self.anim_alpha = int(255 * t)
            return False

        return True

    def draw(self, screen):
        """Draw the ball on the screen."""
        if self.anim_alpha <= 0:
            return

        x, y = self.get_pixel_pos()
        color = BALL_COLORS[self.color_idx]

        # Create a surface for the ball with alpha
        ball_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)

        radius = int(BALL_RADIUS * self.anim_scale)
        center = (CELL_SIZE // 2, CELL_SIZE // 2)

        if radius > 0:
            # Outer circle (main color)
            pygame.draw.circle(ball_surface, (*color, self.anim_alpha), center, radius)

            # Inner highlight (lighter spot top-left)
            highlight = tuple(min(255, c + 80) for c in color)
            pygame.draw.circle(ball_surface, (*highlight, self.anim_alpha),
                               (center[0] - 5, center[1] - 5), radius - 6)

            # Small bright spot
            bright = tuple(min(255, c + 120) for c in color)
            pygame.draw.circle(ball_surface, (*bright, self.anim_alpha),
                               (center[0] - 8, center[1] - 8), radius // 3)

            # Outline
            pygame.draw.circle(ball_surface, (*WHITE, self.anim_alpha // 3),
                               center, radius, 1)

        screen.blit(ball_surface, (x - CELL_SIZE // 2, y - CELL_SIZE // 2))


class ColorLines:
    """Main game class."""

    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Color Lines")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("segoeuisymbol", 32)
        self.font_medium = pygame.font.SysFont("segoeuisymbol", 22)
        self.font_small = pygame.font.SysFont("segoeuisymbol", 16)

        self.reset_game()

    def reset_game(self):
        """Reset the game state."""
        self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.score = 0
        self.selected_ball = None
        self.animating = False
        self.game_over = False
        self.move_completed = False  # True when a move animation finished (to spawn new balls if no lines)
        self.pending_new_balls = []  # Colors for the next 3 balls
        self.show_next = True

        # Generate initial 5 balls
        for _ in range(5):
            self._place_random_ball()

        # Generate the first set of "next 3 balls"
        self._generate_next_balls()

    def _generate_next_balls(self):
        """Generate the next 3 ball colors."""
        self.pending_new_balls = [random.randint(0, len(BALL_COLORS) - 1)
                                  for _ in range(NEW_BALLS_COUNT)]

    def _place_random_ball(self):
        """Place a random ball on an empty cell."""
        empty = self._get_empty_cells()
        if not empty:
            return None
        row, col = random.choice(empty)
        color_idx = random.randint(0, len(BALL_COLORS) - 1)
        ball = Ball(color_idx, row, col)
        ball.anim_state = ANIM_APPEAR
        ball.anim_progress = 0
        self.board[row][col] = ball
        return ball

    def _get_empty_cells(self):
        """Get list of empty cell coordinates."""
        empty = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row][col] is None:
                    empty.append((row, col))
        return empty

    def get_ball_at(self, row, col):
        """Get the ball at the given board position."""
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row][col]
        return None

    def get_ball_at_pixel(self, x, y):
        """Get the ball at the given pixel position."""
        col = (x - MARGIN) // CELL_SIZE
        row = (y - HEADER_HEIGHT) // CELL_SIZE
        return self.get_ball_at(row, col)

    def find_lines(self):
        """Find all lines of 5+ same-color balls. Returns a set of (row, col) tuples."""
        lines = set()

        # Check all 4 directions: horizontal, vertical, diagonal down-right, diagonal down-left
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                ball = self.board[row][col]
                if ball is None:
                    continue

                for dr, dc in directions:
                    # Count consecutive same-color balls in this direction
                    count = 1
                    positions = [(row, col)]

                    # Forward direction
                    r, c = row + dr, col + dc
                    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                        b = self.board[r][c]
                        if b and b.color_idx == ball.color_idx and not b.marked_for_destroy:
                            count += 1
                            positions.append((r, c))
                            r += dr
                            c += dc
                        else:
                            break

                    # Backward direction
                    r, c = row - dr, col - dc
                    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                        b = self.board[r][c]
                        if b and b.color_idx == ball.color_idx and not b.marked_for_destroy:
                            count += 1
                            positions.append((r, c))
                            r -= dr
                            c -= dc
                        else:
                            break

                    if count >= LINE_LENGTH:
                        for pos in positions:
                            lines.add(pos)

        return lines

    def remove_lines(self, lines):
        """Mark balls at given positions for destruction."""
        for row, col in lines:
            ball = self.board[row][col]
            if ball and not ball.marked_for_destroy:
                ball.marked_for_destroy = True
                ball.anim_state = ANIM_DESTROY
                ball.anim_progress = 0

    def move_ball(self, ball, target_row, target_col):
        """Move a ball to a new position."""
        # Clear old position
        self.board[ball.row][ball.col] = None

        # Set target
        ball.target_row = target_row
        ball.target_col = target_col
        ball.anim_state = ANIM_MOVE
        ball.anim_progress = 0

        # Place in new position
        self.board[target_row][target_col] = ball

    def has_path(self, r1, c1, r2, c2):
        """Check if there is a clear path between two cells (horizontal/vertical moves only)."""
        if r1 == r2 and c1 == c2:
            return False

        # BFS to find a path through empty cells
        visited = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        queue = [(r1, c1)]
        visited[r1][c1] = True

        while queue:
            r, c = queue.pop(0)
            if r == r2 and c == c2:
                return True

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and not visited[nr][nc]:
                    # Can move through empty cells or the destination cell
                    if self.board[nr][nc] is None or (nr == r2 and nc == c2):
                        visited[nr][nc] = True
                        queue.append((nr, nc))

        return False

    def has_valid_moves(self):
        """Check if there are any valid moves (at least one empty cell)."""
        return len(self._get_empty_cells()) > 0

    def handle_click(self, pos):
        """Handle mouse click."""
        if self.animating or self.game_over:
            return

        x, y = pos
        ball = self.get_ball_at_pixel(x, y)

        if ball is None:
            # Clicked on empty cell - if a ball is selected, try to move it
            col = (x - MARGIN) // CELL_SIZE
            row = (y - HEADER_HEIGHT) // CELL_SIZE
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                if self.board[row][col] is None and self.selected_ball is not None:
                    # Check if there's a valid path to the destination
                    if self.has_path(self.selected_ball.row, self.selected_ball.col, row, col):
                        # Move the selected ball
                        self.move_ball(self.selected_ball, row, col)
                        self.selected_ball = None
                        self.move_completed = True
                        self.animating = True
        else:
            # Clicked on a ball
            if self.selected_ball == ball:
                self.selected_ball = None  # Deselect
            else:
                self.selected_ball = ball  # Select

    def update(self):
        """Update game state."""
        if not self.animating:
            return

        # Update all ball animations
        all_done = True
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                ball = self.board[row][col]
                if ball and ball.anim_state != ANIM_NONE:
                    if not ball.update_animation():
                        all_done = False

        if all_done:
            # Remove any balls that finished their destroy animation
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    ball = self.board[row][col]
                    if ball and ball.marked_for_destroy:
                        self.board[row][col] = None

            # Check for lines
            lines = self.find_lines()
            if lines:
                self.score += len(lines) * 10
                self.remove_lines(lines)
                self.move_completed = False  # Free turn - don't spawn new balls
                self.animating = True
            elif self.move_completed:
                # Move completed and no lines formed - spawn the 3 new balls
                self.move_completed = False
                for color_idx in self.pending_new_balls:
                    empty = self._get_empty_cells()
                    if not empty:
                        self.game_over = True
                        self.animating = False
                        return
                    row, col = random.choice(empty)
                    ball = Ball(color_idx, row, col)
                    ball.anim_state = ANIM_APPEAR
                    ball.anim_progress = 0
                    self.board[row][col] = ball

                self._generate_next_balls()
                self.animating = True
            else:
                self.animating = False

                # Check if game is over
                if not self.has_valid_moves():
                    self.game_over = True

    def draw_board(self):
        """Draw the game board."""
        # Draw background
        self.screen.fill(DARK_GRAY)

        # Draw header
        pygame.draw.rect(self.screen, GRAY, (0, 0, WINDOW_WIDTH, HEADER_HEIGHT))
        pygame.draw.line(self.screen, LIGHT_GRAY, (0, HEADER_HEIGHT),
                         (WINDOW_WIDTH, HEADER_HEIGHT), 2)

        # Draw score
        score_text = self.font_large.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (MARGIN, 15))

        # Draw "Next 3" label and balls
        if self.show_next and self.pending_new_balls:
            next_label = self.font_small.render("Next:", True, LIGHT_GRAY)
            self.screen.blit(next_label, (WINDOW_WIDTH - 180, 15))

            for i, color_idx in enumerate(self.pending_new_balls):
                bx = WINDOW_WIDTH - 160 + i * 50
                by = 45
                color = BALL_COLORS[color_idx]
                pygame.draw.circle(self.screen, color, (bx, by), 14)
                highlight = tuple(min(255, c + 80) for c in color)
                pygame.draw.circle(self.screen, highlight, (bx - 3, by - 3), 10)
                bright = tuple(min(255, c + 120) for c in color)
                pygame.draw.circle(self.screen, bright, (bx - 5, by - 5), 5)

        # Draw hint
        if self.selected_ball:
            hint_text = self.font_small.render("Click an empty cell to move", True, GOLD)
            self.screen.blit(hint_text, (MARGIN, 55))

        # Draw board background
        board_rect = pygame.Rect(MARGIN, HEADER_HEIGHT, BOARD_PIXEL_SIZE, BOARD_PIXEL_SIZE)
        pygame.draw.rect(self.screen, BOARD_BG, board_rect)
        pygame.draw.rect(self.screen, LIGHT_GRAY, board_rect, 2)

        # Draw cells
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                rect = pygame.Rect(MARGIN + col * CELL_SIZE, HEADER_HEIGHT + row * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                color = CELL_COLOR1 if (row + col) % 2 == 0 else CELL_COLOR2
                pygame.draw.rect(self.screen, color, rect)

        # Draw balls
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                ball = self.board[row][col]
                if ball and not ball.marked_for_destroy:
                    ball.draw(self.screen)

        # Draw selection highlight
        if self.selected_ball:
            x = MARGIN + self.selected_ball.col * CELL_SIZE
            y = HEADER_HEIGHT + self.selected_ball.row * CELL_SIZE
            pygame.draw.rect(self.screen, GOLD, (x, y, CELL_SIZE, CELL_SIZE), 3)

        # Draw game over overlay
        if self.game_over:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))

            game_over_text = self.font_large.render("Game Over!", True, WHITE)
            text_rect = game_over_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 30))
            self.screen.blit(game_over_text, text_rect)

            final_score_text = self.font_medium.render(f"Final Score: {self.score}", True, GOLD)
            score_rect = final_score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
            self.screen.blit(final_score_text, score_rect)

            restart_text = self.font_small.render("Press R to restart or ESC to quit", True, WHITE)
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 60))
            self.screen.blit(restart_text, restart_rect)

    def run(self):
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset_game()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)

            self.update()
            self.draw_board()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = ColorLines()
    game.run()