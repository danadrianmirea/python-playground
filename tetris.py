# tetris.py - Classic Tetris game

import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
PLAY_WIDTH = 300   # 10 columns * 30px
PLAY_HEIGHT = 600  # 20 rows * 30px
SIDE_PANEL = 200
SCREEN_WIDTH = PLAY_WIDTH + SIDE_PANEL
SCREEN_HEIGHT = PLAY_HEIGHT
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris")

# Game constants
GRID_SIZE = 30
COLS = 10
ROWS = 20
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
LIGHT_GRAY = (100, 100, 100)
DARK_GRAY = (20, 20, 20)

# Tetromino colors
CYAN = (0, 255, 255)       # I
YELLOW = (255, 255, 0)     # O
PURPLE = (160, 50, 255)    # T
GREEN = (50, 200, 50)      # S
RED = (255, 50, 50)        # Z
BLUE = (50, 50, 255)       # J
ORANGE = (255, 165, 0)     # L

# Tetromino shapes
SHAPES = {
    'I': [[1, 1, 1, 1]],
    'O': [[1, 1],
          [1, 1]],
    'T': [[0, 1, 0],
          [1, 1, 1]],
    'S': [[0, 1, 1],
          [1, 1, 0]],
    'Z': [[1, 1, 0],
          [0, 1, 1]],
    'J': [[1, 0, 0],
          [1, 1, 1]],
    'L': [[0, 0, 1],
          [1, 1, 1]],
}

SHAPE_COLORS = {
    'I': CYAN,
    'O': YELLOW,
    'T': PURPLE,
    'S': GREEN,
    'Z': RED,
    'J': BLUE,
    'L': ORANGE,
}

SHAPE_NAMES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']

# Fonts
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)
title_font = pygame.font.Font(None, 48)
game_over_font = pygame.font.Font(None, 60)

# Clock
clock = pygame.time.Clock()


def rotate_matrix(matrix):
    """Rotate a 2D matrix 90 degrees clockwise."""
    return [list(row) for row in zip(*matrix[::-1])]


class Tetromino:
    """Represents a falling tetromino piece."""

    def __init__(self, shape_name):
        self.name = shape_name
        self.shape = [row[:] for row in SHAPES[shape_name]]
        self.color = SHAPE_COLORS[shape_name]
        self.row = 0
        self.col = COLS // 2 - len(self.shape[0]) // 2

    def rotate(self):
        """Rotate the piece clockwise."""
        return rotate_matrix(self.shape)

    def get_blocks(self):
        """Return list of (row, col) positions occupied by this piece."""
        blocks = []
        for r, row in enumerate(self.shape):
            for c, val in enumerate(row):
                if val:
                    blocks.append((self.row + r, self.col + c))
        return blocks


class Tetris:
    """Main game class."""

    def __init__(self):
        self.board = [[BLACK for _ in range(COLS)] for _ in range(ROWS)]
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.game_over = False
        self.bag = []
        self.drop_timer = 0
        self.drop_delay = 500  # ms
        self.lock_delay = 0
        self.lock_moves = 0
        self.soft_dropping = False

        # Auto-repeat (DAS) settings
        self.das_delay = 170  # ms before auto-repeat starts
        self.das_rate = 50    # ms between auto-repeat actions
        self.das_left_timer = 0
        self.das_right_timer = 0
        self.das_down_timer = 0
        self.das_left_active = False
        self.das_right_active = False
        self.das_down_active = False
        self.das_left_triggered = False
        self.das_right_triggered = False
        self.das_down_triggered = False

        self._spawn_piece()

    def _get_next_from_bag(self):
        """Get next piece using 7-bag randomizer."""
        if not self.bag:
            self.bag = SHAPE_NAMES[:]
            random.shuffle(self.bag)
        return self.bag.pop()

    def _spawn_piece(self):
        """Spawn the next piece."""
        if self.next_piece is None:
            self.next_piece = Tetromino(self._get_next_from_bag())
        self.current_piece = self.next_piece
        self.next_piece = Tetromino(self._get_next_from_bag())
        self.lock_delay = 0
        self.lock_moves = 0

        # Check if new piece collides immediately (game over)
        if self._check_collision(self.current_piece.get_blocks()):
            self.game_over = True

    def _check_collision(self, blocks):
        """Check if any blocks collide with walls or existing pieces."""
        for row, col in blocks:
            if col < 0 or col >= COLS or row >= ROWS:
                return True
            if row >= 0 and self.board[row][col] != BLACK:
                return True
        return False

    def _lock_piece(self):
        """Lock the current piece into the board."""
        for row, col in self.current_piece.get_blocks():
            if 0 <= row < ROWS and 0 <= col < COLS:
                self.board[row][col] = self.current_piece.color
        self._clear_lines()
        self._spawn_piece()

    def _clear_lines(self):
        """Clear completed lines and update score."""
        lines_to_clear = []
        for r in range(ROWS):
            if all(self.board[r][c] != BLACK for c in range(COLS)):
                lines_to_clear.append(r)

        if lines_to_clear:
            # Remove completed lines
            for r in lines_to_clear:
                del self.board[r]
                self.board.insert(0, [BLACK for _ in range(COLS)])

            # Update score
            count = len(lines_to_clear)
            points = [0, 100, 300, 500, 800]
            self.score += points[count] * self.level
            self.lines_cleared += count
            self.level = self.lines_cleared // 10 + 1
            self.drop_delay = max(100, 500 - (self.level - 1) * 40)

    def move_left(self):
        """Move piece left."""
        if self.current_piece and not self.game_over:
            self.current_piece.col -= 1
            if self._check_collision(self.current_piece.get_blocks()):
                self.current_piece.col += 1
            else:
                self.lock_moves = 0

    def move_right(self):
        """Move piece right."""
        if self.current_piece and not self.game_over:
            self.current_piece.col += 1
            if self._check_collision(self.current_piece.get_blocks()):
                self.current_piece.col -= 1
            else:
                self.lock_moves = 0

    def move_down(self):
        """Move piece down one row."""
        if self.current_piece and not self.game_over:
            self.current_piece.row += 1
            if self._check_collision(self.current_piece.get_blocks()):
                self.current_piece.row -= 1
                self._lock_piece()
                return False
            self.lock_moves = 0
            return True
        return False

    def rotate_piece(self):
        """Rotate the current piece."""
        if self.current_piece and not self.game_over:
            original_shape = self.current_piece.shape
            self.current_piece.shape = self.current_piece.rotate()
            if self._check_collision(self.current_piece.get_blocks()):
                # Wall kick - try shifting left/right
                for offset in [1, -1, 2, -2]:
                    self.current_piece.col += offset
                    if not self._check_collision(self.current_piece.get_blocks()):
                        self.lock_moves = 0
                        return
                    self.current_piece.col -= offset
                # Revert rotation
                self.current_piece.shape = original_shape
            else:
                self.lock_moves = 0

    def hard_drop(self):
        """Drop piece instantly to the bottom."""
        if self.current_piece and not self.game_over:
            while not self._check_collision(self.current_piece.get_blocks()):
                self.current_piece.row += 1
            self.current_piece.row -= 1
            self._lock_piece()

    def get_ghost_row(self):
        """Get the row where the piece would land if hard-dropped."""
        if not self.current_piece:
            return 0
        original_row = self.current_piece.row
        while not self._check_collision(self.current_piece.get_blocks()):
            self.current_piece.row += 1
        ghost_row = self.current_piece.row - 1
        self.current_piece.row = original_row
        return ghost_row

    def update(self, dt):
        """Update game state. Called every frame."""
        if self.game_over:
            return

        # Auto-repeat handling
        if self.das_left_active:
            self.das_left_timer += dt
            if not self.das_left_triggered:
                if self.das_left_timer >= self.das_delay:
                    self.das_left_triggered = True
                    self.das_left_timer = 0
                    self.move_left()
            else:
                if self.das_left_timer >= self.das_rate:
                    self.das_left_timer = 0
                    self.move_left()

        if self.das_right_active:
            self.das_right_timer += dt
            if not self.das_right_triggered:
                if self.das_right_timer >= self.das_delay:
                    self.das_right_triggered = True
                    self.das_right_timer = 0
                    self.move_right()
            else:
                if self.das_right_timer >= self.das_rate:
                    self.das_right_timer = 0
                    self.move_right()

        if self.das_down_active:
            self.das_down_timer += dt
            if not self.das_down_triggered:
                if self.das_down_timer >= self.das_delay:
                    self.das_down_triggered = True
                    self.das_down_timer = 0
                    self.move_down()
            else:
                if self.das_down_timer >= self.das_rate:
                    self.das_down_timer = 0
                    self.move_down()

        self.drop_timer += dt
        if self.drop_timer >= self.drop_delay:
            self.drop_timer = 0
            if not self.move_down():
                pass  # piece was locked

    def draw(self, surface):
        """Draw the entire game."""
        surface.fill(BLACK)

        # Draw play area border
        pygame.draw.rect(surface, LIGHT_GRAY, (0, 0, PLAY_WIDTH, PLAY_HEIGHT), 2)

        # Draw grid
        for r in range(ROWS):
            for c in range(COLS):
                color = self.board[r][c]
                if color != BLACK:
                    self._draw_block(surface, c, r, color)

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_row = self.get_ghost_row()
            for r, row in enumerate(self.current_piece.shape):
                for c, val in enumerate(row):
                    if val:
                        block_row = ghost_row + r
                        block_col = self.current_piece.col + c
                        if block_row >= 0:
                            rect = pygame.Rect(
                                block_col * GRID_SIZE + 1,
                                block_row * GRID_SIZE + 1,
                                GRID_SIZE - 2, GRID_SIZE - 2
                            )
                            pygame.draw.rect(surface, self.current_piece.color, rect, 1, border_radius=2)

        # Draw current piece
        if self.current_piece and not self.game_over:
            for r, row in enumerate(self.current_piece.shape):
                for c, val in enumerate(row):
                    if val:
                        block_row = self.current_piece.row + r
                        block_col = self.current_piece.col + c
                        if block_row >= 0:
                            self._draw_block(surface, block_col, block_row, self.current_piece.color)

        # Draw side panel
        self._draw_side_panel(surface)

        # Draw game over
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            surface.blit(overlay, (0, 0))

            game_over_text = game_over_font.render("GAME OVER", True, RED)
            surface.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 60))

            restart_text = font.render("Press SPACE to restart", True, WHITE)
            surface.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 10))

    def _draw_block(self, surface, col, row, color):
        """Draw a single block with 3D effect."""
        x = col * GRID_SIZE
        y = row * GRID_SIZE
        rect = pygame.Rect(x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2)
        pygame.draw.rect(surface, color, rect, border_radius=3)

        # Highlight (top-left)
        lighter = tuple(min(255, c + 60) for c in color)
        pygame.draw.line(surface, lighter, (x + 2, y + 2), (x + GRID_SIZE - 3, y + 2), 2)
        pygame.draw.line(surface, lighter, (x + 2, y + 2), (x + 2, y + GRID_SIZE - 3), 2)

        # Shadow (bottom-right)
        darker = tuple(max(0, c - 60) for c in color)
        pygame.draw.line(surface, darker, (x + GRID_SIZE - 3, y + 2), (x + GRID_SIZE - 3, y + GRID_SIZE - 3), 2)
        pygame.draw.line(surface, darker, (x + 2, y + GRID_SIZE - 3), (x + GRID_SIZE - 3, y + GRID_SIZE - 3), 2)

    def _draw_side_panel(self, surface):
        """Draw the side panel with score, level, and next piece."""
        panel_x = PLAY_WIDTH + 20

        # Title
        title_text = title_font.render("TETRIS", True, WHITE)
        surface.blit(title_text, (panel_x, 20))

        # Next piece
        next_label = font.render("Next:", True, WHITE)
        surface.blit(next_label, (panel_x, 100))

        if self.next_piece:
            shape = self.next_piece.shape
            color = self.next_piece.color
            offset_x = panel_x + 30
            offset_y = 140
            for r, row in enumerate(shape):
                for c, val in enumerate(row):
                    if val:
                        rect = pygame.Rect(
                            offset_x + c * 25 + 1,
                            offset_y + r * 25 + 1,
                            23, 23
                        )
                        pygame.draw.rect(surface, color, rect, border_radius=2)

        # Score
        score_label = font.render("Score:", True, WHITE)
        surface.blit(score_label, (panel_x, 250))
        score_text = font.render(str(self.score), True, YELLOW)
        surface.blit(score_text, (panel_x, 280))

        # Level
        level_label = font.render("Level:", True, WHITE)
        surface.blit(level_label, (panel_x, 330))
        level_text = font.render(str(self.level), True, YELLOW)
        surface.blit(level_text, (panel_x, 360))

        # Lines
        lines_label = font.render("Lines:", True, WHITE)
        surface.blit(lines_label, (panel_x, 410))
        lines_text = font.render(str(self.lines_cleared), True, YELLOW)
        surface.blit(lines_text, (panel_x, 440))

        # Controls
        controls_y = 500
        controls = [
            "Controls:",
            "← →  Move",
            "↑     Rotate",
            "↓     Soft drop",
            "Space  Hard drop",
            "WASD  Alt controls",
        ]
        for i, text in enumerate(controls):
            color = LIGHT_GRAY if i > 0 else WHITE
            ctrl_text = small_font.render(text, True, color)
            surface.blit(ctrl_text, (panel_x, controls_y + i * 22))


def main():
    """Main game loop."""
    game = Tetris()
    running = True

    while running:
        dt = clock.tick(FPS)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if game.game_over:
                    if event.key == pygame.K_SPACE:
                        game = Tetris()
                    continue

                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    game.move_left()
                    game.das_left_active = True
                    game.das_left_timer = 0
                    game.das_left_triggered = False
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    game.move_right()
                    game.das_right_active = True
                    game.das_right_timer = 0
                    game.das_right_triggered = False
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    game.soft_dropping = True
                    game.das_down_active = True
                    game.das_down_timer = 0
                    game.das_down_triggered = False
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    game.rotate_piece()
                elif event.key == pygame.K_SPACE:
                    game.hard_drop()
                    game.drop_timer = 0

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    game.das_left_active = False
                    game.das_left_triggered = False
                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    game.das_right_active = False
                    game.das_right_triggered = False
                if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    game.soft_dropping = False
                    game.das_down_active = False
                    game.das_down_triggered = False

        # Update
        if game.soft_dropping:
            game.drop_timer += dt * 5  # 5x faster when soft dropping
        game.update(dt)

        # Draw
        game.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()