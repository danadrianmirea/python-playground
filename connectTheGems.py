"""
Connect The Gems - A match puzzle game
Click and drag to connect 2 or more gems of the same color.
Release to remove them and score points.
"""

import pygame
import random
import sys
import math

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 8
GEM_SIZE = 60
HEADER_HEIGHT = 80
BOARD_PIXEL_SIZE = BOARD_SIZE * GEM_SIZE
WINDOW_WIDTH = BOARD_PIXEL_SIZE
WINDOW_HEIGHT = BOARD_PIXEL_SIZE + HEADER_HEIGHT
FPS = 60
MIN_CONNECT = 2

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
LIGHT_GRAY = (100, 100, 100)
DARK_GRAY = (20, 20, 20)
GOLD = (255, 215, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
PURPLE = (200, 50, 255)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
PINK = (255, 100, 200)
YELLOW = (255, 255, 50)

# Gem types with colors and shapes
GEM_TYPES = [
    {"color": RED, "symbol": "♦", "name": "Ruby"},
    {"color": BLUE, "symbol": "●", "name": "Sapphire"},
    {"color": GREEN, "symbol": "▲", "name": "Emerald"},
    {"color": PURPLE, "symbol": "★", "name": "Amethyst"},
    {"color": ORANGE, "symbol": "■", "name": "Topaz"},
    {"color": CYAN, "symbol": "◆", "name": "Diamond"},
]

# Animation states
ANIM_NONE = 0
ANIM_FALL = 1
ANIM_DESTROY = 2


class Gem:
    """Represents a single gem on the board."""

    def __init__(self, gem_type, row, col):
        self.type = gem_type
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
        self.anim_duration = 10
        self.marked_for_destroy = False

    def get_pixel_pos(self):
        """Get the pixel position of the gem."""
        x = self.col * GEM_SIZE + GEM_SIZE // 2 + self.anim_offset_x
        y = self.row * GEM_SIZE + GEM_SIZE // 2 + HEADER_HEIGHT + self.anim_offset_y
        return x, y

    def update_animation(self):
        """Update animation state."""
        if self.anim_state == ANIM_NONE:
            return True

        self.anim_progress += 1
        t = self.anim_progress / self.anim_duration

        if self.anim_state == ANIM_FALL:
            if t >= 1:
                self.row = self.target_row
                self.anim_offset_y = 0
                self.anim_state = ANIM_NONE
                return True
            # Accelerating fall
            t = t * t
            dy = (self.target_row - self.row) * GEM_SIZE
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

        return True

    def draw(self, screen, font, highlight=False):
        """Draw the gem on the screen."""
        if self.anim_alpha <= 0:
            return

        x, y = self.get_pixel_pos()
        gem_data = GEM_TYPES[self.type]

        # Create a surface for the gem with alpha
        gem_surface = pygame.Surface((GEM_SIZE, GEM_SIZE), pygame.SRCALPHA)

        # Draw gem background (circle)
        radius = int(GEM_SIZE * 0.4 * self.anim_scale)
        center = (GEM_SIZE // 2, GEM_SIZE // 2)

        # Outer circle
        color = gem_data["color"]
        if highlight:
            color = tuple(min(255, c + 60) for c in color[:3])
        pygame.draw.circle(gem_surface, (*color[:3], self.anim_alpha),
                           center, radius)

        # Inner highlight
        highlight_color = tuple(min(255, c + 80) for c in gem_data["color"][:3])
        if highlight:
            highlight_color = tuple(min(255, c + 40) for c in highlight_color[:3])
        pygame.draw.circle(gem_surface, (*highlight_color, self.anim_alpha),
                           (center[0] - 3, center[1] - 3), radius - 5)

        # Draw symbol
        if self.anim_scale > 0.1:
            symbol = gem_data["symbol"]
            text = font.render(symbol, True, (*WHITE, self.anim_alpha))
            text_rect = text.get_rect(center=center)
            scaled_size = int(text.get_width() * self.anim_scale)
            if scaled_size > 0:
                scaled_text = pygame.transform.scale(text, (scaled_size, scaled_size))
                scaled_rect = scaled_text.get_rect(center=center)
                gem_surface.blit(scaled_text, scaled_rect)

        screen.blit(gem_surface, (x - GEM_SIZE // 2, y - GEM_SIZE // 2))


class ConnectTheGems:
    """Main game class."""

    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Connect The Gems")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("segoeuisymbol", 36)
        self.font_medium = pygame.font.SysFont("segoeuisymbol", 24)
        self.font_small = pygame.font.SysFont("segoeuisymbol", 18)
        self.font_gem = pygame.font.SysFont("segoeuisymbol", 28)

        self.reset_game()

    def reset_game(self):
        """Reset the game state."""
        self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.score = 0
        self.animating = False
        self.game_over = False
        self.combo_count = 0

        # Drag state
        self.dragging = False
        self.drag_path = []  # List of (row, col) tuples in order
        self.drag_color = None
        self.drag_start_gem = None

        # Initialize board with no initial matches
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                available = list(range(len(GEM_TYPES)))
                # Remove types that would create a match
                if row >= 2:
                    if self.board[row - 1][col].type == self.board[row - 2][col].type:
                        if self.board[row - 1][col].type in available:
                            available.remove(self.board[row - 1][col].type)
                if col >= 2:
                    if self.board[row][col - 1].type == self.board[row][col - 2].type:
                        if self.board[row][col - 1].type in available:
                            available.remove(self.board[row][col - 1].type)
                gem_type = random.choice(available)
                self.board[row][col] = Gem(gem_type, row, col)

    def get_gem_at(self, row, col):
        """Get the gem at the given board position."""
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row][col]
        return None

    def get_gem_at_pixel(self, x, y):
        """Get the gem at the given pixel position."""
        col = x // GEM_SIZE
        row = (y - HEADER_HEIGHT) // GEM_SIZE
        return self.get_gem_at(row, col)

    def is_adjacent_in_path(self, r1, c1, r2, c2):
        """Check if two positions are adjacent (including diagonally)."""
        return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1 and (r1 != r2 or c1 != c2)

    def find_matches(self):
        """Find all matches on the board. Returns a set of (row, col) tuples."""
        matches = set()

        # Check horizontal matches
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE - 2):
                gem1 = self.board[row][col]
                gem2 = self.board[row][col + 1]
                gem3 = self.board[row][col + 2]
                if gem1 and gem2 and gem3:
                    if gem1.type == gem2.type == gem3.type:
                        matches.add((row, col))
                        matches.add((row, col + 1))
                        matches.add((row, col + 2))

        # Check vertical matches
        for row in range(BOARD_SIZE - 2):
            for col in range(BOARD_SIZE):
                gem1 = self.board[row][col]
                gem2 = self.board[row + 1][col]
                gem3 = self.board[row + 2][col]
                if gem1 and gem2 and gem3:
                    if gem1.type == gem2.type == gem3.type:
                        matches.add((row, col))
                        matches.add((row + 1, col))
                        matches.add((row + 2, col))

        return matches

    def remove_gems(self, positions):
        """Mark gems at given positions for destruction."""
        for row, col in positions:
            gem = self.board[row][col]
            if gem and not gem.marked_for_destroy:
                gem.marked_for_destroy = True
                gem.anim_state = ANIM_DESTROY
                gem.anim_progress = 0

    def apply_gravity(self):
        """Make gems fall down to fill empty spaces."""
        moved = False
        for col in range(BOARD_SIZE):
            # Find empty spaces from bottom to top
            write_row = BOARD_SIZE - 1
            for row in range(BOARD_SIZE - 1, -1, -1):
                if self.board[row][col] is not None:
                    if row != write_row:
                        self.board[write_row][col] = self.board[row][col]
                        self.board[write_row][col].target_row = write_row
                        self.board[write_row][col].anim_state = ANIM_FALL
                        self.board[write_row][col].anim_progress = 0
                        self.board[row][col] = None
                        moved = True
                    write_row -= 1

            # Fill empty spaces at the top with new gems
            for row in range(write_row, -1, -1):
                gem_type = random.randint(0, len(GEM_TYPES) - 1)
                new_gem = Gem(gem_type, row, col)
                new_gem.target_row = row
                new_gem.anim_state = ANIM_FALL
                new_gem.anim_progress = 0
                # Start from above the board
                new_gem.row = row - (write_row - row + 1)
                self.board[row][col] = new_gem
                moved = True

        return moved

    def has_valid_moves(self):
        """Check if there are any valid moves remaining (at least 3 same-color gems adjacent)."""
        # Check for any group of 3+ same-color gems connected horizontally or vertically
        visited = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if visited[row][col] or self.board[row][col] is None:
                    continue

                gem_type = self.board[row][col].type
                # BFS to find connected group of same color
                group = []
                queue = [(row, col)]
                visited[row][col] = True

                while queue:
                    r, c = queue.pop(0)
                    group.append((r, c))

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            if not visited[nr][nc]:
                                neighbor = self.board[nr][nc]
                                if neighbor and neighbor.type == gem_type:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))

                if len(group) >= MIN_CONNECT:
                    return True

        return False

    def handle_mouse_down(self, pos):
        """Handle mouse button press."""
        if self.animating or self.game_over:
            return

        x, y = pos
        gem = self.get_gem_at_pixel(x, y)
        if gem is None:
            return

        self.dragging = True
        self.drag_path = [(gem.row, gem.col)]
        self.drag_color = gem.type
        self.drag_start_gem = gem

    def handle_mouse_move(self, pos):
        """Handle mouse movement while dragging."""
        if not self.dragging:
            return

        x, y = pos
        gem = self.get_gem_at_pixel(x, y)
        if gem is None:
            return

        r, c = gem.row, gem.col
        last_r, last_c = self.drag_path[-1]

        # Only add if it's a new cell adjacent to the last one
        if (r, c) != (last_r, last_c):
            if self.is_adjacent_in_path(last_r, last_c, r, c):
                # Check if this gem is the same color
                if gem.type == self.drag_color:
                    # Check if we're revisiting a cell - if so, trim the path back
                    if (r, c) in self.drag_path:
                        idx = self.drag_path.index((r, c))
                        self.drag_path = self.drag_path[:idx + 1]
                    else:
                        self.drag_path.append((r, c))
                # If different color, don't add but don't stop dragging
            # If not adjacent, ignore

    def handle_mouse_up(self, pos):
        """Handle mouse button release."""
        if not self.dragging:
            return

        self.dragging = False

        # Check if we connected enough gems
        if len(self.drag_path) >= MIN_CONNECT:
            # Verify all gems in the path still exist and have the same color
            valid = True
            for r, c in self.drag_path:
                gem = self.board[r][c]
                if gem is None or gem.type != self.drag_color or gem.marked_for_destroy:
                    valid = False
                    break

            if valid:
                # Score: more gems = higher multiplier
                self.combo_count += 1
                combo_bonus = self.combo_count * 10
                self.score += len(self.drag_path) * 15 + combo_bonus

                # Remove the connected gems
                self.remove_gems(self.drag_path)
                self.animating = True

        self.drag_path = []

    def update(self):
        """Update game state."""
        if not self.animating:
            return

        # Update all gem animations
        all_done = True
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                gem = self.board[row][col]
                if gem and gem.anim_state != ANIM_NONE:
                    if not gem.update_animation():
                        all_done = False

        if all_done:
            # Remove any gems that finished their destroy animation
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    gem = self.board[row][col]
                    if gem and gem.marked_for_destroy:
                        self.board[row][col] = None

            # Check for chain reaction matches from gravity
            matches = self.find_matches()
            if matches:
                self.combo_count += 1
                combo_bonus = self.combo_count * 10
                self.score += len(matches) * 10 + combo_bonus

                # Remove matched gems
                self.remove_gems(matches)

                # Wait for destroy animation
                self.animating = True
            else:
                # Apply gravity
                moved = self.apply_gravity()
                if moved:
                    self.animating = True
                else:
                    self.animating = False
                    self.combo_count = 0

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
        self.screen.blit(score_text, (20, 20))

        # Draw combo
        if self.combo_count > 1:
            combo_text = self.font_medium.render(f"Combo x{self.combo_count}!", True, GOLD)
            self.screen.blit(combo_text, (WINDOW_WIDTH - 150, 25))

        # Draw hint text
        hint_text = self.font_small.render("Drag to connect 2+ same-color gems", True, LIGHT_GRAY)
        self.screen.blit(hint_text, (WINDOW_WIDTH // 2 - hint_text.get_width() // 2, 55))

        # Draw board background
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                rect = pygame.Rect(col * GEM_SIZE, row * GEM_SIZE + HEADER_HEIGHT,
                                   GEM_SIZE, GEM_SIZE)
                color = GRAY if (row + col) % 2 == 0 else DARK_GRAY
                pygame.draw.rect(self.screen, color, rect)

        # Build set of highlighted positions
        highlight_set = set(self.drag_path)

        # Draw gems
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                gem = self.board[row][col]
                if gem and not gem.marked_for_destroy:
                    highlighted = (row, col) in highlight_set
                    gem.draw(self.screen, self.font_gem, highlight=highlighted)

        # Draw drag line
        if self.dragging and len(self.drag_path) >= 2:
            points = []
            for r, c in self.drag_path:
                px = c * GEM_SIZE + GEM_SIZE // 2
                py = r * GEM_SIZE + GEM_SIZE // 2 + HEADER_HEIGHT
                points.append((px, py))

            if len(points) >= 2:
                # Draw the connection line
                color = GEM_TYPES[self.drag_color]["color"]
                pygame.draw.lines(self.screen, color, False, points, 4)

                # Draw circles at each point
                for point in points:
                    pygame.draw.circle(self.screen, color, point, 6)
                    pygame.draw.circle(self.screen, WHITE, point, 6, 2)

            # Draw path length indicator
            if len(self.drag_path) >= MIN_CONNECT:
                last_point = points[-1]
                text = self.font_medium.render(str(len(self.drag_path)), True, GOLD)
                text_rect = text.get_rect(center=(last_point[0], last_point[1] - 25))
                self.screen.blit(text, text_rect)

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
                        self.handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.handle_mouse_up(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_move(event.pos)

            self.update()
            self.draw_board()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = ConnectTheGems()
    game.run()