"""
Qix - Classic Arcade Game Implementation
=========================================
The player moves along the edges of a rectangular playfield, drawing lines
to claim territory while avoiding the Qix (a wandering line creature) and
other hazards (sparx that patrol claimed edges).

Controls:
- Arrow keys / WASD to move
- Hold SPACE to draw a line into unclaimed territory
- When releasing SPACE or hitting a boundary, the line is committed
- Claim at least 75% of the area to win
"""

import pygame
import random
import math
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Playfield (the area where the game takes place)
FIELD_LEFT = 40
FIELD_TOP = 40
FIELD_RIGHT = SCREEN_WIDTH - 40
FIELD_BOTTOM = SCREEN_HEIGHT - 40
FIELD_WIDTH = FIELD_RIGHT - FIELD_LEFT
FIELD_HEIGHT = FIELD_BOTTOM - FIELD_TOP
GRID_SIZE = 10  # movement step in pixels

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (40, 40, 40)
ORANGE = (255, 165, 0)
PURPLE = (160, 32, 240)

# Colors for claimed territory (cycling)
TERRITORY_COLORS = [
    (50, 50, 150),
    (60, 60, 160),
    (70, 70, 170),
    (80, 80, 180),
]

# Game states
STATE_TITLE = 0
STATE_PLAYING = 1
STATE_WIN = 2
STATE_LOSE = 3

# Qix behavior
QIX_MIN_LENGTH = 20
QIX_MAX_LENGTH = 40
QIX_MOVE_INTERVAL = 3  # frames between Qix movements

# Sparx (enemies that patrol claimed edges)
SPARX_SPEED = 2
SPARX_COUNT = 2

# Win condition: percentage of area to claim
WIN_PERCENTAGE = 75

# Invulnerability frames after start/respawn
INVULNERABLE_FRAMES = 60  # 1 second at 60 FPS

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def snap_to_grid(value):
    """Snap a coordinate to the nearest grid point."""
    return round(value / GRID_SIZE) * GRID_SIZE


def point_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def lerp_color(c1, c2, t):
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


# ---------------------------------------------------------------------------
# Qix class - the wandering line creature
# ---------------------------------------------------------------------------

class Qix:
    """The Qix is a connected line of segments that wanders around the
    unclaimed area.  The player must avoid touching it."""

    def __init__(self, field_rect):
        self.field_rect = field_rect
        self.segments = []  # list of (x, y) points
        self.direction = random.choice(["up", "down", "left", "right"])
        self.move_counter = 0
        self._init_random_path()

    def _init_random_path(self):
        """Create an initial random path for the Qix, well inside the field."""
        margin = GRID_SIZE * 5
        cx = random.randint(
            self.field_rect.left + margin,
            self.field_rect.right - margin
        )
        cy = random.randint(
            self.field_rect.top + margin,
            self.field_rect.bottom - margin
        )
        cx = snap_to_grid(cx)
        cy = snap_to_grid(cy)

        length = random.randint(QIX_MIN_LENGTH, QIX_MAX_LENGTH)
        self.segments = [(cx, cy)]
        for _ in range(length):
            self._extend_random()
        while len(self.segments) < QIX_MIN_LENGTH:
            self._extend_random()

    def _extend_random(self):
        """Extend the Qix by one segment in a random valid direction."""
        head = self.segments[-1]
        directions = [
            (0, -GRID_SIZE),
            (0, GRID_SIZE),
            (-GRID_SIZE, 0),
            (GRID_SIZE, 0),
        ]
        valid = []
        for dx, dy in directions:
            nx, ny = head[0] + dx, head[1] + dy
            if self._in_unclaimed_area(nx, ny):
                valid.append((dx, dy))
        if not valid:
            return
        dx, dy = random.choice(valid)
        self.segments.append((head[0] + dx, head[1] + dy))

    def _in_unclaimed_area(self, x, y):
        """Check if a point is within the field interior (not on the boundary)."""
        return (self.field_rect.left + GRID_SIZE <= x <= self.field_rect.right - GRID_SIZE and
                self.field_rect.top + GRID_SIZE <= y <= self.field_rect.bottom - GRID_SIZE)

    def update(self, claimed_mask, field_rect):
        """Move the Qix. It tries to stay in unclaimed territory."""
        if not self.segments:
            return

        self.move_counter += 1
        if self.move_counter < QIX_MOVE_INTERVAL:
            return
        self.move_counter = 0

        head = self.segments[-1]
        dx, dy = self._direction_vector(self.direction)
        new_head = (head[0] + dx, head[1] + dy)

        in_area = self._in_unclaimed_area(new_head[0], new_head[1])
        is_claimed = self._is_claimed(new_head, claimed_mask, field_rect)
        if in_area and not is_claimed:
            self.segments.append(new_head)
        else:
            self._change_direction(claimed_mask, field_rect)

        while len(self.segments) > QIX_MAX_LENGTH:
            self.segments.pop(0)

    def _direction_vector(self, direction):
        return {
            "up": (0, -GRID_SIZE),
            "down": (0, GRID_SIZE),
            "left": (-GRID_SIZE, 0),
            "right": (GRID_SIZE, 0),
        }.get(direction, (0, 0))

    def _change_direction(self, claimed_mask, field_rect):
        """Change to a random valid direction."""
        head = self.segments[-1]
        directions = ["up", "down", "left", "right"]
        random.shuffle(directions)
        for d in directions:
            dx, dy = self._direction_vector(d)
            nx, ny = head[0] + dx, head[1] + dy
            if (self._in_unclaimed_area(nx, ny) and
                    not self._is_claimed((nx, ny), claimed_mask, field_rect)):
                self.direction = d
                self.segments.append((nx, ny))
                return
        reverse_map = {"up": "down", "down": "up", "left": "right", "right": "left"}
        self.direction = reverse_map.get(self.direction, "up")

    def _is_claimed(self, point, claimed_mask, field_rect):
        """Check if a point is in claimed territory."""
        if claimed_mask is None:
            return False
        px, py = point
        mx = int((px - field_rect.left) / GRID_SIZE)
        my = int((py - field_rect.top) / GRID_SIZE)
        if 0 <= mx < claimed_mask.get_width() and 0 <= my < claimed_mask.get_height():
            return claimed_mask.get_at((mx, my))[0] > 0
        return False

    def check_collision(self, point):
        """Check if a point collides with any Qix segment."""
        for seg in self.segments:
            if point_distance(point, seg) < GRID_SIZE // 2:
                return True
        return False

    def draw(self, screen):
        """Draw the Qix as a glowing line."""
        if len(self.segments) < 2:
            return
        for width in (6, 4, 2):
            color = lerp_color(RED, YELLOW, width / 6)
            pygame.draw.lines(screen, color, False, self.segments, width)
        pygame.draw.lines(screen, YELLOW, False, self.segments, 2)


# ---------------------------------------------------------------------------
# Sparx class - enemies that patrol claimed edges
# ---------------------------------------------------------------------------

class Sparx:
    """Sparx patrol along the edges of claimed territory."""

    def __init__(self, field_rect, edge_points):
        self.field_rect = field_rect
        self.edge_points = edge_points
        # Start at a random position along the edge, not at index 0
        if edge_points:
            self.index = random.randint(0, len(edge_points) - 1)
        else:
            self.index = 0
        self.direction = random.choice([-1, 1])
        self.speed = SPARX_SPEED
        self.progress = random.random()  # random starting offset

    def update(self, edge_points):
        """Move along the edge."""
        self.edge_points = edge_points
        if len(self.edge_points) < 2:
            return

        self.progress += self.speed * self.direction / GRID_SIZE

        if self.progress >= 1.0:
            self.progress = 0.0
            self.index = (self.index + 1) % len(self.edge_points)
        elif self.progress < 0.0:
            self.progress = 1.0
            self.index = (self.index - 1) % len(self.edge_points)

    def get_position(self):
        """Get current interpolated position."""
        if len(self.edge_points) < 2:
            return (0, 0)
        p1 = self.edge_points[self.index]
        p2 = self.edge_points[(self.index + 1) % len(self.edge_points)]
        t = max(0.0, min(1.0, self.progress))
        return (
            int(p1[0] + (p2[0] - p1[0]) * t),
            int(p1[1] + (p2[1] - p1[1]) * t),
        )

    def check_collision(self, point):
        """Check if a point collides with this sparx."""
        pos = self.get_position()
        return point_distance(point, pos) < GRID_SIZE // 2

    def draw(self, screen):
        """Draw the sparx as a small bright diamond."""
        pos = self.get_position()
        size = 6
        points = [
            (pos[0], pos[1] - size),
            (pos[0] + size, pos[1]),
            (pos[0], pos[1] + size),
            (pos[0] - size, pos[1]),
        ]
        pygame.draw.polygon(screen, CYAN, points)
        pygame.draw.polygon(screen, WHITE, points, 1)


# ---------------------------------------------------------------------------
# Player class
# ---------------------------------------------------------------------------

class Player:
    """The player moves along the edges of claimed territory and can draw
    lines into unclaimed territory to claim more area."""

    def __init__(self, field_rect):
        self.field_rect = field_rect
        # Start at the top-left corner of the field
        self.x = field_rect.left
        self.y = field_rect.top
        self.drawing = False
        self.draw_line = []
        self.speed = GRID_SIZE
        self.alive = True
        self.invulnerable_frames = INVULNERABLE_FRAMES

    def update(self, keys, claimed_mask, qix, sparxes):
        """Update player position based on input."""
        if not self.alive:
            return

        # Count down invulnerability
        if self.invulnerable_frames > 0:
            self.invulnerable_frames -= 1

        dx, dy = 0, 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx = -self.speed
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx = self.speed
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            dy = -self.speed
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy = self.speed

        space_pressed = keys[pygame.K_SPACE]

        if dx != 0 or dy != 0:
            new_x = self.x + dx
            new_y = self.y + dy

            new_x = max(self.field_rect.left, min(self.field_rect.right, new_x))
            new_y = max(self.field_rect.top, min(self.field_rect.bottom, new_y))

            new_x = snap_to_grid(new_x)
            new_y = snap_to_grid(new_y)

            if space_pressed:
                if not self._is_claimed((new_x, new_y), claimed_mask):
                    if not self.drawing:
                        self.drawing = True
                        self.draw_line = [(self.x, self.y)]
                    self.draw_line.append((new_x, new_y))
                    self.x, self.y = new_x, new_y
                else:
                    if self.drawing and len(self.draw_line) > 1:
                        self._commit_line(claimed_mask)
                    self.drawing = False
                    self.draw_line = []
                    self.x, self.y = new_x, new_y
            else:
                if self._is_claimed((new_x, new_y), claimed_mask) or self._is_on_boundary(new_x, new_y):
                    if self.drawing and len(self.draw_line) > 1:
                        self._commit_line(claimed_mask)
                    self.drawing = False
                    self.draw_line = []
                    self.x, self.y = new_x, new_y

        # Check collision with Qix (skip if invulnerable)
        if self.invulnerable_frames <= 0 and qix and qix.check_collision((self.x, self.y)):
            self.alive = False

        # Check collision with sparxes (skip if invulnerable)
        if self.invulnerable_frames <= 0:
            for sparx in sparxes:
                if sparx.check_collision((self.x, self.y)):
                    self.alive = False

    def _is_claimed(self, point, claimed_mask):
        """Check if a point is in claimed territory."""
        if claimed_mask is None:
            return False
        px, py = point
        mx = int((px - self.field_rect.left) / GRID_SIZE)
        my = int((py - self.field_rect.top) / GRID_SIZE)
        if 0 <= mx < claimed_mask.get_width() and 0 <= my < claimed_mask.get_height():
            return claimed_mask.get_at((mx, my))[0] > 0
        return False

    def _is_on_boundary(self, x, y):
        """Check if a point is on the outer boundary of the field."""
        return (x == self.field_rect.left or x == self.field_rect.right or
                y == self.field_rect.top or y == self.field_rect.bottom)

    def _commit_line(self, claimed_mask):
        """Commit the drawn line and claim the smaller enclosed area."""
        if len(self.draw_line) < 2:
            return

        for point in self.draw_line:
            px, py = point
            mx = int((px - self.field_rect.left) / GRID_SIZE)
            my = int((py - self.field_rect.top) / GRID_SIZE)
            if 0 <= mx < claimed_mask.get_width() and 0 <= my < claimed_mask.get_height():
                claimed_mask.set_at((mx, my), (255, 255, 255, 255))

        self._flood_fill_claim(claimed_mask)

    def _flood_fill_claim(self, claimed_mask):
        """Use flood fill to claim the smaller area created by the draw line."""
        if len(self.draw_line) < 2:
            return

        seeds = []
        for i in range(len(self.draw_line) - 1):
            p1 = self.draw_line[i]
            p2 = self.draw_line[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx != 0:
                seeds.append((p1[0], p1[1] - GRID_SIZE))
                seeds.append((p1[0], p1[1] + GRID_SIZE))
            if dy != 0:
                seeds.append((p1[0] - GRID_SIZE, p1[1]))
                seeds.append((p1[0] + GRID_SIZE, p1[1]))

        best_fill = []

        for seed in seeds:
            if not self._is_claimed(seed, claimed_mask):
                fill = self._flood_fill(seed, claimed_mask)
                if fill and (not best_fill or len(fill) < len(best_fill)):
                    best_fill = fill

        total_cells = (FIELD_WIDTH // GRID_SIZE + 1) * (FIELD_HEIGHT // GRID_SIZE + 1)
        if best_fill and len(best_fill) < total_cells // 2:
            for point in best_fill:
                px, py = point
                mx = int((px - self.field_rect.left) / GRID_SIZE)
                my = int((py - self.field_rect.top) / GRID_SIZE)
                if 0 <= mx < claimed_mask.get_width() and 0 <= my < claimed_mask.get_height():
                    claimed_mask.set_at((mx, my), (255, 255, 255, 255))

    def _flood_fill(self, start_point, claimed_mask):
        """Flood fill from a start point, returning all connected unclaimed points."""
        if self._is_claimed(start_point, claimed_mask):
            return []

        width = claimed_mask.get_width()
        height = claimed_mask.get_height()

        sx = int((start_point[0] - self.field_rect.left) / GRID_SIZE)
        sy = int((start_point[1] - self.field_rect.top) / GRID_SIZE)

        if not (0 <= sx < width and 0 <= sy < height):
            return []

        visited = set()
        queue = [(sx, sy)]
        result = []

        while queue and len(result) < 10000:
            cx, cy = queue.pop(0)
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            px = self.field_rect.left + cx * GRID_SIZE
            py = self.field_rect.top + cy * GRID_SIZE
            result.append((px, py))

            for nx, ny in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]:
                if 0 <= nx < width and 0 <= ny < height:
                    if (nx, ny) not in visited:
                        if claimed_mask.get_at((nx, ny))[0] == 0:
                            queue.append((nx, ny))

        return result

    def draw(self, screen):
        """Draw the player."""
        if not self.alive:
            return

        # Blink when invulnerable
        if self.invulnerable_frames > 0 and (self.invulnerable_frames // 5) % 2 == 0:
            return  # Skip drawing every other 5 frames to create blink effect

        rect = pygame.Rect(self.x - 5, self.y - 5, 10, 10)
        pygame.draw.rect(screen, WHITE, rect)
        pygame.draw.rect(screen, CYAN, rect, 2)

        if self.drawing and len(self.draw_line) > 1:
            pygame.draw.lines(screen, YELLOW, False, self.draw_line, 3)
            for i in range(0, len(self.draw_line) - 1, 2):
                pygame.draw.line(screen, WHITE, self.draw_line[i], self.draw_line[i + 1], 1)


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class Game:
    """Main game class managing all game objects and state."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Qix")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        self.field_rect = pygame.Rect(FIELD_LEFT, FIELD_TOP,
                                       FIELD_RIGHT - FIELD_LEFT,
                                       FIELD_BOTTOM - FIELD_TOP)

        mask_width = FIELD_WIDTH // GRID_SIZE + 1
        mask_height = FIELD_HEIGHT // GRID_SIZE + 1
        self.claimed_mask = pygame.Surface((mask_width, mask_height), pygame.SRCALPHA)
        self.claimed_mask.fill((0, 0, 0, 0))

        self._claim_boundary()

        self.state = STATE_TITLE
        self.player = None
        self.qix = None
        self.sparxes = []
        self.edge_points = []

        self.claimed_percentage = 0.0
        self.frame_count = 0
        self.color_cycle = 0

        self.title_qix_segments = []

    def _init_game_objects(self):
        """Initialize game objects for a new game."""
        self.player = Player(self.field_rect)
        self.qix = Qix(self.field_rect)
        self.sparxes = []
        self.edge_points = self._get_edge_points()
        for _ in range(SPARX_COUNT):
            self.sparxes.append(Sparx(self.field_rect, self.edge_points))
        self.claimed_percentage = 0.0
        self.frame_count = 0

    def _claim_boundary(self):
        """Claim the outer boundary of the field."""
        mask_width = self.claimed_mask.get_width()
        mask_height = self.claimed_mask.get_height()
        for x in range(mask_width):
            self.claimed_mask.set_at((x, 0), (255, 255, 255, 255))
            self.claimed_mask.set_at((x, mask_height - 1), (255, 255, 255, 255))
        for y in range(mask_height):
            self.claimed_mask.set_at((0, y), (255, 255, 255, 255))
            self.claimed_mask.set_at((mask_width - 1, y), (255, 255, 255, 255))

    def _get_edge_points(self):
        """Get all points on the boundary of claimed territory."""
        points = []
        mask_width = self.claimed_mask.get_width()
        mask_height = self.claimed_mask.get_height()

        for x in range(mask_width):
            if self.claimed_mask.get_at((x, 0))[0] > 0:
                px = self.field_rect.left + x * GRID_SIZE
                py = self.field_rect.top
                points.append((px, py))

        for y in range(1, mask_height - 1):
            if self.claimed_mask.get_at((mask_width - 1, y))[0] > 0:
                px = self.field_rect.right
                py = self.field_rect.top + y * GRID_SIZE
                points.append((px, py))

        for x in range(mask_width - 1, -1, -1):
            if self.claimed_mask.get_at((x, mask_height - 1))[0] > 0:
                px = self.field_rect.left + x * GRID_SIZE
                py = self.field_rect.bottom
                points.append((px, py))

        for y in range(mask_height - 2, 0, -1):
            if self.claimed_mask.get_at((0, y))[0] > 0:
                px = self.field_rect.left
                py = self.field_rect.top + y * GRID_SIZE
                points.append((px, py))

        return points

    def _calculate_claimed_percentage(self):
        """Calculate the percentage of the field that has been claimed."""
        total = 0
        claimed = 0
        mask_width = self.claimed_mask.get_width()
        mask_height = self.claimed_mask.get_height()
        for x in range(mask_width):
            for y in range(mask_height):
                total += 1
                if self.claimed_mask.get_at((x, y))[0] > 0:
                    claimed += 1
        if total == 0:
            return 0.0
        return (claimed / total) * 100

    def reset(self):
        """Reset the game for a new round."""
        self.claimed_mask.fill((0, 0, 0, 0))
        self._claim_boundary()
        self._init_game_objects()

    def update(self):
        """Update game state."""
        if self.state == STATE_TITLE:
            self.color_cycle = (self.color_cycle + 1) % (len(TERRITORY_COLORS) * 10)
            return

        if self.state != STATE_PLAYING:
            return

        self.frame_count += 1
        keys = pygame.key.get_pressed()

        self.player.update(keys, self.claimed_mask, self.qix, self.sparxes)

        if not self.player.alive:
            self.state = STATE_LOSE
            return

        self.qix.update(self.claimed_mask, self.field_rect)

        self.edge_points = self._get_edge_points()
        for sparx in self.sparxes:
            sparx.update(self.edge_points)

        self.claimed_percentage = self._calculate_claimed_percentage()

        if self.claimed_percentage >= WIN_PERCENTAGE:
            self.state = STATE_WIN

        self.color_cycle = (self.color_cycle + 1) % (len(TERRITORY_COLORS) * 10)

    def draw(self):
        """Draw everything."""
        self.screen.fill(BLACK)

        if self.state == STATE_TITLE:
            self._draw_title()
        elif self.state == STATE_PLAYING:
            self._draw_game()
        elif self.state == STATE_WIN:
            self._draw_game()
            self._draw_overlay("YOU WIN!", GREEN)
        elif self.state == STATE_LOSE:
            self._draw_game()
            self._draw_overlay("GAME OVER", RED)

        pygame.display.flip()

    def _draw_title(self):
        """Draw the title screen."""
        pygame.draw.rect(self.screen, GRAY, self.field_rect, 2)

        if not self.title_qix_segments:
            cx = self.field_rect.centerx
            cy = self.field_rect.centery
            self.title_qix_segments = [(cx, cy)]
            for i in range(30):
                angle = i * 0.5
                x = cx + int(math.cos(angle) * 80)
                y = cy + int(math.sin(angle * 1.3) * 60)
                self.title_qix_segments.append((x, y))

        if len(self.title_qix_segments) >= 2:
            for width in (6, 4, 2):
                color = lerp_color(RED, YELLOW, width / 6)
                pygame.draw.lines(self.screen, color, False, self.title_qix_segments, width)
            pygame.draw.lines(self.screen, YELLOW, False, self.title_qix_segments, 2)

        title = self.font_large.render("QIX", True, CYAN)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80))
        self.screen.blit(title, title_rect)

        instructions = [
            "Arrow Keys / WASD - Move",
            "SPACE - Draw line into unclaimed area",
            "Avoid the QIX and SPARX!",
            f"Claim {WIN_PERCENTAGE}% of the area to win!",
            "",
            "Press SPACE to start",
        ]
        y_offset = SCREEN_HEIGHT // 2 - 20
        for line in instructions:
            text = self.font_small.render(line, True, WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
            self.screen.blit(text, text_rect)
            y_offset += 30

    def _draw_game(self):
        """Draw the game field and all objects."""
        self._draw_claimed_territory()

        pygame.draw.rect(self.screen, WHITE, self.field_rect, 2)

        if self.qix:
            self.qix.draw(self.screen)

        for sparx in self.sparxes:
            sparx.draw(self.screen)

        if self.player:
            self.player.draw(self.screen)

        self._draw_hud()

    def _draw_claimed_territory(self):
        """Draw the claimed territory with a nice pattern."""
        mask_width = self.claimed_mask.get_width()
        mask_height = self.claimed_mask.get_height()
        color_idx = (self.color_cycle // 10) % len(TERRITORY_COLORS)
        base_color = TERRITORY_COLORS[color_idx]

        for x in range(mask_width):
            for y in range(mask_height):
                if self.claimed_mask.get_at((x, y))[0] > 0:
                    px = self.field_rect.left + x * GRID_SIZE
                    py = self.field_rect.top + y * GRID_SIZE
                    variation = ((x + y) % 3) * 10
                    color = (
                        min(255, base_color[0] + variation),
                        min(255, base_color[1] + variation),
                        min(255, base_color[2] + variation),
                    )
                    rect = pygame.Rect(px, py, GRID_SIZE, GRID_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)

    def _draw_hud(self):
        """Draw the heads-up display."""
        text = self.font_small.render(
            f"Claimed: {self.claimed_percentage:.1f}%  Target: {WIN_PERCENTAGE}%",
            True, WHITE)
        self.screen.blit(text, (10, 10))

        status = "ALIVE" if (self.player and self.player.alive) else "DEAD"
        color = GREEN if (self.player and self.player.alive) else RED
        status_text = self.font_small.render(f"Status: {status}", True, color)
        self.screen.blit(status_text, (SCREEN_WIDTH - 150, 10))

    def _draw_overlay(self, message, color):
        """Draw a semi-transparent overlay with a message."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))

        text = self.font_large.render(message, True, color)
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
        self.screen.blit(text, text_rect)

        restart_text = self.font_small.render("Press SPACE to restart", True, WHITE)
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
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
                    elif event.key == pygame.K_SPACE:
                        if self.state == STATE_TITLE:
                            self._init_game_objects()
                            self.state = STATE_PLAYING
                        elif self.state in (STATE_WIN, STATE_LOSE):
                            self.reset()
                            self.state = STATE_PLAYING

            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()