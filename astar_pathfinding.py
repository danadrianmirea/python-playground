# astar_pathfinding.py - Simple A* pathfinding demo with Pygame
#
# Controls:
#   Left click / drag  -> place walls
#   Right click        -> find a path to the clicked tile and animate the player there
#   C                  -> clear all walls
#   R                  -> reset player to (1, 1)
#
# The player is a white square that starts at grid position (1, 1).
# Each right click runs A* from the player's current position to the clicked
# destination, then animates the player along the found route. The player's
# last destination becomes the starting point for the next search.

import pygame
import heapq
import sys

# Initialize pygame
pygame.init()

# Screen / grid dimensions
CELL_SIZE = 32
GRID_COLS = 25
GRID_ROWS = 20
SCREEN_WIDTH = GRID_COLS * CELL_SIZE
SCREEN_HEIGHT = GRID_ROWS * CELL_SIZE
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("A* Pathfinding Demo")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
LIGHT_GRAY = (70, 70, 70)
WALL_COLOR = (120, 120, 120)
PATH_COLOR = (60, 120, 220)
DEST_COLOR = (220, 60, 60)

# Fonts
font = pygame.font.Font(None, 24)

# Clock
clock = pygame.time.Clock()
FPS = 60

# Movement animation speed (cells per second)
MOVE_SPEED = 8.0

# Grid state
walls = set()                 # set of (col, row) tiles that are walls
player_pos = (1, 1)           # current player grid position
path = []                     # remaining path to walk (list of tiles)
destination = None            # last clicked destination (for drawing)
move_progress = 0.0           # 0..1 progress toward the next path tile


def in_bounds(col, row):
    return 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS


def cell_from_mouse(pos):
    """Convert a pixel position to a grid cell."""
    x, y = pos
    return x // CELL_SIZE, y // CELL_SIZE


def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal):
    """Return a list of tiles from start to goal (inclusive), or None."""
    if start == goal:
        return [start]
    if goal in walls or start in walls:
        return None

    open_heap = []
    counter = 0  # tie-breaker so heapq never compares tuples with equal f
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))

    came_from = {}
    g_score = {start: 0}
    closed = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruct path
            result = [current]
            while current in came_from:
                current = came_from[current]
                result.append(current)
            result.reverse()
            return result

        if current in closed:
            continue
        closed.add(current)

        cx, cy = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (cx + dx, cy + dy)
            if not in_bounds(*neighbor):
                continue
            if neighbor in walls:
                continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))

    return None


def draw_grid():
    screen.fill(BLACK)

    # Grid lines
    for col in range(GRID_COLS + 1):
        x = col * CELL_SIZE
        pygame.draw.line(screen, GRAY, (x, 0), (x, SCREEN_HEIGHT))
    for row in range(GRID_ROWS + 1):
        y = row * CELL_SIZE
        pygame.draw.line(screen, GRAY, (0, y), (SCREEN_WIDTH, y))

    # Walls
    for (col, row) in walls:
        rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, WALL_COLOR, rect)

    # Remaining path preview
    for (col, row) in path:
        rect = pygame.Rect(col * CELL_SIZE + 6, row * CELL_SIZE + 6,
                           CELL_SIZE - 12, CELL_SIZE - 12)
        pygame.draw.rect(screen, PATH_COLOR, rect)

    # Destination marker
    if destination is not None:
        col, row = destination
        rect = pygame.Rect(col * CELL_SIZE + 4, row * CELL_SIZE + 4,
                           CELL_SIZE - 8, CELL_SIZE - 8)
        pygame.draw.rect(screen, DEST_COLOR, rect, 2)

    # Player (white square), interpolated while moving
    px, py = player_pos
    draw_x = px * CELL_SIZE
    draw_y = py * CELL_SIZE
    if path:
        nx, ny = path[0]
        draw_x = (px + (nx - px) * move_progress) * CELL_SIZE
        draw_y = (py + (ny - py) * move_progress) * CELL_SIZE
    player_rect = pygame.Rect(int(draw_x) + 2, int(draw_y) + 2,
                              CELL_SIZE - 4, CELL_SIZE - 4)
    pygame.draw.rect(screen, WHITE, player_rect)

    # Help text
    help_text = font.render("LMB: walls  |  RMB: path  |  C: clear  |  R: reset", True, LIGHT_GRAY)
    screen.blit(help_text, (8, 8))


def main():
    global player_pos, path, destination, move_progress

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click: place a wall
                    cell = cell_from_mouse(event.pos)
                    if in_bounds(*cell) and cell != player_pos and cell not in walls:
                        walls.add(cell)
                        # Cancel any in-progress path since the map changed
                        path = []
                        move_progress = 0.0

                elif event.button == 3:  # Right click: single event, find path
                    cell = cell_from_mouse(event.pos)
                    if in_bounds(*cell) and cell not in walls:
                        destination = cell
                        found = astar(player_pos, cell)
                        if found and len(found) > 1:
                            path = found[1:]  # exclude current tile
                            move_progress = 0.0
                        else:
                            path = []

            elif event.type == pygame.MOUSEMOTION:
                # Dragging with left button held: place walls continuously
                if event.buttons[0]:
                    cell = cell_from_mouse(event.pos)
                    if in_bounds(*cell) and cell != player_pos and cell not in walls:
                        walls.add(cell)
                        path = []
                        move_progress = 0.0

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    walls.clear()
                    path = []
                    move_progress = 0.0
                elif event.key == pygame.K_r:
                    player_pos = (1, 1)
                    path = []
                    destination = None
                    move_progress = 0.0

        # Animate movement along the path
        if path:
            move_progress += MOVE_SPEED * dt
            while move_progress >= 1.0 and path:
                player_pos = path.pop(0)
                move_progress -= 1.0
            if not path:
                move_progress = 0.0

        draw_grid()
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()