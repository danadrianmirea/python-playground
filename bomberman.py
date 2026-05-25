# bomberman.py - Classic Bomberman game

import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 700
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Bomberman")

# Game constants
TILE_SIZE = 40
GRID_COLS = SCREEN_WIDTH // TILE_SIZE   # 20
GRID_ROWS = (SCREEN_HEIGHT - 100) // TILE_SIZE  # 15 (leave 100px for HUD)
PLAY_AREA_HEIGHT = GRID_ROWS * TILE_SIZE
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (60, 60, 60)
DARK_GRAY = (40, 40, 40)
BROWN = (139, 69, 19)
DARK_BROWN = (101, 67, 33)
DARK_RED = (180, 30, 30)
YELLOW = (255, 255, 0)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 50, 255)
ORANGE = (255, 165, 0)
PURPLE = (180, 50, 180)
CYAN = (0, 255, 255)
LIGHT_BROWN = (200, 160, 100)
HUD_BG = (20, 20, 30)

# Tile types
EMPTY = 0
WALL = 1
SOFT_BLOCK = 2
BOMB = 3
EXPLOSION = 4

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Fonts
font = pygame.font.Font(None, 28)
title_font = pygame.font.Font(None, 48)
hud_font = pygame.font.Font(None, 32)


class Player:
    def __init__(self, grid_x, grid_y, color, controls, player_id):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pixel_x = grid_x * TILE_SIZE + TILE_SIZE // 2
        self.pixel_y = grid_y * TILE_SIZE + TILE_SIZE // 2
        self.color = color
        self.controls = controls
        self.player_id = player_id
        self.radius = TILE_SIZE // 2 - 4
        self.speed = 3
        self.alive = True
        self.max_bombs = 1
        self.active_bombs = 0
        self.bomb_power = 1
        self.move_timer = 0
        self.move_delay = 4  # frames between tile moves

    def handle_input(self, keys):
        if not self.alive:
            return

        dx, dy = 0, 0
        if keys[self.controls['up']]:
            dy = -1
        elif keys[self.controls['down']]:
            dy = 1
        elif keys[self.controls['left']]:
            dx = -1
        elif keys[self.controls['right']]:
            dx = 1

        if dx != 0 or dy != 0:
            self.move_timer += 1
            if self.move_timer >= self.move_delay:
                self.move_timer = 0
                new_x = self.grid_x + dx
                new_y = self.grid_y + dy
                if can_move_to(new_x, new_y):
                    self.grid_x = new_x
                    self.grid_y = new_y
                    self.pixel_x = self.grid_x * TILE_SIZE + TILE_SIZE // 2
                    self.pixel_y = self.grid_y * TILE_SIZE + TILE_SIZE // 2
        else:
            self.move_timer = 0

    def place_bomb(self):
        if not self.alive:
            return None
        if self.active_bombs >= self.max_bombs:
            return None
        if grid[self.grid_y][self.grid_x] != EMPTY:
            return None
        # Check if there's already a bomb here
        for b in bombs:
            if b.grid_x == self.grid_x and b.grid_y == self.grid_y:
                return None
        self.active_bombs += 1
        return Bomb(self.grid_x, self.grid_y, self.bomb_power, self)

    def draw(self, surface):
        if not self.alive:
            return
        pygame.draw.circle(surface, self.color,
                           (self.pixel_x, self.pixel_y), self.radius)
        # Draw face
        eye_offset = 6
        pygame.draw.circle(surface, WHITE,
                           (self.pixel_x - 4, self.pixel_y - 3), 3)
        pygame.draw.circle(surface, WHITE,
                           (self.pixel_x + 4, self.pixel_y - 3), 3)
        pygame.draw.circle(surface, BLACK,
                           (self.pixel_x - 4, self.pixel_y - 3), 1)
        pygame.draw.circle(surface, BLACK,
                           (self.pixel_x + 4, self.pixel_y - 3), 1)


class Bomb:
    def __init__(self, grid_x, grid_y, power, owner):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.power = power
        self.owner = owner
        self.timer = 120  # 2 seconds at 60 FPS
        self.exploded = False
        self.pulse = 0

    def update(self):
        if self.exploded:
            return
        self.timer -= 1
        self.pulse += 0.1
        if self.timer <= 0:
            self.explode()

    def explode(self):
        self.exploded = True
        self.owner.active_bombs -= 1
        cells = [(self.grid_x, self.grid_y)]

        # Spread in 4 directions
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for i in range(1, self.power + 1):
                cx, cy = self.grid_x + dx * i, self.grid_y + dy * i
                if not (0 <= cx < GRID_COLS and 0 <= cy < GRID_ROWS):
                    break
                if grid[cy][cx] == WALL:
                    break
                cells.append((cx, cy))
                if grid[cy][cx] == SOFT_BLOCK:
                    grid[cy][cx] = EMPTY
                    break

        explosions.append(Explosion(cells))
        grid[self.grid_y][self.grid_x] = EMPTY

    def draw(self, surface):
        if self.exploded:
            return
        x = self.grid_x * TILE_SIZE + TILE_SIZE // 2
        y = self.grid_y * TILE_SIZE + TILE_SIZE // 2
        pulse_radius = 8 + int(3 * pygame.math.Vector2(1, 0).rotate(self.pulse * 360).x)
        pygame.draw.circle(surface, BLACK, (x, y), TILE_SIZE // 2 - 2)
        pygame.draw.circle(surface, RED, (x, y), TILE_SIZE // 2 - 4)
        # Fuse spark
        spark = int(4 + 3 * pygame.math.Vector2(1, 0).rotate(self.pulse * 720).x)
        pygame.draw.circle(surface, YELLOW, (x + 8, y - 8), 3 + spark // 4)
        # Timer text
        secs = max(1, self.timer // 60 + 1)
        text = font.render(str(secs), True, WHITE)
        text_rect = text.get_rect(center=(x, y))
        surface.blit(text, text_rect)


class Explosion:
    def __init__(self, cells):
        self.cells = cells
        self.timer = 20  # frames

    def update(self):
        self.timer -= 1
        return self.timer > 0

    def draw(self, surface):
        alpha = min(255, self.timer * 20)
        for cx, cy in self.cells:
            x = cx * TILE_SIZE
            y = cy * TILE_SIZE
            # Outer ring
            pygame.draw.rect(surface, (255, 100, 0, alpha),
                             (x + 2, y + 2, TILE_SIZE - 4, TILE_SIZE - 4))
            pygame.draw.rect(surface, (255, 200, 0, alpha),
                             (x + 6, y + 6, TILE_SIZE - 12, TILE_SIZE - 12))
            # Center
            pygame.draw.rect(surface, (255, 255, 200, alpha),
                             (x + 10, y + 10, TILE_SIZE - 20, TILE_SIZE - 20))


class Enemy:
    def __init__(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pixel_x = grid_x * TILE_SIZE + TILE_SIZE // 2
        self.pixel_y = grid_y * TILE_SIZE + TILE_SIZE // 2
        self.alive = True
        self.move_timer = 0
        self.move_delay = 20  # slower than player
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.stuck_counter = 0

    def update(self):
        if not self.alive:
            return

        self.move_timer += 1
        if self.move_timer >= self.move_delay:
            self.move_timer = 0

            # Try current direction
            dx, dy = self.direction
            new_x = self.grid_x + dx
            new_y = self.grid_y + dy

            if can_move_to(new_x, new_y):
                self.grid_x = new_x
                self.grid_y = new_y
                self.pixel_x = self.grid_x * TILE_SIZE + TILE_SIZE // 2
                self.pixel_y = self.grid_y * TILE_SIZE + TILE_SIZE // 2
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1
                # Pick a new random direction
                self.direction = random.choice([UP, DOWN, LEFT, RIGHT])

    def draw(self, surface):
        if not self.alive:
            return
        x, y = self.pixel_x, self.pixel_y
        r = TILE_SIZE // 2 - 4
        # Body
        pygame.draw.circle(surface, RED, (x, y), r)
        pygame.draw.circle(surface, DARK_RED, (x, y), r - 3)
        # Eyes (angry)
        pygame.draw.circle(surface, WHITE, (x - 5, y - 4), 4)
        pygame.draw.circle(surface, WHITE, (x + 5, y - 4), 4)
        pygame.draw.circle(surface, BLACK, (x - 5, y - 4), 2)
        pygame.draw.circle(surface, BLACK, (x + 5, y - 4), 2)
        # Angry eyebrows
        pygame.draw.line(surface, BLACK, (x - 9, y - 9), (x - 3, y - 7), 2)
        pygame.draw.line(surface, BLACK, (x + 9, y - 9), (x + 3, y - 7), 2)
        # Feet
        foot_offset = 6
        pygame.draw.rect(surface, DARK_RED,
                         (x - foot_offset - 3, y + r - 3, 6, 6))
        pygame.draw.rect(surface, DARK_RED,
                         (x + foot_offset - 3, y + r - 3, 6, 6))


class PowerUp:
    def __init__(self, grid_x, grid_y, type):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.type = type  # 'bomb' or 'power'
        self.pulse = 0

    def update(self):
        self.pulse += 0.05

    def draw(self, surface):
        x = self.grid_x * TILE_SIZE + TILE_SIZE // 2
        y = self.grid_y * TILE_SIZE + TILE_SIZE // 2
        r = TILE_SIZE // 2 - 6

        if self.type == 'bomb':
            color = ORANGE
            label = 'B'
        else:
            color = CYAN
            label = 'P'

        glow = int(3 * pygame.math.Vector2(1, 0).rotate(self.pulse * 360).x)
        pygame.draw.circle(surface, color, (x, y), r + glow)
        pygame.draw.circle(surface, BLACK, (x, y), r - 2)
        text = font.render(label, True, color)
        text_rect = text.get_rect(center=(x, y))
        surface.blit(text, text_rect)


# Global game state
grid = [[EMPTY for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
bombs = []
explosions = []
enemies = []
powerups = []
players = []
game_over = False
winner = None
level = 1
score = 0


def can_move_to(grid_x, grid_y):
    """Check if a tile is walkable."""
    if not (0 <= grid_x < GRID_COLS and 0 <= grid_y < GRID_ROWS):
        return False
    if grid[grid_y][grid_x] != EMPTY:
        return False
    # Check for bombs
    for b in bombs:
        if b.grid_x == grid_x and b.grid_y == grid_y and not b.exploded:
            return False
    return True


def generate_level(lvl):
    """Generate a new level layout."""
    global grid, enemies, powerups, bombs, explosions, players

    grid = [[EMPTY for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    bombs = []
    explosions = []
    enemies = []
    powerups = []

    # Place indestructible walls in a checkerboard pattern
    for y in range(GRID_ROWS):
        for x in range(GRID_COLS):
            if x % 2 == 1 and y % 2 == 1:
                grid[y][x] = WALL

    # Place border walls
    for x in range(GRID_COLS):
        grid[0][x] = WALL
        grid[GRID_ROWS - 1][x] = WALL
    for y in range(GRID_ROWS):
        grid[y][0] = WALL
        grid[y][GRID_COLS - 1] = WALL

    # Clear spawn areas for player 1 (top-left)
    spawn_zones = [(1, 1), (1, 2), (2, 1)]
    for sx, sy in spawn_zones:
        grid[sy][sx] = EMPTY

    # Clear spawn areas for player 2 (bottom-right) if 2 players
    spawn_zones2 = [(GRID_COLS - 2, GRID_ROWS - 2),
                    (GRID_COLS - 3, GRID_ROWS - 2),
                    (GRID_COLS - 2, GRID_ROWS - 3)]
    for sx, sy in spawn_zones2:
        grid[sy][sx] = EMPTY

    # Place soft blocks
    num_soft = 40 + lvl * 5
    placed = 0
    while placed < num_soft:
        x = random.randint(1, GRID_COLS - 2)
        y = random.randint(1, GRID_ROWS - 2)
        if grid[y][x] == EMPTY:
            grid[y][x] = SOFT_BLOCK
            placed += 1

    # Place enemies
    num_enemies = min(2 + lvl, 8)
    for _ in range(num_enemies):
        for _ in range(50):
            x = random.randint(2, GRID_COLS - 3)
            y = random.randint(2, GRID_ROWS - 3)
            if grid[y][x] == EMPTY and not (x <= 3 and y <= 3) and not (x >= GRID_COLS - 4 and y >= GRID_ROWS - 4):
                enemies.append(Enemy(x, y))
                break

    # Place power-ups (hidden under soft blocks)
    num_powerups = min(3 + lvl, 8)
    placed_pu = 0
    while placed_pu < num_powerups:
        x = random.randint(1, GRID_COLS - 2)
        y = random.randint(1, GRID_ROWS - 2)
        if grid[y][x] == SOFT_BLOCK:
            # Store powerup info - will appear when block is destroyed
            pu_type = random.choice(['bomb', 'power'])
            powerups.append(PowerUp(x, y, pu_type))
            placed_pu += 1


def check_collisions():
    """Check player-enemy and player-explosion collisions."""
    global game_over, winner, score

    # Player vs Enemy
    for player in players:
        if not player.alive:
            continue
        for enemy in enemies:
            if not enemy.alive:
                continue
            if player.grid_x == enemy.grid_x and player.grid_y == enemy.grid_y:
                player.alive = False

    # Player vs Explosion
    for player in players:
        if not player.alive:
            continue
        for exp in explosions:
            if (player.grid_x, player.grid_y) in exp.cells:
                player.alive = False

    # Enemy vs Explosion
    for enemy in enemies:
        if not enemy.alive:
            continue
        for exp in explosions:
            if (enemy.grid_x, enemy.grid_y) in exp.cells:
                enemy.alive = False
                score += 100

    # Player vs PowerUp
    for player in players:
        if not player.alive:
            continue
        for pu in powerups[:]:
            if player.grid_x == pu.grid_x and player.grid_y == pu.grid_y:
                if pu.type == 'bomb':
                    player.max_bombs += 1
                elif pu.type == 'power':
                    player.bomb_power += 1
                powerups.remove(pu)

    # Check win/lose conditions
    alive_players = [p for p in players if p.alive]
    alive_enemies = [e for e in enemies if e.alive]

    if len(alive_players) == 0:
        game_over = True
        winner = None
    elif len(alive_enemies) == 0:
        game_over = True
        if len(alive_players) == 1:
            winner = alive_players[0]
        else:
            winner = alive_players  # multiple winners


def draw_hud(surface):
    """Draw the HUD at the bottom."""
    hud_y = PLAY_AREA_HEIGHT
    pygame.draw.rect(surface, HUD_BG, (0, hud_y, SCREEN_WIDTH, 100))

    # Level
    level_text = hud_font.render(f"Level: {level}", True, WHITE)
    surface.blit(level_text, (20, hud_y + 10))

    # Score
    score_text = hud_font.render(f"Score: {score}", True, YELLOW)
    surface.blit(score_text, (20, hud_y + 45))

    # Player info
    for i, player in enumerate(players):
        px = 250 + i * 250
        color_text = hud_font.render(f"P{i+1}:", True, player.color)
        surface.blit(color_text, (px, hud_y + 10))

        bombs_text = hud_font.render(f"Bombs: {player.active_bombs}/{player.max_bombs}", True, ORANGE)
        surface.blit(bombs_text, (px + 50, hud_y + 10))

        power_text = hud_font.render(f"Power: {player.bomb_power}", True, CYAN)
        surface.blit(power_text, (px + 50, hud_y + 45))

    # Controls hint
    controls_text = font.render("P1: WASD + Space | P2: Arrows + Enter", True, GRAY)
    surface.blit(controls_text, (SCREEN_WIDTH - 380, hud_y + 75))


def draw_game_over(surface):
    """Draw game over screen."""
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill(BLACK)
    surface.blit(overlay, (0, 0))

    if winner is None:
        text = title_font.render("GAME OVER", True, RED)
    elif isinstance(winner, list):
        text = title_font.render("DRAW!", True, YELLOW)
    else:
        text = title_font.render(f"P{winner.player_id + 1} WINS!", True, winner.color)
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40))
    surface.blit(text, text_rect)

    score_text = hud_font.render(f"Final Score: {score}", True, WHITE)
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
    surface.blit(score_text, score_rect)

    restart_text = font.render("Press R to restart or Q to quit", True, WHITE)
    restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 70))
    surface.blit(restart_text, restart_rect)


def reset_game():
    """Reset the entire game."""
    global game_over, winner, level, score, players
    game_over = False
    winner = None
    level = 1
    score = 0

    players = [
        Player(1, 1, BLUE, {
            'up': pygame.K_w, 'down': pygame.K_s,
            'left': pygame.K_a, 'right': pygame.K_d,
            'bomb': pygame.K_SPACE
        }, 0),
        Player(GRID_COLS - 2, GRID_ROWS - 2, GREEN, {
            'up': pygame.K_UP, 'down': pygame.K_DOWN,
            'left': pygame.K_LEFT, 'right': pygame.K_RIGHT,
            'bomb': pygame.K_RETURN
        }, 1)
    ]

    generate_level(level)


def next_level():
    """Advance to the next level."""
    global level, game_over, winner
    level += 1
    game_over = False
    winner = None

    # Reset player positions
    players[0].grid_x = 1
    players[0].grid_y = 1
    players[0].pixel_x = 1 * TILE_SIZE + TILE_SIZE // 2
    players[0].pixel_y = 1 * TILE_SIZE + TILE_SIZE // 2
    players[0].alive = True
    players[0].active_bombs = 0

    if len(players) > 1:
        players[1].grid_x = GRID_COLS - 2
        players[1].grid_y = GRID_ROWS - 2
        players[1].pixel_x = (GRID_COLS - 2) * TILE_SIZE + TILE_SIZE // 2
        players[1].pixel_y = (GRID_ROWS - 2) * TILE_SIZE + TILE_SIZE // 2
        players[1].alive = True
        players[1].active_bombs = 0

    generate_level(level)


def draw_grid_tiles(surface):
    """Draw the game grid."""
    for y in range(GRID_ROWS):
        for x in range(GRID_COLS):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)

            if grid[y][x] == WALL:
                # Brick wall pattern
                pygame.draw.rect(surface, BROWN, rect)
                pygame.draw.rect(surface, DARK_BROWN, rect, 1)
                # Brick lines
                pygame.draw.line(surface, DARK_BROWN,
                                 (x * TILE_SIZE, y * TILE_SIZE + TILE_SIZE // 2),
                                 (x * TILE_SIZE + TILE_SIZE, y * TILE_SIZE + TILE_SIZE // 2), 1)
                pygame.draw.line(surface, DARK_BROWN,
                                 (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE),
                                 (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2), 1)
            elif grid[y][x] == SOFT_BLOCK:
                pygame.draw.rect(surface, LIGHT_BROWN, rect)
                pygame.draw.rect(surface, BROWN, rect, 2)
                # Cross pattern
                pygame.draw.line(surface, BROWN,
                                 (x * TILE_SIZE + 4, y * TILE_SIZE + 4),
                                 (x * TILE_SIZE + TILE_SIZE - 4, y * TILE_SIZE + TILE_SIZE - 4), 1)
                pygame.draw.line(surface, BROWN,
                                 (x * TILE_SIZE + TILE_SIZE - 4, y * TILE_SIZE + 4),
                                 (x * TILE_SIZE + 4, y * TILE_SIZE + TILE_SIZE - 4), 1)
            elif grid[y][x] == EMPTY:
                # Checkered floor pattern
                if (x + y) % 2 == 0:
                    pygame.draw.rect(surface, (50, 50, 50), rect)
                else:
                    pygame.draw.rect(surface, (55, 55, 55), rect)


def main():
    global game_over

    clock = pygame.time.Clock()
    reset_game()
    running = True
    level_transition_timer = 0

    while running:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if game_over:
                    if event.key == pygame.K_r:
                        reset_game()
                else:
                    # Bomb placement
                    for player in players:
                        if event.key == player.controls['bomb']:
                            bomb = player.place_bomb()
                            if bomb:
                                bombs.append(bomb)
                                grid[bomb.grid_y][bomb.grid_x] = BOMB

        if not game_over:
            keys = pygame.key.get_pressed()

            # Update players
            for player in players:
                player.handle_input(keys)

            # Update bombs
            for bomb in bombs[:]:
                bomb.update()
                if bomb.exploded:
                    bombs.remove(bomb)

            # Update explosions
            explosions[:] = [e for e in explosions if e.update()]

            # Update enemies
            for enemy in enemies:
                enemy.update()

            # Update powerups (visual pulse)
            for pu in powerups:
                pu.update()

            # Check collisions
            check_collisions()

            # Level transition
            if game_over and winner is not None and not isinstance(winner, list):
                level_transition_timer += 1
                if level_transition_timer > 90:  # 1.5 seconds delay
                    next_level()
                    level_transition_timer = 0

        # Draw everything
        screen.fill(BLACK)
        draw_grid_tiles(screen)

        # Draw powerups
        for pu in powerups:
            pu.draw(screen)

        # Draw enemies
        for enemy in enemies:
            enemy.draw(screen)

        # Draw bombs
        for bomb in bombs:
            bomb.draw(screen)

        # Draw explosions
        for exp in explosions:
            exp.draw(screen)

        # Draw players
        for player in players:
            player.draw(screen)

        # Draw HUD
        draw_hud(screen)

        # Draw game over
        if game_over:
            draw_game_over(screen)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()