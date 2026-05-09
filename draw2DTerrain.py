import pygame
import random
import math
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SKY_BLUE = (135, 206, 235)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Terrain Generator")
clock = pygame.time.Clock()

# Terrain configuration
maxTerrainHeight = 300
numberOfSegments = 100
roughness = 0.3
baseHeight = HEIGHT - 100


def generate_terrain_heights():
    """Generate terrain height values using sine waves and noise."""
    heights = []

    # Random phase offsets so each R press gives different terrain
    phase1 = random.uniform(0, 2 * math.pi)
    phase2 = random.uniform(0, 2 * math.pi)
    phase3 = random.uniform(0, 2 * math.pi)

    for i in range(numberOfSegments):
        x = i * (WIDTH / (numberOfSegments - 1))
        h = baseHeight

        # Multiple octaves of sine with random phase for natural variation
        h -= math.sin(x * 0.003 + phase1) * maxTerrainHeight * 0.5
        h -= math.sin(x * 0.008 + phase2) * maxTerrainHeight * 0.3
        h -= math.sin(x * 0.02 + phase3) * maxTerrainHeight * 0.15

        # Add random roughness
        h -= random.uniform(-1, 1) * maxTerrainHeight * roughness * 0.2

        heights.append(h)

    return heights



def draw_terrain(surface, heights):
    """Draw the terrain as a filled polygon at the bottom of the screen."""
    segment_width = WIDTH / (numberOfSegments - 1)

    # Build polygon points: along the terrain top edge, then down to bottom-right, across to bottom-left
    points = []
    for i in range(numberOfSegments):
        x = i * segment_width
        y = int(heights[i])
        points.append((x, y))


    # Close the polygon at the bottom of the screen
    points.append((WIDTH, HEIGHT))
    points.append((0, HEIGHT))

    # Draw filled terrain
    terrain_color = (34, 139, 34)  # Forest green
    pygame.draw.polygon(surface, terrain_color, points)

    # Draw the top edge line (the terrain surface)
    edge_color = (0, 80, 0)  # Dark green
    for i in range(numberOfSegments - 1):
        x1 = i * segment_width
        y1 = int(heights[i])
        x2 = (i + 1) * segment_width
        y2 = int(heights[i + 1])
        pygame.draw.line(surface, edge_color, (x1, y1), (x2, y2), 2)



def draw_sky(surface):
    """Draw a gradient sky background."""
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(SKY_BLUE[0] * (1 - t) + 200 * t)
        g = int(SKY_BLUE[1] * (1 - t) + 220 * t)
        b = int(SKY_BLUE[2] * (1 - t) + 255 * t)
        pygame.draw.line(surface, (r, g, b), (0, y), (WIDTH, y))


def draw_ui(surface):
    """Draw UI overlay."""
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)

    title = font.render("2D Terrain Generator", True, WHITE)
    surface.blit(title, (10, 10))

    instructions = [
        "R: Generate new terrain",
        "Up/Down: Adjust max height",
        "Left/Right: Adjust roughness",
    ]

    for i, text in enumerate(instructions):
        surface.blit(small_font.render(text, True, (200, 200, 200)), (10, 35 + i * 18))

    stats_lines = [
        f"Max Height: {maxTerrainHeight}",
        f"Roughness: {roughness:.2f}",
        f"Segments: {numberOfSegments}",
    ]

    for i, text in enumerate(stats_lines):
        surface.blit(small_font.render(text, True, (200, 255, 200)), (10, HEIGHT - 60 + i * 18))


def main():
    """Main game loop."""
    global maxTerrainHeight, roughness
    heights = generate_terrain_heights()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    heights = generate_terrain_heights()
                elif event.key == pygame.K_UP:
                    maxTerrainHeight = min(500, maxTerrainHeight + 20)
                    heights = generate_terrain_heights()
                elif event.key == pygame.K_DOWN:
                    maxTerrainHeight = max(50, maxTerrainHeight - 20)
                    heights = generate_terrain_heights()
                elif event.key == pygame.K_LEFT:
                    roughness = max(0.0, roughness - 0.1)
                    heights = generate_terrain_heights()
                elif event.key == pygame.K_RIGHT:
                    roughness = min(1.0, roughness + 0.1)
                    heights = generate_terrain_heights()

        # Draw
        draw_sky(screen)
        draw_terrain(screen, heights)
        draw_ui(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
