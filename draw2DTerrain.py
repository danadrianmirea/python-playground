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
minTerrainHeight = 0
waterLevel = 100
numberOfSegments = 50
roughness = 1.0
baseHeight = HEIGHT



def generate_terrain_heights():
    """Generate terrain using a random walk, then apply erosion smoothing."""
    heights = []

    # Start at a random height within the allowed range
    current_height = random.uniform(minTerrainHeight, maxTerrainHeight)

    for i in range(numberOfSegments):
        # Max step change per segment depends on roughness
        # At roughness=0, max step is small (smooth). At roughness=1, max step is large (jagged)
        max_step = maxTerrainHeight * (0.05 + roughness * 0.25)

        # Random walk: move up or down by a random amount, clamped by max_step
        current_height += random.uniform(-max_step, max_step)

        # Clamp to allowed range
        current_height = max(minTerrainHeight, min(maxTerrainHeight, current_height))

        heights.append(current_height)

    # Apply erosion: multiple passes of a smoothing kernel
    # This simulates natural erosion by wearing down sharp peaks and filling valleys
    erosion_passes = 3
    for _ in range(erosion_passes):
        eroded = heights[:]
        for i in range(1, len(heights) - 1):
            # Each point moves toward the average of its neighbors
            # Sharp peaks get pulled down, sharp valleys get filled up
            eroded[i] = heights[i] * 0.5 + (heights[i - 1] + heights[i + 1]) * 0.25
        heights = eroded

    # Convert heights to screen Y coordinates
    screen_heights = [baseHeight - h for h in heights]
    return screen_heights








def draw_water(surface):
    water_level = baseHeight - waterLevel
    water_color = SKY_BLUE;
    pygame.draw.rect(surface, water_color, (0, water_level, WIDTH, HEIGHT - water_level))


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
        draw_water(screen)
        draw_terrain(screen, heights)
        draw_ui(screen)


        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
