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
maxTerrainHeight = 600
minTerrainHeight = 0
waterLevelPct = 0.3       
hillsLevelPct = 0.6
mountainLevelPct = 0.9
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
    water_level = baseHeight - int(maxTerrainHeight * waterLevelPct)
    water_color = pygame.Color(0, 0, 255)
    pygame.draw.rect(surface, water_color, (0, water_level, WIDTH, HEIGHT - water_level))



def lerp_color(c1, c2, t):
    """Linearly interpolate between two colors."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def get_terrain_color(height):
    """Get terrain color based on height with smooth interpolation between zones."""
    # Compute actual level values from percentages of maxTerrainHeight
    water_level = maxTerrainHeight * waterLevelPct
    hills_level = maxTerrainHeight * hillsLevelPct
    mountain_level = maxTerrainHeight * mountainLevelPct

    # Zone colors
    SEABED = (0, 0, 0)          # Dark greenish-brown (deep underwater)
    SAND = (34, 139, 34)        # (beach/waterline)
    GRASS = (34, 139, 34)       # Forest green
    HILL = (139, 119, 80)       # Brown
    SNOW = (255, 255, 255)      # White


    if height <= water_level:
        # Below water: interpolate from SEABED (deep) to SAND (near waterline)
        t = height / water_level if water_level > 0 else 0
        return lerp_color(SEABED, SAND, t)

    # Above water: beach zone from waterline up to water_level + small transition
    beach_top = water_level + (hills_level - water_level) * 0.15
    if height <= beach_top:
        # Beach: interpolate from SAND to GRASS
        t = (height - water_level) / (beach_top - water_level) if beach_top > water_level else 0
        return lerp_color(SAND, GRASS, t)

    elif height <= hills_level:
        # Water level to hills: interpolate GRASS -> HILL
        t = (height - beach_top) / (hills_level - beach_top)
        return lerp_color(GRASS, HILL, t)

    elif height <= mountain_level:
        # Hills to mountain: interpolate HILL -> SNOW
        t = (height - hills_level) / (mountain_level - hills_level)
        return lerp_color(HILL, SNOW, t)
    else:
        # Above mountain: pure snow
        return SNOW



def draw_terrain(surface, heights):
    """Draw the terrain with height-based coloring varying per pixel column."""
    segment_width = WIDTH / (numberOfSegments - 1)

    # Draw each pixel column from the terrain surface down to the bottom
    for x in range(WIDTH):
        # Find which segment this x falls into
        seg_index = int(x / segment_width)
        seg_index = min(seg_index, numberOfSegments - 2)

        # Interpolate the terrain Y at this exact x position
        x1 = seg_index * segment_width
        x2 = (seg_index + 1) * segment_width
        t = (x - x1) / (x2 - x1) if x2 > x1 else 0
        terrain_y = int(heights[seg_index] + (heights[seg_index + 1] - heights[seg_index]) * t)

        # Draw vertical line from terrain surface down to bottom, varying color by height
        for y in range(terrain_y, HEIGHT):
            # Convert screen Y to actual height value
            height_val = baseHeight - y
            color = get_terrain_color(height_val)
            surface.set_at((x, y), color)





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
