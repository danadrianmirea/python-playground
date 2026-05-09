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
WATER_BLUE = (0, 0, 255)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Terrain Generator")
clock = pygame.time.Clock()

# Terrain configuration
maxTerrainHeight = 600
minTerrainHeight = 0
mountainLevel = 500
waterLevelPct = 0.2      # % of mountainLevel
hillsLevelPct = 0.8       # % of mountainLevel
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
    water_level = baseHeight - int(mountainLevel * waterLevelPct)
    water_color = WATER_BLUE
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
    # Compute actual level values from percentages of mountainLevel
    water_level = mountainLevel * waterLevelPct
    hills_level = mountainLevel * hillsLevelPct


    # Zone colors
    SEABED = (0, 0, 0)          # Dark greenish-brown (deep underwater)
    SAND = (34, 139, 34)        # (beach/waterline)
    GRASS = (34, 139, 34)       # Forest green
    HILL = (189, 183, 107)      # Yellowish khaki
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

    elif height <= mountainLevel:
        # Hills to mountain: interpolate HILL -> SNOW
        t = (height - hills_level) / (mountainLevel - hills_level)

        return lerp_color(HILL, SNOW, t)
    else:
        # Above mountain: pure snow
        return SNOW
    

def draw_terrain(surface, heights):
    """Draw the terrain with height-based coloring using horizontal strips.
    
    Draws horizontal lines from bottom to top, but only where the terrain
    surface is above the current y position. This avoids drawing over the sky.
    """
    segment_width = WIDTH / (numberOfSegments - 1)

    # Precompute the terrain surface Y for every x pixel
    terrain_surface_y = []
    for x in range(WIDTH):
        seg_index = int(x / segment_width)
        seg_index = min(seg_index, numberOfSegments - 2)
        x1 = seg_index * segment_width
        x2 = (seg_index + 1) * segment_width
        t = (x - x1) / (x2 - x1) if x2 > x1 else 0
        terrain_surface_y.append(int(heights[seg_index] + (heights[seg_index + 1] - heights[seg_index]) * t))

    # Find the minimum and maximum surface y for quick checks
    min_surface_y = min(terrain_surface_y)
    max_surface_y = max(terrain_surface_y)

    # Draw horizontal lines from bottom to top
    for y in range(HEIGHT - 1, -1, -1):
        # Quick check: if y is below the lowest surface point, terrain exists everywhere
        if y >= max_surface_y:
            height_val = baseHeight - y
            color = get_terrain_color(height_val)
            pygame.draw.line(surface, color, (0, y), (WIDTH, y))
        elif y >= min_surface_y:
            # y is in the surface range - find x-ranges where terrain exists
            height_val = baseHeight - y
            color = get_terrain_color(height_val)

            x_start = None
            x_end = None
            for x in range(WIDTH):
                if terrain_surface_y[x] <= y:
                    if x_start is None:
                        x_start = x
                    x_end = x
                elif x_start is not None:
                    pygame.draw.line(surface, color, (x_start, y), (x_end, y))
                    x_start = None
                    x_end = None

            if x_start is not None:
                pygame.draw.line(surface, color, (x_start, y), (x_end, y))
        # else: y is above the highest surface point, no terrain here - skip


def draw_sky(surface):
    """Draw a gradient sky background."""
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(SKY_BLUE[0] * (1 - t) + 200 * t)
        g = int(SKY_BLUE[1] * (1 - t) + 220 * t)
        b = int(SKY_BLUE[2] * (1 - t) + 255 * t)
        pygame.draw.line(surface, (r, g, b), (0, y), (WIDTH, y))

def draw_ui(surface, fps, render_time_ms):
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
        surface.blit(small_font.render(text, True, (255, 255, 255)), (10, 35 + i * 18))

    # Performance stats (top-right)
    perf_text = f"FPS: {fps:.0f}  |  Render: {render_time_ms:.1f}ms"
    perf_surf = small_font.render(perf_text, True, (255, 255, 100))
    surface.blit(perf_surf, (WIDTH - perf_surf.get_width() - 10, 10))

    # Terrain stats (bottom-left)
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
                    maxTerrainHeight = min(800, maxTerrainHeight + 20)

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

        # Measure render time
        render_start = pygame.time.get_ticks()

        # Draw
        draw_sky(screen)
        draw_water(screen)
        draw_terrain(screen, heights)

        render_time = pygame.time.get_ticks() - render_start

        # Get FPS from clock
        fps = clock.get_fps()

        draw_ui(screen, fps, render_time)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
