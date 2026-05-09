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
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
BROWN = (139, 90, 43)
SKY_BLUE = (135, 206, 235)
DARK_BROWN = (101, 67, 33)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Terrain Generator")
clock = pygame.time.Clock()

# Terrain configuration
terrain_config = {
    'maxTerrainHeight': 300,        # Maximum height variation of terrain
    'numberOfTerrainLines': 3,      # Number of terrain layers (for depth)
    'segmentWidth': 20,             # Width of each terrain segment
    'roughness': 0.6,               # How jagged the terrain is (0.0 - 1.0)
    'baseHeight': HEIGHT - 100,     # Base Y position of the terrain floor
    'colorVariance': 30,            # Color variation between layers
}


def generate_terrain_heights():
    """Generate terrain height values using midpoint displacement."""
    config = terrain_config
    num_segments = WIDTH // config['segmentWidth'] + 1
    heights = []

    for i in range(num_segments):
        # Use sine waves combined with noise for natural-looking terrain
        x = i * config['segmentWidth']
        h = config['baseHeight']

        # Multiple octaves of sine for natural variation
        h -= math.sin(x * 0.003) * config['maxTerrainHeight'] * 0.5
        h -= math.sin(x * 0.008 + 1.3) * config['maxTerrainHeight'] * 0.3
        h -= math.sin(x * 0.02 + 2.7) * config['maxTerrainHeight'] * 0.15

        # Add random roughness
        h -= random.uniform(-1, 1) * config['maxTerrainHeight'] * config['roughness'] * 0.2

        heights.append(h)

    return heights


def generate_terrain():
    """Generate complete terrain data with multiple layers, colors, and grass."""
    config = terrain_config
    base_heights = generate_terrain_heights()
    num_layers = config['numberOfTerrainLines']

    layers = []
    layer_colors = []
    for layer in range(num_layers):
        layer_heights = []
        # Each subsequent layer is slightly higher (closer to camera) and offset
        layer_offset = layer * 15  # Vertical offset for depth effect
        for h in base_heights:
            # Add some independent variation per layer
            varied = h - layer_offset + random.uniform(-5, 5)
            layer_heights.append(varied)
        layers.append(layer_heights)

        # Generate a fixed color for this layer (so it doesn't flicker each frame)
        t = layer / max(1, num_layers - 1)  # 0 = back, 1 = front
        r = int(BROWN[0] + (GREEN[0] - BROWN[0]) * t + random.uniform(-config['colorVariance'], config['colorVariance']))
        g = int(BROWN[1] + (GREEN[1] - BROWN[1]) * t + random.uniform(-config['colorVariance'], config['colorVariance']))
        b = int(BROWN[2] + (GREEN[2] - BROWN[2]) * t + random.uniform(-config['colorVariance'], config['colorVariance']))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        layer_colors.append((r, g, b))

    # Generate grass data once (so it doesn't flicker each frame)
    grass_data = []
    if layers:
        top_layer = layers[-1]
        for i in range(0, len(top_layer), 3):
            x = i * config['segmentWidth']
            y = int(top_layer[i])
            for _ in range(3):
                gx = x + random.randint(-4, 4)
                gy = y - random.randint(3, 10)
                gc = (random.randint(50, 100), random.randint(120, 200), random.randint(20, 60))
                grass_data.append((gx, y, gy, gc))

    return layers, layer_colors, grass_data


def draw_terrain(surface, layers, layer_colors, grass_data):
    """Draw the terrain layers."""
    config = terrain_config
    num_segments = WIDTH // config['segmentWidth'] + 1

    for layer_idx, layer_heights in enumerate(layers):
        color = layer_colors[layer_idx]
        darker_color = (
            max(0, color[0] - 40),
            max(0, color[1] - 40),
            max(0, color[2] - 40)
        )


        points = []
        for i in range(num_segments):
            x = i * config['segmentWidth']
            y = int(layer_heights[i])
            points.append((x, y))

        # Close the polygon at the bottom
        if points:
            points.append((WIDTH, HEIGHT))
            points.append((0, HEIGHT))

            # Draw filled terrain
            pygame.draw.polygon(surface, color, points)

            # Draw the top line of the terrain (the surface)
            for i in range(len(points) - 3):
                x1 = points[i][0]
                y1 = points[i][1]
                x2 = points[i + 1][0]
                y2 = points[i + 1][1]
                pygame.draw.line(surface, darker_color, (x1, y1), (x2, y2), 2)

    # Draw pre-generated grass tufts on the topmost layer
    for gx, ground_y, gy, gc in grass_data:
        pygame.draw.line(surface, gc, (gx, ground_y), (gx, gy), 1)



def draw_sky(surface):
    """Draw a gradient sky background."""
    for y in range(HEIGHT):
        # Gradient from sky blue at top to lighter near horizon
        t = y / HEIGHT
        r = int(SKY_BLUE[0] * (1 - t) + 200 * t)
        g = int(SKY_BLUE[1] * (1 - t) + 220 * t)
        b = int(SKY_BLUE[2] * (1 - t) + 255 * t)
        pygame.draw.line(surface, (r, g, b), (0, y), (WIDTH, y))


def draw_ui(surface):
    """Draw UI overlay."""
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)

    # Title
    title = font.render("2D Terrain Generator", True, WHITE)
    surface.blit(title, (10, 10))

    # Instructions
    instructions = [
        "R: Generate new terrain",
        "Up/Down: Adjust max height",
        "Left/Right: Adjust roughness",
        "+/-: Change number of layers",
    ]

    for i, text in enumerate(instructions):
        surface.blit(small_font.render(text, True, (200, 200, 200)), (10, 35 + i * 18))

    # Stats
    config = terrain_config
    stats_lines = [
        f"Max Height: {config['maxTerrainHeight']}",
        f"Layers: {config['numberOfTerrainLines']}",
        f"Roughness: {config['roughness']:.2f}",
        f"Segments: {WIDTH // config['segmentWidth']}",
    ]
    for i, text in enumerate(stats_lines):
        surface.blit(small_font.render(text, True, (200, 255, 200)), (10, HEIGHT - 80 + i * 18))


def main():
    """Main game loop."""
    layers, layer_colors, grass_data = generate_terrain()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Regenerate terrain
                    layers, layer_colors, grass_data = generate_terrain()
                elif event.key == pygame.K_UP:
                    terrain_config['maxTerrainHeight'] = min(500, terrain_config['maxTerrainHeight'] + 20)
                    layers, layer_colors, grass_data = generate_terrain()
                elif event.key == pygame.K_DOWN:
                    terrain_config['maxTerrainHeight'] = max(50, terrain_config['maxTerrainHeight'] - 20)
                    layers, layer_colors, grass_data = generate_terrain()
                elif event.key == pygame.K_LEFT:
                    terrain_config['roughness'] = max(0.0, terrain_config['roughness'] - 0.1)
                    layers, layer_colors, grass_data = generate_terrain()
                elif event.key == pygame.K_RIGHT:
                    terrain_config['roughness'] = min(1.0, terrain_config['roughness'] + 0.1)
                    layers, layer_colors, grass_data = generate_terrain()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    terrain_config['numberOfTerrainLines'] = min(8, terrain_config['numberOfTerrainLines'] + 1)
                    layers, layer_colors, grass_data = generate_terrain()
                elif event.key == pygame.K_MINUS:
                    terrain_config['numberOfTerrainLines'] = max(1, terrain_config['numberOfTerrainLines'] - 1)
                    layers, layer_colors, grass_data = generate_terrain()

        # Draw
        draw_sky(screen)
        draw_terrain(screen, layers, layer_colors, grass_data)
        draw_ui(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()




if __name__ == "__main__":
    main()
