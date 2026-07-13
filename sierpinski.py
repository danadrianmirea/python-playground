import pygame
import sys

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sierpinski Triangle")
clock = pygame.time.Clock()

# Colors
BG_COLOR = (20, 20, 30)
TRI_COLOR = (255, 113, 41)  # Orange
CENTER_COLOR = (46, 47, 41)  # Dark grey (negative space)

# Base case minimum side length
MIN_SIDE = 3

def midpoint(p1, p2):
    """Return the midpoint of two points."""
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def sierpinski(screen, v0, v1, v2):
    """Draw a Sierpinski Triangle recursively."""
    # If the triangle is small enough, just draw it
    if v2[0] - v0[0] < MIN_SIDE:
        pygame.draw.polygon(screen, TRI_COLOR, [v0, v1, v2])
    else:
        # Draw the main triangle
        pygame.draw.polygon(screen, TRI_COLOR, [v0, v1, v2])

        # Calculate midpoints
        mid0 = midpoint(v0, v1)
        mid1 = midpoint(v0, v2)
        mid2 = midpoint(v1, v2)

        # Draw the center "negative space" triangle
        pygame.draw.polygon(screen, CENTER_COLOR, [mid0, mid1, mid2])

        # Recursively draw the other three sub-triangles
        sierpinski(screen, v0, mid0, mid1)
        sierpinski(screen, mid0, v1, mid2)
        sierpinski(screen, mid1, mid2, v2)

# Main loop
def main():
    running = True

    # Define the main triangle vertices
    # v0 = lower-left, v1 = upper, v2 = lower-right
    v0 = (50, HEIGHT - 50)
    v1 = (WIDTH // 2, 50)
    v2 = (WIDTH - 50, HEIGHT - 50)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Redraw on spacebar press
                screen.fill(BG_COLOR)
                sierpinski(screen, v0, v1, v2)
                pygame.display.flip()

        # Draw on first run
        screen.fill(BG_COLOR)
        sierpinski(screen, v0, v1, v2)
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()