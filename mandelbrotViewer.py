import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Set up display
WIDTH = 800
HEIGHT = 800  # Changed to match WIDTH for square window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mandelbrot Set")

# Initial view parameters
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5  # These values already maintain a square aspect ratio
max_iter = 100

# Store the current Mandelbrot set
current_pixels = None

# Rectangle drawing variables
drawing = False
start_pos = None
current_pos = None

def mandelbrot(h, w, x_min, x_max, y_min, y_max, max_iter):
    x = np.linspace(x_min, x_max, w)
    y = np.linspace(y_min, y_max, h)
    c = x[:, np.newaxis] + 1j * y
    z = np.zeros_like(c)
    mask = np.ones_like(c, dtype=bool)
    output = np.zeros_like(c, dtype=int)
    
    for i in range(max_iter):
        z[mask] = z[mask]**2 + c[mask]
        mask_new = abs(z) < 2
        output[mask & ~mask_new] = i
        mask = mask_new
    
    return output.T

def draw_mandelbrot():
    global current_pixels
    if current_pixels is None:
        current_pixels = mandelbrot(HEIGHT, WIDTH, x_min, x_max, y_min, y_max, max_iter)
    surface = pygame.surfarray.make_surface(current_pixels * 255 / max_iter)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

def update_mandelbrot():
    global current_pixels
    current_pixels = mandelbrot(HEIGHT, WIDTH, x_min, x_max, y_min, y_max, max_iter)
    draw_mandelbrot()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                drawing = True
                start_pos = event.pos
                current_pos = event.pos
            elif event.button == 3:  # Right click
                # Zoom out by a factor of 10
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = (x_max - x_min) * 10
                height = (y_max - y_min) * 10
                x_min = center_x - width / 2
                x_max = center_x + width / 2
                y_min = center_y - height / 2
                y_max = center_y + height / 2
                update_mandelbrot()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and drawing:  # Left click release
                drawing = False
                if start_pos and current_pos:
                    # Calculate the rectangle bounds
                    rect_x1 = min(start_pos[0], current_pos[0])
                    rect_x2 = max(start_pos[0], current_pos[0])
                    rect_y1 = min(start_pos[1], current_pos[1])
                    rect_y2 = max(start_pos[1], current_pos[1])
                    
                    # Convert rectangle coordinates to complex plane coordinates
                    new_x_min = x_min + (x_max - x_min) * rect_x1 / WIDTH
                    new_x_max = x_min + (x_max - x_min) * rect_x2 / WIDTH
                    new_y_min = y_min + (y_max - y_min) * (HEIGHT - rect_y2) / HEIGHT
                    new_y_max = y_min + (y_max - y_min) * (HEIGHT - rect_y1) / HEIGHT
                    
                    # Calculate aspect ratios
                    screen_ratio = WIDTH / HEIGHT
                    rect_ratio = (rect_x2 - rect_x1) / (rect_y2 - rect_y1)
                    
                    # Adjust bounds to maintain aspect ratio
                    if rect_ratio > screen_ratio:
                        # Rectangle is wider than screen ratio
                        center_y = (new_y_min + new_y_max) / 2
                        height = (new_x_max - new_x_min) / screen_ratio
                        new_y_min = center_y - height / 2
                        new_y_max = center_y + height / 2
                    else:
                        # Rectangle is taller than screen ratio
                        center_x = (new_x_min + new_x_max) / 2
                        width = (new_y_max - new_y_min) * screen_ratio
                        new_x_min = center_x - width / 2
                        new_x_max = center_x + width / 2
                    
                    # Update view bounds
                    x_min, x_max = new_x_min, new_x_max
                    y_min, y_max = new_y_min, new_y_max
                    
                    # Reset drawing variables
                    start_pos = None
                    current_pos = None
                    
                    # Update and redraw with new bounds
                    update_mandelbrot()
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                current_pos = event.pos
                # Draw the rectangle without recalculating the Mandelbrot set
                draw_mandelbrot()
                if start_pos and current_pos:
                    rect = pygame.Rect(
                        min(start_pos[0], current_pos[0]),
                        min(start_pos[1], current_pos[1]),
                        abs(current_pos[0] - start_pos[0]),
                        abs(current_pos[1] - start_pos[1])
                    )
                    pygame.draw.rect(screen, (255, 255, 255), rect, 1)
                    pygame.display.flip()
    
    # Initial draw
    if running and current_pixels is None:
        draw_mandelbrot()

pygame.quit()
