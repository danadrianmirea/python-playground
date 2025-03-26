import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Set up display
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mandelbrot Set")

# Initial view parameters
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5
max_iter = 100

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
    pixels = mandelbrot(HEIGHT, WIDTH, x_min, x_max, y_min, y_max, max_iter)
    surface = pygame.surfarray.make_surface(pixels * 255 / max_iter)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Get mouse position and convert to complex coordinates
            mouse_x, mouse_y = event.pos
            # Convert screen coordinates to complex plane coordinates
            x = x_min + (x_max - x_min) * mouse_x / WIDTH
            y = y_min + (y_max - y_min) * (HEIGHT - mouse_y) / HEIGHT
            
            # Zoom factor (0.1 for zoom in, 10 for zoom out)
            zoom_factor = 0.1 if event.button == 1 else 10.0
            
            # Calculate new view parameters
            x_span = (x_max - x_min) * zoom_factor
            y_span = (y_max - y_min) * zoom_factor
            
            # Calculate the relative position of the clicked point
            rel_x = (x - x_min) / (x_max - x_min)
            rel_y = (y - y_min) / (y_max - y_min)
            
            # Calculate new bounds while maintaining the clicked point's relative position
            x_min = x - x_span * rel_x
            x_max = x + x_span * (1 - rel_x)
            y_min = y - y_span * rel_y
            y_max = y + y_span * (1 - rel_y)
            
            # Redraw
            draw_mandelbrot()
    
    # Initial draw
    if running:
        draw_mandelbrot()

pygame.quit()
