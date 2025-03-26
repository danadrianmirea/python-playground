import pygame
import numpy as np
from colorsys import hsv_to_rgb

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

# Color palette settings
color_mode = 0  # 0: Smooth colormap, 1: Classic rainbow, 2: Fire palette
color_shift = 0.0  # color rotation value

# Add a global variable to store colored pixels
colored_pixels = None

# Cache for color lookup tables
color_lookup_cache = {}

def mandelbrot(h, w, x_min, x_max, y_min, y_max, max_iter):
    # Use float64 for better precision
    x = np.linspace(x_min, x_max, w, dtype=np.float64)
    y = np.linspace(y_min, y_max, h, dtype=np.float64)
    
    # Create complex array with float64 precision
    c = x[:, np.newaxis] + 1j * y
    z = np.zeros_like(c, dtype=np.complex128)
    mask = np.ones_like(c, dtype=bool)
    output = np.zeros_like(c, dtype=int)
    
    # Optimize the calculation to reduce floating-point errors
    for i in range(max_iter):
        z[mask] = z[mask]**2 + c[mask]
        mask_new = abs(z) < 2
        output[mask & ~mask_new] = i
        mask = mask_new
    
    return output.T

def create_smooth_colormap():
    """Create a lookup table for smooth color mapping"""
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        # Normalized value in [0, 1]
        t = i / 255.0
        
        # Calculate HSV
        h = t  # hue = normalized value
        s = 0.8  # saturation
        v = 1.0 if t < 0.95 else (1.0 - t) * 20  # falloff for high values
        
        # Convert HSV to RGB
        r, g, b = hsv_to_rgb(h, s, v)
        cmap[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return cmap

def color_map(iterations, max_iter, mode=0, shift=0.0):
    """Map iteration counts to colors using different coloring schemes"""
    global color_lookup_cache
    
    # Create a single optimized color map function that avoids unnecessary operations
    height, width = iterations.shape
    
    # Create a mask for points in the set (didn't escape)
    in_set = iterations == max_iter
    
    # Initialize the RGB array with zeros (black for in-set points)
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Skip further processing if all points are in set
    if np.all(in_set):
        return rgb_array
    
    # Normalize iteration counts for escaped points (only where needed)
    mask = ~in_set
    
    if mode == 0:  # Smooth colormap with log smoothing
        # Check if we have a cached colormap
        cache_key = f"smooth_{shift:.2f}"
        if cache_key not in color_lookup_cache:
            # Create the colormap and cache it
            base_cmap = create_smooth_colormap()
            
            # Apply shift if needed
            if shift != 0:
                # Shift the colormap by shifting indices
                shift_amount = int(shift * 256) % 256
                shifted_cmap = np.zeros((256, 3), dtype=np.uint8)
                shifted_cmap[:256-shift_amount] = base_cmap[shift_amount:]
                shifted_cmap[256-shift_amount:] = base_cmap[:shift_amount]
                color_lookup_cache[cache_key] = shifted_cmap
            else:
                color_lookup_cache[cache_key] = base_cmap
        
        # Use the cached colormap
        cmap = color_lookup_cache[cache_key]
        
        # Compute normalized values only for points outside the set
        norm_values = np.zeros_like(iterations, dtype=np.float64)
        norm_values[mask] = iterations[mask] / max_iter
        
        # Apply logarithmic smoothing
        norm_values[mask] = np.log(norm_values[mask] * 0.5 + 0.5) / np.log(1.5)
        
        # Scale to 0-255 for lookup table index
        indices = (norm_values[mask] * 255).astype(np.uint8)
        
        # Use the lookup table to apply colors (vectorized)
        mask_indices = np.where(mask)
        rgb_array[mask_indices[0], mask_indices[1]] = cmap[indices]
    
    elif mode == 1:  # Rainbow palette using sine waves
        # Direct RGB calculation for rainbow palette
        norm_values = np.zeros_like(iterations, dtype=np.float64)
        norm_values[mask] = iterations[mask] / max_iter
        
        # Scale and shift
        phase = ((norm_values * 3.0) + shift) % 1.0
        
        # Convert phase to angle in radians (0 to 2π)
        angle = phase * 2 * np.pi
        
        # Calculate RGB components using sine waves (120° phase shifts)
        # This is faster than HSV conversion and gives rainbow-like effect
        r = np.zeros_like(angle)
        g = np.zeros_like(angle)
        b = np.zeros_like(angle)
        
        r[mask] = np.sin(angle[mask]) * 0.5 + 0.5
        g[mask] = np.sin(angle[mask] + 2*np.pi/3) * 0.5 + 0.5
        b[mask] = np.sin(angle[mask] + 4*np.pi/3) * 0.5 + 0.5
        
        # Scale to enhance colors
        r = np.clip(r * 1.5, 0, 1)
        g = np.clip(g * 1.5, 0, 1)
        b = np.clip(b * 1.5, 0, 1)
        
        # Scale to 0-255 and convert to uint8
        rgb_array[..., 0] = (r * 255).astype(np.uint8)
        rgb_array[..., 1] = (g * 255).astype(np.uint8)
        rgb_array[..., 2] = (b * 255).astype(np.uint8)
    
    elif mode == 2:  # Fire palette
        # Simplified fire palette calculation
        norm_values = np.zeros_like(iterations, dtype=np.float64)
        norm_values[mask] = iterations[mask] / max_iter
        
        # Apply shift
        values = (norm_values + shift) % 1.0
        
        # Pre-allocate arrays
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        
        # Red component: 0→1 in first quarter, then stay at 1
        r[mask & (values < 0.25)] = values[mask & (values < 0.25)] * 4
        r[mask & (values >= 0.25)] = 1.0
        
        # Green component: 0 in first quarter, 0→1 in second quarter, then 1
        mask2 = mask & (values >= 0.25) & (values < 0.5)
        g[mask2] = (values[mask2] - 0.25) * 4
        g[mask & (values >= 0.5)] = 1.0
        
        # Blue component: 0 in first half, 0→1 in third quarter, then 1
        mask3 = mask & (values >= 0.5) & (values < 0.75)
        b[mask3] = (values[mask3] - 0.5) * 4
        b[mask & (values >= 0.75)] = 1.0
        
        # Apply intensity reduction in the last quarter
        mask4 = mask & (values >= 0.75)
        intensity = 1.0 - (values[mask4] - 0.75) * 0.8
        
        r[mask4] *= intensity
        g[mask4] *= intensity
        b[mask4] *= intensity
        
        # Scale to 0-255 and set in the rgb_array
        rgb_array[..., 0] = (r * 255).astype(np.uint8)
        rgb_array[..., 1] = (g * 255).astype(np.uint8)
        rgb_array[..., 2] = (b * 255).astype(np.uint8)
    
    return rgb_array

def draw_mandelbrot():
    global current_pixels
    global colored_pixels
    if current_pixels is None:
        current_pixels = mandelbrot(HEIGHT, WIDTH, x_min, x_max, y_min, y_max, max_iter)
    
    # Always recalculate the colored pixels to reflect current color_mode and color_shift
    colored_pixels = color_map(current_pixels, max_iter, color_mode, color_shift)
    
    # Create surface from colored pixels
    surface = pygame.surfarray.make_surface(colored_pixels)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

def update_mandelbrot():
    global current_pixels
    global colored_pixels
    current_pixels = mandelbrot(HEIGHT, WIDTH, x_min, x_max, y_min, y_max, max_iter)
    colored_pixels = color_map(current_pixels, max_iter, color_mode, color_shift)
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
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                # Change color mode
                color_mode = (color_mode + 1) % 3
                draw_mandelbrot()
            elif event.key == pygame.K_LEFT:
                # Shift colors left
                color_shift = (color_shift - 0.1) % 1.0
                draw_mandelbrot()
            elif event.key == pygame.K_RIGHT:
                # Shift colors right
                color_shift = (color_shift + 0.1) % 1.0
                draw_mandelbrot()
            elif event.key == pygame.K_i:
                # Increase max iterations
                max_iter = min(max_iter * 2, 2000)
                update_mandelbrot()
            elif event.key == pygame.K_d:
                # Decrease max iterations
                max_iter = max(max_iter // 2, 50)
                update_mandelbrot()
            # Add key to display current parameters
            elif event.key == pygame.K_p:
                print(f"Current settings: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}, max_iter={max_iter}")
                print(f"Color mode: {color_mode}, Color shift: {color_shift}")
    
    # Initial draw
    if running and current_pixels is None:
        draw_mandelbrot()

pygame.quit()
