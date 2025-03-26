import pygame
import numpy as np
from colorsys import hsv_to_rgb
import sys
import traceback

try:
    # Initialize Pygame
    pygame.init()
    pygame.font.init()
    
    # Check if pygame is properly initialized
    if not pygame.display.get_init():
        print("Error: pygame.display could not be initialized.")
        sys.exit(1)
    
    # Set up display
    WIDTH = 600
    HEIGHT = 600  # Changed to match WIDTH for square window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mandelbrot Set")
    
    # Set up font for help text
    font = pygame.font.SysFont('Arial', 12)
    
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
    
    # UI visibility control
    show_ui_panels = False  # Initially hide the UI panels
    
    # Color palette settings
    color_mode = 2  # 0: Smooth colormap, 1: Classic rainbow, 2: Fire palette
    color_shift = 0.0  # color rotation value
    
    # Add a global variable to store colored pixels
    colored_pixels = None
    
    # Add a global variable to store the base surface
    base_surface = None
    
    # Cache for color lookup tables
    color_lookup_cache = {}
    
    # Zoom history stack to store previous viewports
    zoom_history = []
    # Store initial view as the first item in history
    zoom_history.append((x_min, x_max, y_min, y_max, max_iter))
    
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

    # Add a function to draw the top help message
    def draw_top_message():
        """Draw a help message at the top of the screen"""
        message = "Press H to toggle help and debug info"
        message_font = pygame.font.SysFont('Arial', 14, bold=True)
        text_surface = message_font.render(message, True, (255, 255, 255))
        
        # Create a semi-transparent background
        text_rect = text_surface.get_rect()
        text_rect.centerx = WIDTH // 2
        text_rect.top = 10
        
        bg_rect = text_rect.inflate(20, 10)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surface.set_alpha(180)
        bg_surface.fill((20, 20, 40))
        
        # Draw background and text
        screen.blit(bg_surface, bg_rect)
        screen.blit(text_surface, text_rect)
        
        # Add a subtle border
        pygame.draw.rect(screen, (100, 100, 150), bg_rect, 1)

    def draw_ui_panel():
        """Draw UI panels with matching styling"""
        # If panels are hidden, return immediately
        if not show_ui_panels:
            return
            
        # Help panel (left)
        help_texts = [
            "Controls:",
            "Left click and drag: Select zoom area",
            "Right click: Zoom out",
            "C: Change color mode",
            "Left/Right: Shift colors",
            "I/D: Increase/Decrease iterations", 
            "P: Print current settings",
            "Backspace: Zoom out",
            "H: Toggle help panels",
            "ESC: Exit"
        ]
        
        # Settings panel (right)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        
        # Get zoom level relative to initial view
        initial_width = 3.0  # Initial x_min=-2, x_max=1 width
        zoom_level = initial_width / width
        
        settings_texts = [
            "Current Settings:",
            f"Center: ({x_center:.6f}, {y_center:.6f})",
            f"Width: {width:.6f}",
            f"Zoom: {zoom_level:.2f}x",
            f"Iterations: {max_iter}",
            f"Color: {['Smooth', 'Rainbow', 'Fire'][color_mode]} (Shift: {color_shift:.1f})"
        ]
        
        # Determine the maximum number of lines to make panels the same height
        max_lines = max(len(help_texts), len(settings_texts))
        panel_height = max_lines * 15 + 10
        
        # Create title font
        title_font = pygame.font.SysFont('Arial', 13, bold=True)
        
        # Draw left panel (Help)
        left_bg_rect = pygame.Rect(10, HEIGHT - panel_height, 230, panel_height)
        left_bg_surface = pygame.Surface((left_bg_rect.width, left_bg_rect.height))
        left_bg_surface.set_alpha(200)
        left_bg_surface.fill((20, 20, 40))
        screen.blit(left_bg_surface, left_bg_rect)
        pygame.draw.rect(screen, (100, 100, 150), left_bg_rect, 1)
        
        # Draw right panel (Settings)
        right_bg_rect = pygame.Rect(WIDTH - 240, HEIGHT - panel_height, 230, panel_height)
        right_bg_surface = pygame.Surface((right_bg_rect.width, right_bg_rect.height))
        right_bg_surface.set_alpha(200)
        right_bg_surface.fill((20, 20, 40))
        screen.blit(right_bg_surface, right_bg_rect)
        pygame.draw.rect(screen, (100, 100, 150), right_bg_rect, 1)
        
        # Draw help text
        y_offset = HEIGHT - panel_height + 5
        for i, text in enumerate(help_texts):
            if i == 0:
                text_surface = title_font.render(text, True, (255, 255, 255))
            else:
                text_surface = font.render(text, True, (220, 220, 255))
            
            text_rect = text_surface.get_rect()
            text_rect.x = 15
            text_rect.y = y_offset
            screen.blit(text_surface, text_rect)
            y_offset += 15
        
        # Draw settings text
        y_offset = HEIGHT - panel_height + 5
        for i, text in enumerate(settings_texts):
            if i == 0:
                text_surface = title_font.render(text, True, (255, 255, 255))
            else:
                text_surface = font.render(text, True, (220, 220, 255))
            
            text_rect = text_surface.get_rect()
            text_rect.right = WIDTH - 15
            text_rect.y = y_offset
            screen.blit(text_surface, text_rect)
            y_offset += 15

    def draw_mandelbrot():
        global current_pixels, colored_pixels, base_surface
        if current_pixels is None:
            current_pixels = mandelbrot(HEIGHT, WIDTH, x_min, x_max, y_min, y_max, max_iter)
        
        # Always recalculate the colored pixels to reflect current color_mode and color_shift
        colored_pixels = color_map(current_pixels, max_iter, color_mode, color_shift)
        
        # Create and save the base surface for later use
        base_surface = pygame.surfarray.make_surface(colored_pixels)
        
        # Display the surface
        screen.blit(base_surface, (0, 0))
        
        # Always draw the top message
        draw_top_message()
        
        # Draw UI panels (help and settings) if enabled
        draw_ui_panel()
        
        pygame.display.flip()

    def update_mandelbrot():
        global current_pixels, colored_pixels, base_surface
        current_pixels = mandelbrot(HEIGHT, WIDTH, x_min, x_max, y_min, y_max, max_iter)
        colored_pixels = color_map(current_pixels, max_iter, color_mode, color_shift)
        draw_mandelbrot()

    def zoom_out():
        """Revert to the previous view from history"""
        global x_min, x_max, y_min, y_max, max_iter, current_pixels
        
        # Check if we have a previous view
        if len(zoom_history) > 1:  # Keep at least one item in history
            # Pop the most recent view from history
            previous_view = zoom_history.pop()
            x_min, x_max, y_min, y_max, max_iter = previous_view
            
            # Reset current pixels to force recalculation
            current_pixels = None
            update_mandelbrot()
            
            # Print current coordinates for debugging
            print(f"Zoomed out to: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
        else:
            print("No more zoom history available")

    def calculate_zoom_area(start_pos, current_pos):
        """Calculate the square zoom area from a selection rectangle"""
        # Calculate the original rectangle bounds drawn by the user
        rect_x1 = min(start_pos[0], current_pos[0])
        rect_x2 = max(start_pos[0], current_pos[0])
        rect_y1 = min(start_pos[1], current_pos[1])
        rect_y2 = max(start_pos[1], current_pos[1])
        
        # Find the center of the rectangle (as floating point for precision)
        center_x = (rect_x1 + rect_x2) / 2
        center_y = (rect_y1 + rect_y2) / 2
        
        # Get the longest dimension (width or height)
        rect_width = rect_x2 - rect_x1
        rect_height = rect_y2 - rect_y1
        zoom_length = max(rect_width, rect_height)
        
        # Calculate the square bounds centered at the original center point
        # Store as floats to maintain precision
        square_x1 = center_x - zoom_length / 2
        square_y1 = center_y - zoom_length / 2
        square_x2 = center_x + zoom_length / 2
        square_y2 = center_y + zoom_length / 2
        
        # Calculate the complex plane coordinates of the center point
        # Use floating point division for precision
        complex_center_x = x_min + (x_max - x_min) * center_x / WIDTH
        # For y coordinate, remember screen y increases downward but complex plane y increases upward
        complex_center_y = y_max - (y_max - y_min) * center_y / HEIGHT
        
        # Calculate the size in the complex plane that will be zoomed into
        complex_width = (x_max - x_min) * zoom_length / WIDTH
        
        # Calculate zoom factor based on the zoom length relative to screen size
        zoom_factor = WIDTH / zoom_length
        
        # Cap the zoom factor to avoid extreme zooms
        zoom_factor = min(zoom_factor, 10.0)
        
        # Calculate the new boundaries in the complex plane
        new_x_min = complex_center_x - complex_width / 2
        new_x_max = complex_center_x + complex_width / 2
        new_y_min = complex_center_y - complex_width / 2
        new_y_max = complex_center_y + complex_width / 2
        
        return {
            "rect": (rect_x1, rect_y1, rect_width, rect_height),
            "screen_center": (center_x, center_y),
            "complex_center": (complex_center_x, complex_center_y),
            "zoom_length": zoom_length,
            "zoom_factor": zoom_factor,
            "square": (square_x1, square_y1, square_x2, square_y2),
            "complex_width": complex_width,
            "new_bounds": (new_x_min, new_x_max, new_y_min, new_y_max)
        }

    def draw_selection_rectangle():
        """Draw the current Mandelbrot set with a selection rectangle overlay"""
        if base_surface is None or start_pos is None or current_pos is None:
            return
        
        # Create a copy of the base surface to avoid modifying it
        screen.blit(base_surface, (0, 0))
        
        # Calculate zoom area
        zoom_area = calculate_zoom_area(start_pos, current_pos)
        
        # Extract values from zoom area
        rect_x1, rect_y1, rect_width, rect_height = zoom_area["rect"]
        center_x, center_y = zoom_area["screen_center"]
        square_x1, square_y1, square_x2, square_y2 = zoom_area["square"]
        complex_center_x, complex_center_y = zoom_area["complex_center"]
        zoom_length = zoom_area["zoom_length"]
        complex_width = zoom_area["complex_width"]
        zoom_factor = zoom_area["zoom_factor"]
        
        # Draw the original rectangle (dimmed)
        original_rect = pygame.Rect(rect_x1, rect_y1, rect_width, rect_height)
        pygame.draw.rect(screen, (100, 100, 100), original_rect, 1)
        
        # Draw the square that will be zoomed into (highlighted)
        # Convert floating point coordinates to integers for drawing
        square_rect = pygame.Rect(
            int(square_x1), 
            int(square_y1), 
            int(square_x2 - square_x1), 
            int(square_y2 - square_y1)
        )
        pygame.draw.rect(screen, (255, 255, 255), square_rect, 2)
        
        # Display zoom coordinates in the square
        zoom_info_font = pygame.font.SysFont('Arial', 10)
        
        # Show the center coordinates
        zoom_text1 = zoom_info_font.render(
            f"Center: ({complex_center_x:.4f}, {complex_center_y:.4f})", 
            True, 
            (255, 255, 255)
        )
        zoom_text_rect1 = zoom_text1.get_rect()
        zoom_text_rect1.centerx = int(center_x)
        zoom_text_rect1.y = int(square_y1) + 4
        
        # Show the zoom factor
        zoom_text2 = zoom_info_font.render(
            f"Zoom: {zoom_factor:.1f}x", 
            True, 
            (255, 255, 255)
        )
        zoom_text_rect2 = zoom_text2.get_rect()
        zoom_text_rect2.centerx = int(center_x)
        zoom_text_rect2.y = int(square_y1) + 19
        
        # Add a small background behind the text for readability
        text_bg_rect1 = zoom_text_rect1.inflate(6, 4)
        text_bg1 = pygame.Surface((text_bg_rect1.width, text_bg_rect1.height))
        text_bg1.set_alpha(180) 
        text_bg1.fill((0, 0, 0))
        screen.blit(text_bg1, text_bg_rect1)
        
        text_bg_rect2 = zoom_text_rect2.inflate(6, 4)
        text_bg2 = pygame.Surface((text_bg_rect2.width, text_bg_rect2.height))
        text_bg2.set_alpha(180) 
        text_bg2.fill((0, 0, 0))
        screen.blit(text_bg2, text_bg_rect2)
        
        # Draw the text
        screen.blit(zoom_text1, zoom_text_rect1)
        screen.blit(zoom_text2, zoom_text_rect2)
        
        # Draw a crosshair at the center
        pygame.draw.line(screen, (255, 255, 0), (int(center_x) - 5, int(center_y)), (int(center_x) + 5, int(center_y)), 1)
        pygame.draw.line(screen, (255, 255, 0), (int(center_x), int(center_y) - 5), (int(center_x), int(center_y) + 5), 1)
        
        # Always draw the top message
        draw_top_message()
        
        # Draw UI panels if enabled
        draw_ui_panel()
        
        pygame.display.flip()

    def zoom_to_selection(start_pos, current_pos):
        """Zoom to the selected area"""
        global x_min, x_max, y_min, y_max, current_pixels
        
        # Calculate zoom area
        zoom_area = calculate_zoom_area(start_pos, current_pos)
        
        # Extract the values we need for zooming
        complex_center_x, complex_center_y = zoom_area["complex_center"]
        zoom_factor = zoom_area["zoom_factor"]
        new_x_min, new_x_max, new_y_min, new_y_max = zoom_area["new_bounds"]
        
        # Save current view to history
        zoom_history.append((x_min, x_max, y_min, y_max, max_iter))
        
        # Limit history size to prevent memory issues
        if len(zoom_history) > 50:
            zoom_history.pop(0)
        
        # Log what we're about to do
        print(f"Zooming to center: ({complex_center_x:.6f}, {complex_center_y:.6f})")
        print(f"New bounds: x=[{new_x_min:.6f}, {new_x_max:.6f}], y=[{new_y_min:.6f}, {new_y_max:.6f}]")
        print(f"Zoom factor: {zoom_factor:.2f}x")
        
        # Apply the new bounds directly
        x_min, x_max = new_x_min, new_x_max
        y_min, y_max = new_y_min, new_y_max
        
        # Reset current pixels to force recalculation
        current_pixels = None
        update_mandelbrot()

    # Print a message indicating successful initialization
    print("Mandelbrot Viewer successfully initialized...")
    
    # Main loop
    running = True
    clock = pygame.time.Clock()  # Add a clock to control the frame rate
    
    # Ensure we start with a rendered Mandelbrot
    current_pixels = None
    draw_mandelbrot()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    drawing = True
                    start_pos = event.pos
                    current_pos = event.pos
                elif event.button == 3:  # Right click - zoom out to previous view
                    zoom_out()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and drawing:  # Left click release
                    drawing = False
                    if start_pos and current_pos:
                        # Calculate zoom area
                        zoom_area = calculate_zoom_area(start_pos, current_pos)
                        
                        # Extract zoom length to check minimum size
                        zoom_length = zoom_area["zoom_length"]
                        
                        # Ensure the rectangle has a minimum size
                        if zoom_length < 5:
                            # Rectangle too small, ignore and restore original view
                            start_pos = None
                            current_pos = None
                            # Simply redisplay the base surface without the selection rectangle
                            if base_surface is not None:
                                screen.blit(base_surface, (0, 0))
                                pygame.display.flip()
                            continue
                        
                        # Use our zoom_to_selection function for consistent zooming
                        zoom_to_selection(start_pos, current_pos)
                        
                        # Reset drawing variables
                        start_pos = None
                        current_pos = None
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    current_pos = event.pos
                    # Draw the selection rectangle without recalculating the Mandelbrot set
                    draw_selection_rectangle()
            elif event.type == pygame.KEYDOWN:
                # Cancel any active selection when pressing any key
                if drawing:
                    drawing = False
                    start_pos = None
                    current_pos = None
                    
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
                elif event.key == pygame.K_BACKSPACE:
                    # Alternative way to zoom out
                    zoom_out()
                # Add key to display current parameters
                elif event.key == pygame.K_p:
                    print(f"Current settings: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}, max_iter={max_iter}")
                    print(f"Color mode: {color_mode}, Color shift: {color_shift}")
                    print(f"Zoom history depth: {len(zoom_history)}")
                # Toggle UI panels with H key
                elif event.key == pygame.K_h:
                    # Toggle UI visibility without global declaration
                    globals()['show_ui_panels'] = not show_ui_panels
                    draw_mandelbrot()
                # Add escape key to exit
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Limit the frame rate
        clock.tick(30)
    
    print("Exiting Mandelbrot Viewer...")
    pygame.quit()

except Exception as e:
    print(f"Error: {e}")
    print(traceback.format_exc())
    pygame.quit()
    sys.exit(1)
