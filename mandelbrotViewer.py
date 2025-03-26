import pygame
import numpy as np
from colorsys import hsv_to_rgb
import sys
import traceback

# Set this to False to disable Numba even if available
USE_NUMBA = True

# Set this to True if the Numba output appears rotated compared to NumPy
TRANSPOSE_NUMBA_OUTPUT = True

# Add Numba import for JIT compilation
try:
    import numba
    from numba import jit, prange
    HAVE_NUMBA = True
    print("Numba found - JIT compilation is available")
    if USE_NUMBA:
        print("Numba JIT compilation enabled for better performance")
    else:
        print("Numba disabled by user configuration - using standard NumPy")
except ImportError:
    HAVE_NUMBA = False
    USE_NUMBA = False
    print("Numba not found - using standard Python (install Numba for better performance)")

try:
    # Initialize Pygame
    pygame.init()
    pygame.font.init()
    
    # Check if pygame is properly initialized
    if not pygame.display.get_init():
        print("Error: pygame.display could not be initialized.")
        sys.exit(1)
    
    # Get the user's display info
    display_info = pygame.display.Info()
    screen_width = display_info.current_w
    screen_height = display_info.current_h
    
    # Calculate the largest square that fits on the screen
    # Leave some margin for taskbars and window borders
    margin = 100  # Pixels of margin to leave around the edges
    max_size = min(screen_width - margin, screen_height - margin)
    
    # Set up display with the largest square possible
    WIDTH = max_size
    HEIGHT = max_size
    print(f"Setting up display with dimensions: {WIDTH}x{HEIGHT}")
    
    # Calculate a scale factor for UI elements based on resolution
    # This ensures UI elements scale appropriately with different resolutions
    SCALE_FACTOR = max(1.0, WIDTH / 600)  # Base scale on a 600x600 reference size
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mandelbrot Set")
    
    # Show loading screen while Numba compiles (if available and enabled)
    if USE_NUMBA and HAVE_NUMBA:
        # Fill screen with dark background
        screen.fill((20, 20, 40))
        
        # Create loading text
        loading_font_size = int(24 * SCALE_FACTOR)
        loading_font = pygame.font.SysFont('Arial', loading_font_size, bold=True)
        loading_text = loading_font.render("Initializing Numba JIT compiler...", True, (255, 255, 255))
        
        # Add explanation text
        info_font_size = int(14 * SCALE_FACTOR)
        info_font = pygame.font.SysFont('Arial', info_font_size)
        info_text1 = info_font.render("This may take a few seconds on first run", True, (200, 200, 200))
        info_text2 = info_font.render("Subsequent runs will be faster due to caching", True, (200, 200, 200))
        
        # Position text in center of screen
        loading_rect = loading_text.get_rect(center=(WIDTH//2, HEIGHT//2 - int(20 * SCALE_FACTOR)))
        info_rect1 = info_text1.get_rect(center=(WIDTH//2, HEIGHT//2 + int(20 * SCALE_FACTOR)))
        info_rect2 = info_text2.get_rect(center=(WIDTH//2, HEIGHT//2 + int(50 * SCALE_FACTOR)))
        
        # Draw text
        screen.blit(loading_text, loading_rect)
        screen.blit(info_text1, info_rect1)
        screen.blit(info_text2, info_rect2)
        
        # Update display to show loading screen
        pygame.display.flip()
    
    # Set up font for help text with size scaled to the resolution
    base_font_size = 12
    scaled_font_size = int(base_font_size * SCALE_FACTOR)
    font = pygame.font.SysFont('Arial', scaled_font_size)
    
    # Initial view parameters
    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5  # These values already maintain a square aspect ratio
    max_iter = 100
    
    # High-quality rendering flag
    high_quality_mode = False
    high_quality_multiplier = 4  # 4x more iterations in high quality mode
    
    # Debug modes
    debug_coordinates = False    # Debug coordinate mappings
    force_numpy = False          # Force NumPy implementation (disable Numba)
    
    # Store the current Mandelbrot set
    current_pixels = None
    
    # Rectangle drawing variables
    drawing = False
    start_pos = None
    current_pos = None
    
    # UI visibility control
    show_ui_panels = False  # Initially hide the UI panels
    
    # Color palette settings
    color_mode = 2  # 0: Smooth colormap, 1: Classic rainbow, 2: Fire palette, 3: Electric blue, 4: Twilight, 5: Greyscale, 6: Neon, 7: Deep Ocean, 8: Vintage
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
    
    # Define a Numba-optimized function for calculating mandelbrot escape times
    if HAVE_NUMBA:
        @jit(nopython=True, parallel=True, fastmath=True, cache=True)
        def mandelbrot_kernel(x, y, max_iter):
            """Numba-optimized kernel for Mandelbrot calculation"""
            height, width = len(y), len(x)
            output = np.zeros((height, width), dtype=np.int32)
            
            # Use prange for parallel computation across rows
            # In NumPy implementation: c[i,j] = x[j] + 1j * y[i]
            # We need to match the same orientation here
            for i in prange(height):
                for j in range(width):
                    # In NumPy broadcasting, x is spread across columns and y across rows
                    # So coordinate (i,j) corresponds to complex number (x[j], y[i])
                    c = complex(x[j], y[i])
                    z = 0.0j
                    
                    # Compute how quickly this point escapes
                    for iteration in range(max_iter):
                        z = z**2 + c
                        if (z.real*z.real + z.imag*z.imag) >= 4.0:
                            output[i, j] = iteration
                            break
                    else:
                        output[i, j] = max_iter
                    
            return output
    
    def mandelbrot(h, w, x_min, x_max, y_min, y_max, max_iter):
        """Calculate the Mandelbrot set"""
        # Use float64 for better precision
        # Set up the x and y ranges with correct orientation:
        # x increases from left to right: x_min at left, x_max at right
        # y is flipped for screen coordinates: y_max at top, y_min at bottom
        # This is because in the complex plane y increases upward, but screen coordinates increase downward
        x = np.linspace(x_min, x_max, w, dtype=np.float64)
        y = np.linspace(y_max, y_min, h, dtype=np.float64)  # Note: y inverted for screen coordinates
        
        if USE_NUMBA and HAVE_NUMBA and not force_numpy:
            # Use Numba-accelerated kernel
            # We pass x and y arrays directly to the kernel
            # The kernel maps (i,j) to complex number (x[j], y[i]), matching the NumPy orientation
            output = mandelbrot_kernel(x, y, max_iter)
            
            # If the output is rotated compared to NumPy, transpose it
            if TRANSPOSE_NUMBA_OUTPUT:
                output = output.T
        else:
            # Use the original NumPy implementation
            # Broadcasting creates a 2D grid of complex numbers where:
            # c[i,j] = x[j] + 1j * y[i]
            # This makes c[0,0] = x[0] + 1j * y[0] = top-left corner
            # And maps screen coordinates directly to the complex plane
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
        
        return output

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

    # Define a Numba-optimized function for smooth coloring
    if HAVE_NUMBA:
        @jit(nopython=True, fastmath=True, cache=True)
        def apply_smooth_colormap(iterations, max_iter, cmap, mask, shift=0.0):
            """Apply smooth colormap with Numba acceleration"""
            height, width = iterations.shape
            rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            for i in range(height):
                for j in range(width):
                    if not mask[i, j]:  # Only process points outside the set
                        # Normalize and apply logarithmic smoothing
                        norm_value = iterations[i, j] / max_iter
                        log_value = np.log(norm_value * 0.5 + 0.5) / np.log(1.5)
                        
                        # Apply shift
                        shifted_value = (log_value + shift) % 1.0
                        
                        # Map to colormap index
                        index = min(255, max(0, int(shifted_value * 255)))
                        
                        # Assign color
                        rgb_array[i, j, 0] = cmap[index, 0]
                        rgb_array[i, j, 1] = cmap[index, 1]
                        rgb_array[i, j, 2] = cmap[index, 2]
            
            return rgb_array

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
            
            if USE_NUMBA and HAVE_NUMBA and mode == 0:
                # Use Numba-accelerated coloring for the smooth colormap
                rgb_array = apply_smooth_colormap(iterations, max_iter, cmap, in_set, shift)
            else:
                # Use the original implementation
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
        
        elif mode == 1:  # Classic rainbow palette using sine waves
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
        
        elif mode == 3:  # Electric blue
            # Electric blue with vibrant cyan to deep blue transitions
            norm_values = np.zeros_like(iterations, dtype=np.float64)
            norm_values[mask] = iterations[mask] / max_iter
            
            # Apply shift
            values = (norm_values + shift) % 1.0
            
            # Allocate arrays
            r = np.zeros_like(values)
            g = np.zeros_like(values)
            b = np.zeros_like(values)
            
            # Blue component: Always high but with some variation
            b[mask] = 0.7 + 0.3 * np.sin(values[mask] * np.pi * 4)
            
            # Green component: Higher in the middle
            g[mask] = 0.4 * np.sin(values[mask] * np.pi * 2) ** 2
            g[mask & (values < 0.5)] += 0.2 + 0.6 * values[mask & (values < 0.5)]
            
            # Red component: Low but with some highlights
            r[mask] = 0.1 * np.sin(values[mask] * np.pi * 8) ** 2
            
            # Add white spark effect for lower values (recently escaped points)
            low_values_mask = mask & (values < 0.15)
            spark = 1.0 - values[low_values_mask] / 0.15
            r[low_values_mask] += spark
            g[low_values_mask] += spark
            
            # Scale to 0-255 and convert to uint8
            rgb_array[..., 0] = (np.clip(r, 0, 1) * 255).astype(np.uint8)
            rgb_array[..., 1] = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            rgb_array[..., 2] = (np.clip(b, 0, 1) * 255).astype(np.uint8)
        
        elif mode == 4:  # Twilight palette (purple to orange)
            # Twilight-inspired color gradient
            norm_values = np.zeros_like(iterations, dtype=np.float64)
            norm_values[mask] = iterations[mask] / max_iter
            
            # Apply shift
            values = (norm_values + shift) % 1.0
            
            # Allocate arrays
            r = np.zeros_like(values)
            g = np.zeros_like(values)
            b = np.zeros_like(values)
            
            # Define the twilight gradient: deep purples to oranges
            # Purple to blue
            mask1 = mask & (values < 0.3)
            r[mask1] = 0.5 + 0.2 * values[mask1] / 0.3
            g[mask1] = 0.2 * values[mask1] / 0.3
            b[mask1] = 0.8 - 0.2 * values[mask1] / 0.3
            
            # Blue to teal
            mask2 = mask & (values >= 0.3) & (values < 0.5)
            r[mask2] = 0.7 * (values[mask2] - 0.3) / 0.2
            g[mask2] = 0.2 + 0.4 * (values[mask2] - 0.3) / 0.2
            b[mask2] = 0.6 + 0.2 * (values[mask2] - 0.3) / 0.2
            
            # Teal to golden
            mask3 = mask & (values >= 0.5) & (values < 0.7)
            r[mask3] = 0.7 + 0.3 * (values[mask3] - 0.5) / 0.2
            g[mask3] = 0.6 + 0.2 * (values[mask3] - 0.5) / 0.2
            b[mask3] = 0.8 - 0.8 * (values[mask3] - 0.5) / 0.2
            
            # Golden to deep red
            mask4 = mask & (values >= 0.7)
            r[mask4] = 1.0
            g[mask4] = 0.8 - 0.8 * (values[mask4] - 0.7) / 0.3
            b[mask4] = 0.0
            
            # Scale to 0-255 and convert to uint8
            rgb_array[..., 0] = (np.clip(r, 0, 1) * 255).astype(np.uint8)
            rgb_array[..., 1] = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            rgb_array[..., 2] = (np.clip(b, 0, 1) * 255).astype(np.uint8)
        
        elif mode == 5:  # Grayscale with smooth log mapping
            # Simple grayscale palette
            norm_values = np.zeros_like(iterations, dtype=np.float64)
            norm_values[mask] = iterations[mask] / max_iter
            
            # Apply logarithmic smoothing
            log_values = np.zeros_like(norm_values)
            log_values[mask] = np.log(norm_values[mask] * 0.5 + 0.5) / np.log(1.5)
            
            # Apply shift by cycling the values (0-1 range)
            values = (log_values + shift) % 1.0
            
            # Set all RGB channels to the same value for grayscale
            # Use a bit of contrast enhancement
            value = np.zeros_like(values)
            value[mask] = np.clip(values[mask] * 1.2, 0, 1)
            
            # Scale to 0-255 and convert to uint8
            rgb_array[..., 0] = (value * 255).astype(np.uint8)
            rgb_array[..., 1] = (value * 255).astype(np.uint8)
            rgb_array[..., 2] = (value * 255).astype(np.uint8)
        
        elif mode == 6:  # Neon palette with bright glows and dark backgrounds
            norm_values = np.zeros_like(iterations, dtype=np.float64)
            norm_values[mask] = iterations[mask] / max_iter
            
            # Apply shift
            values = (norm_values + shift) % 1.0
            
            # Allocate arrays
            r = np.zeros_like(values)
            g = np.zeros_like(values)
            b = np.zeros_like(values)
            
            # Create a striated neon effect based on value
            phase = np.zeros_like(values)
            phase[mask] = values[mask] * 15  # Multiple cycles for striations
            
            # Use sine waves with different frequencies for color pulsing
            r[mask] = 0.5 * np.sin(phase[mask] * 1.0 + 0.0) + 0.5
            g[mask] = 0.5 * np.sin(phase[mask] * 1.0 + 2.0) + 0.5
            b[mask] = 0.5 * np.sin(phase[mask] * 1.0 + 4.0) + 0.5
            
            # Apply a glow effect - brighten colors based on original value
            # This makes recently escaped points brighter
            glow_mask = mask & (values < 0.2)
            
            if np.any(glow_mask):
                glow_strength = 1.0 - values[glow_mask] / 0.2
                
                # Calculate glow phase for each pixel in the glow mask
                glow_phase = np.zeros_like(values)
                glow_phase[glow_mask] = (values[glow_mask] * 3.0) % 3.0
                
                # Create separate masks for each color channel's glow
                r_glow = glow_mask & (glow_phase < 1.0)
                g_glow = glow_mask & (glow_phase >= 1.0) & (glow_phase < 2.0)
                b_glow = glow_mask & (glow_phase >= 2.0)
                
                # Apply glows to respective channels
                if np.any(r_glow):
                    r[r_glow] = r[r_glow] * 0.5 + 0.5 * (1.0 - values[r_glow] / 0.2)
                
                if np.any(g_glow):
                    g[g_glow] = g[g_glow] * 0.5 + 0.5 * (1.0 - values[g_glow] / 0.2)
                
                if np.any(b_glow):
                    b[b_glow] = b[b_glow] * 0.5 + 0.5 * (1.0 - values[b_glow] / 0.2)
            
            # Add secondary glow around edges with sharp contrast
            edge_mask = mask & (values > 0.2) & (values < 0.25)
            if np.any(edge_mask):
                edge_intensity = (values[edge_mask] - 0.2) / 0.05
                
                r[edge_mask] = r[edge_mask] * 0.7 + (1.0 - edge_intensity) * 0.3
                g[edge_mask] = g[edge_mask] * 0.7 + (1.0 - edge_intensity) * 0.3
                b[edge_mask] = b[edge_mask] * 0.7 + (1.0 - edge_intensity) * 0.3
            
            # Darken the background (high iteration values)
            dark_mask = mask & (values >= 0.25)
            if np.any(dark_mask):
                darkness = np.minimum(1.0, (values[dark_mask] - 0.25) * 2)
                
                r[dark_mask] = r[dark_mask] * (1.0 - darkness * 0.8)
                g[dark_mask] = g[dark_mask] * (1.0 - darkness * 0.8)
                b[dark_mask] = b[dark_mask] * (1.0 - darkness * 0.8)
            
            # Scale to 0-255 and convert to uint8
            rgb_array[..., 0] = (np.clip(r, 0, 1) * 255).astype(np.uint8)
            rgb_array[..., 1] = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            rgb_array[..., 2] = (np.clip(b, 0, 1) * 255).astype(np.uint8)
        
        elif mode == 7:  # Deep Ocean
            # Ocean-inspired palette with blues and teals
            norm_values = np.zeros_like(iterations, dtype=np.float64)
            norm_values[mask] = iterations[mask] / max_iter
            
            # Apply logarithmic smoothing to enhance contrast
            log_values = np.zeros_like(norm_values)
            log_values[mask] = np.log(norm_values[mask] * 0.5 + 0.5) / np.log(1.5)
            
            # Apply shift
            values = (log_values + shift) % 1.0
            
            # Allocate arrays
            r = np.zeros_like(values)
            g = np.zeros_like(values)
            b = np.zeros_like(values)
            
            # Blue-green gradient for ocean depths
            # Start with darkest deep blues (lowest values)
            r[mask] = 0.0
            g[mask] = 0.0
            b[mask] = 0.3 + values[mask] * 0.3  # 0.3-0.6 range for blue
            
            # Add green/teal for mid-range values
            midrange_mask = mask & (values > 0.3) & (values < 0.7)
            mid_intensity = (values[midrange_mask] - 0.3) / 0.4
            g[midrange_mask] = mid_intensity * 0.6
            
            # Surface waters with light teal and white foam for high values
            surface_mask = mask & (values >= 0.7) & (values < 0.9)
            surface_intensity = (values[surface_mask] - 0.7) / 0.2
            r[surface_mask] = surface_intensity * 0.3
            g[surface_mask] = 0.6 + surface_intensity * 0.3
            b[surface_mask] = 0.6 + surface_intensity * 0.3
            
            # White foam/highlight at the very highest values
            foam_mask = mask & (values >= 0.9)
            foam_intensity = (values[foam_mask] - 0.9) / 0.1
            
            r[foam_mask] = 0.3 + foam_intensity * 0.7
            g[foam_mask] = 0.9 + foam_intensity * 0.1
            b[foam_mask] = 0.9 + foam_intensity * 0.1
            
            # Add shimmer effect with sine waves
            shimmer_mask = mask & (values > 0.5)
            shimmer = 0.05 * np.sin(values[shimmer_mask] * 50)
            r[shimmer_mask] += shimmer
            g[shimmer_mask] += shimmer
            b[shimmer_mask] += shimmer
            
            # Scale to 0-255 and convert to uint8
            rgb_array[..., 0] = (np.clip(r, 0, 1) * 255).astype(np.uint8)
            rgb_array[..., 1] = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            rgb_array[..., 2] = (np.clip(b, 0, 1) * 255).astype(np.uint8)
        
        elif mode == 8:  # Vintage/Sepia
            # Vintage/sepia tones with a worn look
            norm_values = np.zeros_like(iterations, dtype=np.float64)
            norm_values[mask] = iterations[mask] / max_iter
            
            # Apply shift
            values = (norm_values + shift) % 1.0
            
            # Create base grayscale value with contrast
            gray = np.zeros_like(values)
            gray[mask] = values[mask] ** 0.8  # Slight gamma adjustment for contrast
            
            # Allocate arrays 
            r = np.zeros_like(values)
            g = np.zeros_like(values)
            b = np.zeros_like(values)
            
            # Apply sepia toning - different multipliers for RGB
            r[mask] = np.clip(gray[mask] * 1.2, 0, 1)  # More red
            g[mask] = np.clip(gray[mask] * 0.9, 0, 1)  # Medium green 
            b[mask] = np.clip(gray[mask] * 0.6, 0, 1)  # Less blue
            
            # Add a vignette effect (darker at edges)
            # This would require knowing pixel positions, so we simulate with value
            vignette = np.zeros_like(values)
            vignette[mask] = 1.0 - 0.2 * np.sin(values[mask] * np.pi) ** 2
            
            r[mask] *= vignette[mask]
            g[mask] *= vignette[mask]
            b[mask] *= vignette[mask]
            
            # Add some "aging" noise to simulate vintage look
            # Use a deterministic pattern based on values to avoid pure randomness
            aging_effect = np.zeros_like(values)
            aging_effect[mask] = 0.05 * np.sin(values[mask] * 37.0) * np.sin(values[mask] * 23.0)
            
            r[mask] += aging_effect[mask]
            g[mask] += aging_effect[mask]
            b[mask] += aging_effect[mask]
            
            # Scale to 0-255 and convert to uint8
            rgb_array[..., 0] = (np.clip(r, 0, 1) * 255).astype(np.uint8)
            rgb_array[..., 1] = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            rgb_array[..., 2] = (np.clip(b, 0, 1) * 255).astype(np.uint8)
        
        return rgb_array

    # Add a function to draw the top help message
    def draw_top_message():
        """Draw a help message at the top of the screen"""
        message = "Press H to toggle help and debug info"
        message_font_size = int(14 * SCALE_FACTOR)
        message_font = pygame.font.SysFont('Arial', message_font_size, bold=True)
        text_surface = message_font.render(message, True, (255, 255, 255))
        
        # Create a semi-transparent background
        text_rect = text_surface.get_rect()
        text_rect.centerx = WIDTH // 2
        text_rect.top = int(10 * SCALE_FACTOR)
        
        bg_rect = text_rect.inflate(int(20 * SCALE_FACTOR), int(10 * SCALE_FACTOR))
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surface.set_alpha(180)
        bg_surface.fill((20, 20, 40))
        
        # Draw background and text
        screen.blit(bg_surface, bg_rect)
        screen.blit(text_surface, text_rect)
        
        # Add a subtle border
        pygame.draw.rect(screen, (100, 100, 150), bg_rect, max(1, int(SCALE_FACTOR)))

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
            "Q: Toggle high quality mode",
            "X: Toggle debug mode",
            "N: Toggle Numba/NumPy",
            "T: Toggle Numba transpose",
            "R: Reset view",
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
        
        # Get the name of the current color palette
        color_names = ['Smooth', 'Rainbow', 'Fire', 'Electric Blue', 'Twilight', 'Grayscale', 'Neon', 'Deep Ocean', 'Vintage']
        
        # Add high quality indicator to iteration count
        quality_text = "HQ" if high_quality_mode else "Standard"
        hq_iterations = max_iter * high_quality_multiplier if high_quality_mode else max_iter
        
        # Debug mode indicator
        debug_text = "Debug ON" if debug_coordinates else "Debug OFF"
        
        # Implementation indicator
        if HAVE_NUMBA:
            if USE_NUMBA:
                impl_text = "NumPy" if force_numpy else "Numba"
            else:
                impl_text = "NumPy (Numba disabled in config)"
        else:
            impl_text = "NumPy (Numba not available)"
        
        settings_texts = [
            "Current Settings:",
            f"Center: ({x_center:.6f}, {y_center:.6f})",
            f"Width: {width:.6f}",
            f"Zoom: {zoom_level:.2f}x",
            f"Iterations: {hq_iterations} ({quality_text})",
            f"Color: {color_names[color_mode]} (Shift: {color_shift:.1f})",
            f"Debug: {debug_text}",
            f"Implementation: {impl_text}",
            f"Resolution: {WIDTH}x{HEIGHT}"
        ]
        
        # Determine the maximum number of lines to make panels the same height
        max_lines = max(len(help_texts), len(settings_texts))
        panel_height = int(max_lines * 15 * SCALE_FACTOR + 10 * SCALE_FACTOR)
        panel_width = int(230 * SCALE_FACTOR)
        
        # Create title font
        title_font_size = int(13 * SCALE_FACTOR)
        title_font = pygame.font.SysFont('Arial', title_font_size, bold=True)
        
        # Draw left panel (Help)
        margin = int(10 * SCALE_FACTOR)
        left_bg_rect = pygame.Rect(margin, HEIGHT - panel_height - margin, panel_width, panel_height)
        left_bg_surface = pygame.Surface((left_bg_rect.width, left_bg_rect.height))
        left_bg_surface.set_alpha(200)
        left_bg_surface.fill((20, 20, 40))
        screen.blit(left_bg_surface, left_bg_rect)
        pygame.draw.rect(screen, (100, 100, 150), left_bg_rect, max(1, int(SCALE_FACTOR / 2)))
        
        # Draw right panel (Settings)
        right_bg_rect = pygame.Rect(WIDTH - panel_width - margin, HEIGHT - panel_height - margin, panel_width, panel_height)
        right_bg_surface = pygame.Surface((right_bg_rect.width, right_bg_rect.height))
        right_bg_surface.set_alpha(200)
        right_bg_surface.fill((20, 20, 40))
        screen.blit(right_bg_surface, right_bg_rect)
        pygame.draw.rect(screen, (100, 100, 150), right_bg_rect, max(1, int(SCALE_FACTOR / 2)))
        
        # Calculate line height based on scale
        line_height = int(15 * SCALE_FACTOR)
        
        # Draw help text
        y_offset = HEIGHT - panel_height + int(5 * SCALE_FACTOR)
        for i, text in enumerate(help_texts):
            if i == 0:
                text_surface = title_font.render(text, True, (255, 255, 255))
            else:
                text_surface = font.render(text, True, (220, 220, 255))
            
            text_rect = text_surface.get_rect()
            text_rect.x = margin + int(5 * SCALE_FACTOR)
            text_rect.y = y_offset
            screen.blit(text_surface, text_rect)
            y_offset += line_height
        
        # Draw settings text
        y_offset = HEIGHT - panel_height + int(5 * SCALE_FACTOR)
        for i, text in enumerate(settings_texts):
            if i == 0:
                text_surface = title_font.render(text, True, (255, 255, 255))
            else:
                text_surface = font.render(text, True, (220, 220, 255))
            
            text_rect = text_surface.get_rect()
            text_rect.right = WIDTH - margin - int(5 * SCALE_FACTOR)
            text_rect.y = y_offset
            screen.blit(text_surface, text_rect)
            y_offset += line_height

    def draw_mandelbrot():
        global current_pixels, colored_pixels, base_surface
        if current_pixels is None:
            # Calculate the effective iterations based on quality mode
            effective_iter = max_iter * high_quality_multiplier if high_quality_mode else max_iter
            current_pixels = mandelbrot(HEIGHT, WIDTH, x_min, x_max, y_min, y_max, effective_iter)
        
        # Always recalculate the colored pixels to reflect current color_mode and color_shift
        # Use the effective iterations for coloring
        effective_iter = max_iter * high_quality_multiplier if high_quality_mode else max_iter
        colored_pixels = color_map(current_pixels, effective_iter, color_mode, color_shift)
        
        # Create and save the base surface for later use
        # Ensure the surface is in the correct format for pygame
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
        # Calculate the effective iterations based on quality mode
        effective_iter = max_iter * high_quality_multiplier if high_quality_mode else max_iter
        current_pixels = mandelbrot(HEIGHT, WIDTH, x_min, x_max, y_min, y_max, effective_iter)
        colored_pixels = color_map(current_pixels, effective_iter, color_mode, color_shift)
        draw_mandelbrot()

    def toggle_quality_mode():
        """Toggle between standard and high quality rendering"""
        global high_quality_mode, current_pixels
        high_quality_mode = not high_quality_mode
        
        # Force recalculation with new quality setting
        current_pixels = None
        
        # Update status in console
        if high_quality_mode:
            effective_iter = max_iter * high_quality_multiplier
            print(f"High quality mode enabled: {effective_iter} iterations")
        else:
            print(f"Standard quality mode: {max_iter} iterations")
        
        # Update the display
        update_mandelbrot()

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
            effective_iter = max_iter * high_quality_multiplier if high_quality_mode else max_iter
            print(f"Zoomed out to: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            print(f"Using {effective_iter} iterations ({max_iter} base)")
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
        
        # Ensure zoom_length is not zero to prevent division by zero later
        # Set a minimum size of 5 pixels
        zoom_length = max(5.0, zoom_length)
        
        # Calculate the square bounds centered at the original center point
        # Store as floats to maintain precision
        square_x1 = center_x - zoom_length / 2
        square_y1 = center_y - zoom_length / 2
        square_x2 = center_x + zoom_length / 2
        square_y2 = center_y + zoom_length / 2
        
        # Ensure square is within screen bounds
        square_x1 = max(0, square_x1)
        square_y1 = max(0, square_y1)
        square_x2 = min(WIDTH, square_x2)
        square_y2 = min(HEIGHT, square_y2)
        
        # Recalculate center if square was adjusted
        center_x = (square_x1 + square_x2) / 2
        center_y = (square_y1 + square_y2) / 2
        
        # Map the center point from screen coordinates to complex plane
        # With our updated mandelbrot function, we directly map:
        # - x from 0 to WIDTH maps to x_min to x_max
        # - y from 0 to HEIGHT maps to y_max to y_min (inverted)
        
        # Linear mapping from screen to complex plane
        complex_center_x = x_min + (x_max - x_min) * center_x / WIDTH
        complex_center_y = y_max - (y_max - y_min) * center_y / HEIGHT
        
        # Print debugging information to verify mappings
        if hasattr(globals(), 'debug_coordinates') and globals().get('debug_coordinates'):
            print(f"Screen point: ({center_x}, {center_y}) -> Complex point: ({complex_center_x}, {complex_center_y})")
            # Screen corners mapping
            top_left_complex = (x_min, y_max)
            bottom_right_complex = (x_max, y_min)
            print(f"Screen bounds: (0,0) -> ({WIDTH},{HEIGHT})")
            print(f"Complex bounds: {top_left_complex} -> {bottom_right_complex}")
        
        # Calculate the width in the complex plane corresponding to zoom_length
        complex_width = (x_max - x_min) * zoom_length / WIDTH
        
        # Ensure complex_width is not zero to prevent division by zero
        complex_width = max(complex_width, 1e-10)
        
        # Calculate zoom factor based on the ratio of current width to new width
        current_width = x_max - x_min
        zoom_factor = current_width / complex_width
        
        # Cap the zoom factor to avoid extreme zooms
        zoom_factor = min(zoom_factor, 10.0)
        
        # If zoom was capped, recalculate the complex_width
        if zoom_factor < current_width / complex_width:
            complex_width = current_width / zoom_factor
        
        # Calculate the new boundaries in the complex plane
        new_x_min = complex_center_x - complex_width / 2
        new_x_max = complex_center_x + complex_width / 2
        
        # Maintain aspect ratio for y coordinates
        complex_height = complex_width
        new_y_min = complex_center_y - complex_height / 2
        new_y_max = complex_center_y + complex_height / 2
        
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
        new_x_min, new_x_max, new_y_min, new_y_max = zoom_area["new_bounds"]
        
        # Draw the original rectangle (dimmed)
        original_rect = pygame.Rect(rect_x1, rect_y1, rect_width, rect_height)
        pygame.draw.rect(screen, (100, 100, 100), original_rect, max(1, int(SCALE_FACTOR / 2)))
        
        # Draw the square that will be zoomed into (highlighted)
        # Convert floating point coordinates to integers for drawing
        square_rect = pygame.Rect(
            int(square_x1), 
            int(square_y1), 
            int(square_x2 - square_x1), 
            int(square_y2 - square_y1)
        )
        pygame.draw.rect(screen, (255, 255, 255), square_rect, max(1, int(SCALE_FACTOR)))
        
        # Display zoom coordinates in the square
        zoom_info_font_size = int(10 * SCALE_FACTOR)
        zoom_info_font = pygame.font.SysFont('Arial', zoom_info_font_size)
        
        # Show the center coordinates - use the exact coordinates we'll zoom to
        zoom_text1 = zoom_info_font.render(
            f"Center: ({complex_center_x:.4f}, {complex_center_y:.4f})", 
            True, 
            (255, 255, 255)
        )
        zoom_text_rect1 = zoom_text1.get_rect()
        zoom_text_rect1.centerx = int(center_x)
        zoom_text_rect1.y = int(square_y1) + int(4 * SCALE_FACTOR)
        
        # Show the zoom factor - use the actual calculated factor that will be applied
        zoom_text2 = zoom_info_font.render(
            f"Zoom: {zoom_factor:.1f}x", 
            True, 
            (255, 255, 255)
        )
        zoom_text_rect2 = zoom_text2.get_rect()
        zoom_text_rect2.centerx = int(center_x)
        zoom_text_rect2.y = int(square_y1) + int(19 * SCALE_FACTOR)
        
        # Add a small background behind the text for readability
        text_bg_rect1 = zoom_text_rect1.inflate(int(6 * SCALE_FACTOR), int(4 * SCALE_FACTOR))
        text_bg1 = pygame.Surface((text_bg_rect1.width, text_bg_rect1.height))
        text_bg1.set_alpha(180) 
        text_bg1.fill((0, 0, 0))
        screen.blit(text_bg1, text_bg_rect1)
        
        text_bg_rect2 = zoom_text_rect2.inflate(int(6 * SCALE_FACTOR), int(4 * SCALE_FACTOR))
        text_bg2 = pygame.Surface((text_bg_rect2.width, text_bg_rect2.height))
        text_bg2.set_alpha(180) 
        text_bg2.fill((0, 0, 0))
        screen.blit(text_bg2, text_bg_rect2)
        
        # Draw the text
        screen.blit(zoom_text1, zoom_text_rect1)
        screen.blit(zoom_text2, zoom_text_rect2)
        
        # Draw a crosshair at the center with thickness based on scale
        line_thickness = max(1, int(SCALE_FACTOR / 2))
        crosshair_size = int(5 * SCALE_FACTOR)
        pygame.draw.line(screen, (255, 255, 0), 
                        (int(center_x) - crosshair_size, int(center_y)), 
                        (int(center_x) + crosshair_size, int(center_y)), 
                        line_thickness)
        pygame.draw.line(screen, (255, 255, 0), 
                        (int(center_x), int(center_y) - crosshair_size), 
                        (int(center_x), int(center_y) + crosshair_size), 
                        line_thickness)
        
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
        complex_width = zoom_area["complex_width"]
        new_x_min, new_x_max, new_y_min, new_y_max = zoom_area["new_bounds"]
        
        # Debug information about the zoom
        if debug_coordinates:
            print(f"ZOOM DEBUG INFO:")
            print(f"  Screen selection: {start_pos} to {current_pos}")
            print(f"  Current bounds: x=[{x_min:.6f}, {x_max:.6f}], y=[{y_min:.6f}, {y_max:.6f}]")
            print(f"  New bounds: x=[{new_x_min:.6f}, {new_x_max:.6f}], y=[{new_y_min:.6f}, {new_y_max:.6f}]")
            print(f"  Complex center: ({complex_center_x:.6f}, {complex_center_y:.6f})")
            print(f"  Width in complex plane: {complex_width:.6f}")
            
            # Verify width calculations
            rect_width = abs(new_x_max - new_x_min)
            rect_height = abs(new_y_max - new_y_min)
            aspect_ratio = rect_width / rect_height
            print(f"  Rect dimensions: {rect_width:.6f} x {rect_height:.6f}, aspect ratio: {aspect_ratio:.6f}")
        
        # Save current view to history
        zoom_history.append((x_min, x_max, y_min, y_max, max_iter))
        
        # Limit history size to prevent memory issues
        if len(zoom_history) > 50:
            zoom_history.pop(0)
        
        # Log what we're about to do
        effective_iter = max_iter * high_quality_multiplier if high_quality_mode else max_iter
        print(f"Zooming to center: ({complex_center_x:.6f}, {complex_center_y:.6f})")
        print(f"New bounds: x=[{new_x_min:.6f}, {new_x_max:.6f}], y=[{new_y_min:.6f}, {new_y_max:.6f}]")
        print(f"Zoom factor: {zoom_factor:.2f}x, using {effective_iter} iterations")
        
        # Apply the new bounds directly
        x_min, x_max = new_x_min, new_x_max
        y_min, y_max = new_y_min, new_y_max
        
        # Reset current pixels to force recalculation
        current_pixels = None
        update_mandelbrot()

    def toggle_numpy_mode():
        """Toggle between NumPy and Numba implementations"""
        global force_numpy, current_pixels
        
        # Only toggle if Numba is available and enabled globally
        if USE_NUMBA and HAVE_NUMBA:
            force_numpy = not force_numpy
            current_pixels = None  # Force recalculation
            
            # Update status in console
            if force_numpy:
                print("Using NumPy implementation (Numba disabled for this session)")
            else:
                print("Using Numba implementation for acceleration")
                
            # Update the display
            update_mandelbrot()
        else:
            print("Numba not available or disabled in configuration, using NumPy implementation only")

    # Add a function to toggle Numba output transposition
    def toggle_numba_transpose():
        """Toggle whether Numba output should be transposed"""
        global TRANSPOSE_NUMBA_OUTPUT, current_pixels
        
        # Only toggle if we're using Numba
        if USE_NUMBA and HAVE_NUMBA and not force_numpy:
            TRANSPOSE_NUMBA_OUTPUT = not TRANSPOSE_NUMBA_OUTPUT
            current_pixels = None  # Force recalculation
            
            # Update status in console
            if TRANSPOSE_NUMBA_OUTPUT:
                print("Numba output will be transposed")
            else:
                print("Numba output will not be transposed")
                
            # Update the display
            update_mandelbrot()
        else:
            print("Not using Numba, so transposition toggle has no effect")

    def reset_view():
        """Reset to initial view"""
        global x_min, x_max, y_min, y_max, max_iter, current_pixels
        
        # Reset to initial parameters
        x_min, x_max = -2, 1
        y_min, y_max = -1.5, 1.5
        max_iter = 100
        
        # Force recalculation
        current_pixels = None
        
        # Update the display
        update_mandelbrot()
        
        print("View reset to initial state")

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
                    color_mode = (color_mode + 1) % 9
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
                elif event.key == pygame.K_q:
                    # Toggle quality mode
                    toggle_quality_mode()
                elif event.key == pygame.K_x:
                    # Toggle debug mode
                    globals()['debug_coordinates'] = not debug_coordinates
                    draw_mandelbrot()
                elif event.key == pygame.K_n:
                    # Toggle NumPy mode
                    toggle_numpy_mode()
                elif event.key == pygame.K_t:
                    # Toggle Numba transpose
                    toggle_numba_transpose()
                elif event.key == pygame.K_r:
                    # Reset view
                    reset_view()
        
        # Limit the frame rate
        clock.tick(30)
    
    print("Exiting Mandelbrot Viewer...")
    pygame.quit()

except Exception as e:
    print(f"Error: {e}")
    print(traceback.format_exc())
    pygame.quit()
    sys.exit(1)
