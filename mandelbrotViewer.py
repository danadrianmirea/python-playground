import pygame
import numpy as np
from colorsys import hsv_to_rgb
import sys
import traceback

# Set this to False to disable Numba even if available
USE_NUMBA = True
TRANSPOSE_NUMBA_OUTPUT = True

# Add GPU acceleration flag
USE_GPU = True
APPLY_SMOOTH_COLORING = True

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

# Try to import PyOpenCL for GPU acceleration
try:
    import pyopencl as cl
    from pyopencl import mem_flags as mf
    HAVE_PYOPENCL = True
    print("PyOpenCL detected for GPU acceleration")
except ImportError:
    HAVE_PYOPENCL = False
    print("PyOpenCL not found, GPU acceleration will be disabled")

# Define OpenCL context and queue for GPU computation
cl_ctx = None
cl_queue = None
cl_program = None
cl_buffer_x = None
cl_buffer_y = None

# OpenCL kernel code for Mandelbrot computation
OPENCL_KERNEL = """
// Color mapping utilities for the kernel
double3 hsv_to_rgb(double h, double s, double v) {
    double c = v * s;
    double x = c * (1.0 - fabs(fmod(h * 6.0, 2.0) - 1.0));
    double m = v - c;  // Add this line to define m
    
    double3 rgb;
    if (h < 1.0/6.0)
        rgb = (double3)(c, x, 0.0);
    else if (h < 2.0/6.0)
        rgb = (double3)(x, c, 0.0);
    else if (h < 3.0/6.0)
        rgb = (double3)(0.0, c, x);
    else if (h < 4.0/6.0)
        rgb = (double3)(0.0, x, c);
    else if (h < 5.0/6.0)
        rgb = (double3)(x, 0.0, c);
    else
        rgb = (double3)(c, 0.0, x);
    
    return rgb + (double3)(m, m, m);
}

// Apply log smoothing to normalized value
double apply_log_smooth(double val) {
    return log(val * 0.5 + 0.5) / log(1.5);
}

// Rainbow palette (mode 0)
double3 rainbow_palette(double norm_iter, double shift) {
    // Apply shift and wrap to [0,1]
    double phase = fmod((norm_iter * 3.0) + shift, 1.0);
    
    // Convert phase to angle in radians (0 to 2π)
    double angle = phase * 2.0 * M_PI;
    
    // Calculate RGB components using sine waves (120° phase shifts)
    double r = sin(angle) * 0.5 + 0.5;
    double g = sin(angle + 2.0 * M_PI / 3.0) * 0.5 + 0.5;
    double b = sin(angle + 4.0 * M_PI / 3.0) * 0.5 + 0.5;
    
    // Scale to enhance colors
    r = min(r * 1.5, 1.0);
    g = min(g * 1.5, 1.0);
    b = min(b * 1.5, 1.0);
    
    return (double3)(r, g, b);
}

// Fire palette (mode 1)
double3 fire_palette(double norm_iter, double shift) {
    // Apply shift and wrap to [0,1]
    double t = fmod(norm_iter + shift, 1.0);
    
    double3 rgb = (double3)(0.0, 0.0, 0.0);
    
    // Red component: 0→1 in first quarter, then stay at 1
    if (t < 0.25)
        rgb.x = t * 4.0;
    else
        rgb.x = 1.0;
        
    // Green component: 0 in first quarter, 0→1 in second quarter, then 1
    if (t >= 0.25 && t < 0.5)
        rgb.y = (t - 0.25) * 4.0;
    else if (t >= 0.5)
        rgb.y = 1.0;
        
    // Blue component: 0 in first half, 0→1 in third quarter, then 1
    if (t >= 0.5 && t < 0.75)
        rgb.z = (t - 0.5) * 4.0;
    else if (t >= 0.75)
        rgb.z = 1.0;
        
    // Apply intensity reduction in the last quarter
    if (t >= 0.75) {
        double intensity = 1.0 - (t - 0.75) * 0.8;
        rgb *= intensity;
    }
    
    return rgb;
}

// Electric blue (mode 2)
double3 electric_blue(double norm_iter, double shift) {
    // Apply shift and wrap to [0,1]
    double t = fmod(norm_iter + shift, 1.0);
    
    // Always high blue with some variation
    double b = 0.7 + 0.3 * sin(t * M_PI * 4.0);
    
    // Green component: Higher in the middle
    double g = 0.4 * (sin(t * M_PI * 2.0) * sin(t * M_PI * 2.0)); // Squared using multiplication instead of pow
    if (t < 0.5)
        g += 0.2 + 0.6 * t;
    
    // Red component: Low but with some highlights
    double r = 0.1 * (sin(t * M_PI * 8.0) * sin(t * M_PI * 8.0)); // Squared using multiplication instead of pow
    
    // Add white spark effect for lower values
    if (t < 0.15) {
        double spark = 1.0 - t / 0.15;
        r += spark;
        g += spark;
    }
    
    return (double3)(min(r, 1.0), min(g, 1.0), min(b, 1.0));
}

// Twilight palette (mode 3)
double3 twilight_palette(double norm_iter, double shift) {
    // Apply shift and wrap to [0,1]
    double t = fmod(norm_iter + shift, 1.0);
    double3 rgb = (double3)(0.0, 0.0, 0.0);
    
    // Purple to blue (0.0-0.3)
    if (t < 0.3) {
        rgb.x = 0.5 + 0.2 * t / 0.3;
        rgb.y = 0.2 * t / 0.3;
        rgb.z = 0.8 - 0.2 * t / 0.3;
    }
    // Blue to teal (0.3-0.5)
    else if (t < 0.5) {
        rgb.x = 0.7 * (t - 0.3) / 0.2;
        rgb.y = 0.2 + 0.4 * (t - 0.3) / 0.2;
        rgb.z = 0.6 + 0.2 * (t - 0.3) / 0.2;
    }
    // Teal to golden (0.5-0.7)
    else if (t < 0.7) {
        rgb.x = 0.7 + 0.3 * (t - 0.5) / 0.2;
        rgb.y = 0.6 + 0.2 * (t - 0.5) / 0.2;
        rgb.z = 0.8 - 0.8 * (t - 0.5) / 0.2;
    }
    // Golden to deep red (0.7-1.0)
    else {
        rgb.x = 1.0;
        rgb.y = 0.8 - 0.8 * (t - 0.7) / 0.3;
        rgb.z = 0.0;
    }
    
    return rgb;
}

// Neon palette (mode 4)
double3 neon_palette(double norm_iter, double shift) {
    // Apply shift and wrap to [0,1]
    double t = fmod(norm_iter + shift, 1.0);
    
    // Create a striated neon effect based on value
    double phase = t * 15.0;  // Multiple cycles for striations
    
    // Use sine waves with different frequencies for color pulsing
    double r = 0.5 * sin(phase * 1.0 + 0.0) + 0.5;
    double g = 0.5 * sin(phase * 1.0 + 2.0) + 0.5;
    double b = 0.5 * sin(phase * 1.0 + 4.0) + 0.5;
    
    // Apply a glow effect - brighten colors based on original value
    if (t < 0.2) {
        double glow_strength = 1.0 - t / 0.2;
        
        // Calculate glow phase for each pixel in the glow mask
        double glow_phase = fmod(t * 3.0, 3.0);
        
        // Apply glows to respective channels based on phase
        if (glow_phase < 1.0) {
            r = r * 0.5 + 0.5 * (1.0 - t / 0.2);
        } else if (glow_phase < 2.0) {
            g = g * 0.5 + 0.5 * (1.0 - t / 0.2);
        } else {
            b = b * 0.5 + 0.5 * (1.0 - t / 0.2);
        }
    }
    
    // Add secondary glow around edges with sharp contrast
    if (t > 0.2 && t < 0.25) {
        double edge_intensity = (t - 0.2) / 0.05;
        r = r * 0.7 + (1.0 - edge_intensity) * 0.3;
        g = g * 0.7 + (1.0 - edge_intensity) * 0.3;
        b = b * 0.7 + (1.0 - edge_intensity) * 0.3;
    }
    
    // Darken the background (high iteration values)
    if (t >= 0.25) {
        double darkness = min(1.0, (t - 0.25) * 2.0);
        r *= (1.0 - darkness * 0.8);
        g *= (1.0 - darkness * 0.8);
        b *= (1.0 - darkness * 0.8);
    }
    
    return (double3)(min(r, 1.0), min(g, 1.0), min(b, 1.0));
}

// Vintage/Sepia (mode 5)
double3 vintage_sepia(double norm_iter, double shift) {
    // Apply shift and wrap to [0,1]
    double t = fmod(norm_iter + shift, 1.0);
    
    // Create base grayscale value with contrast
    double gray = pow(t, 0.8);  // Explicitly cast t to double for pow function
    
    // Apply sepia toning - different multipliers for RGB
    double r = min(gray * 1.2, 1.0);  // More red
    double g = min(gray * 0.9, 1.0);  // Medium green
    double b = min(gray * 0.6, 1.0);  // Less blue
    
    // Add a vignette effect (darker at edges)
    double sinValue = sin(t * M_PI);
    double vignette = 1.0 - 0.2 * (sinValue * sinValue);
    r *= vignette;
    g *= vignette;
    b *= vignette;
    
    // Add some "aging" noise to simulate vintage look
    double aging_effect = 0.05 * sin(t * 37.0) * sin(t * 23.0);
    r += aging_effect;
    g += aging_effect;
    b += aging_effect;
    
    return (double3)(min(r, 1.0), min(g, 1.0), min(b, 1.0));
}

__kernel void mandelbrot(__global int *iterations_out,
                         __global uchar *rgb_out,
                         __global double *x_array,
                         __global double *y_array,
                         const int width,
                         const int height,
                         const int max_iter,
                         const int color_mode,
                         const double color_shift)
{
    // Get the index of the current element
    int gid_x = get_global_id(0); // column index
    int gid_y = get_global_id(1); // row index
    
    // Check if we're within bounds
    if (gid_x < width && gid_y < height) {
        // Get the complex coordinates
        double x0 = x_array[gid_x];
        double y0 = y_array[gid_y];
        
        // Initialize z = 0
        double x = 0.0;
        double y = 0.0;
        
        // Initialize iteration counter
        int iteration = 0;
        double x2 = 0.0;
        double y2 = 0.0;
        
        // Main iteration loop
        while (x2 + y2 < 4.0 && iteration < max_iter) {
            // z -> z^2 + c
            y = 2.0 * x * y + y0;
            x = x2 - y2 + x0;
            x2 = x * x;
            y2 = y * y;
            iteration++;
        }
        
        // Store the iteration count in the output buffer
        iterations_out[gid_y * width + gid_x] = iteration;
        
        // Calculate pixel index in the rgb buffer (3 bytes per pixel)
        int rgb_idx = 3 * (gid_y * width + gid_x);
        
        // Apply coloring based on the iteration value
        if (iteration == max_iter) {
            // Inside the set - black
            rgb_out[rgb_idx] = 0;      // R
            rgb_out[rgb_idx + 1] = 0;  // G
            rgb_out[rgb_idx + 2] = 0;  // B
        } else {
            // Normalize the iteration count
            double norm_iter = (double)iteration / (double)max_iter;
            double3 rgb;
            
            // Apply coloring based on selected mode
            switch (color_mode) {
                case 0: // Rainbow palette
                    rgb = rainbow_palette(norm_iter, color_shift);
                    break;
                case 1: // Fire palette
                    rgb = fire_palette(norm_iter, color_shift);
                    break;
                case 2: // Electric blue
                    rgb = electric_blue(norm_iter, color_shift);
                    break;
                case 3: // Twilight palette
                    rgb = twilight_palette(norm_iter, color_shift);
                    break;
                case 4: // Neon palette
                    rgb = neon_palette(norm_iter, color_shift);
                    break;
                case 5: // Vintage/Sepia
                    rgb = vintage_sepia(norm_iter, color_shift);
                    break;
                default: // Default - simple grayscale
                    rgb = (double3)(norm_iter, norm_iter, norm_iter);
                    break;
            }
            
            // Store RGB values in output buffer
            rgb_out[rgb_idx] = (uchar)(rgb.x * 255.0);      // R
            rgb_out[rgb_idx + 1] = (uchar)(rgb.y * 255.0);  // G
            rgb_out[rgb_idx + 2] = (uchar)(rgb.z * 255.0);  // B
        }
    }
}
"""

def init_gpu():
    global cl_ctx, cl_queue, cl_program, HAVE_GPU, cl_buffer_x, cl_buffer_y
    
    # Skip if already initialized or not using GPU
    if cl_ctx is not None:
        return True
        
    # Skip if PyOpenCL not available
    if not HAVE_PYOPENCL:
        HAVE_GPU = False
        return False
        
    try:
        # Get platforms
        platforms = cl.get_platforms()
        if not platforms:
            print("No OpenCL platforms available")
            HAVE_GPU = False
            return False
            
        # Select first platform (could be made more sophisticated to choose GPU)
        platform = platforms[0]
        
        # Get devices
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if not devices:
            print("No OpenCL GPU devices available")
            HAVE_GPU = False
            return False
            
        # Select first device
        device = devices[0]
        print(f"Using GPU: {device.name}")
        
        # Create context and queue
        cl_ctx = cl.Context([device])
        cl_queue = cl.CommandQueue(cl_ctx)
        
        # Use the OPENCL_KERNEL defined at the top of the file
        kernel_source = OPENCL_KERNEL
        
        # Build the program with better error handling
        try:
            cl_program = cl.Program(cl_ctx, kernel_source).build()
        except cl.RuntimeError as e:
            error_msg = str(e)
            print(f"OpenCL program build error: {error_msg}")
            
            # Check for common errors
            if "ambiguous call to 'pow'" in error_msg:
                print("Error details: Ambiguous call to pow() function detected.")
                print("Fix: Explicit type casting needed in pow() function calls.")
            
            HAVE_GPU = False
            return False
        
        # Initialize buffer references as None
        cl_buffer_x = None
        cl_buffer_y = None
        
        # Set GPU as available
        HAVE_GPU = True
        print("GPU acceleration initialized successfully")
        return True
    except Exception as e:
        print(f"GPU initialization failed: {e}")
        HAVE_GPU = False
        return False

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
    x_min, x_max = np.float64(-2), np.float64(1)
    y_min, y_max = np.float64(-1.5), np.float64(1.5)  # These values already maintain a square aspect ratio
    max_iter = 100
    
    # High-quality rendering flag
    high_quality_mode = True
    high_quality_multiplier = 4  # 4x more iterations in high quality mode
    min_quality_multiplier = 2   # Minimum multiplier value
    
    # Debug modes
    debug_coordinates = False    # Debug coordinate mappings
    force_numpy = False          # Force NumPy implementation (disable Numba)
    
    # Zoom modes
    smooth_zoom_mode = True      # Toggle between smooth zoom and rectangle selection
    smooth_zooming = False       # Flag to indicate active smooth zooming 
    smooth_zoom_factor = np.float64(1.02)    # Per-frame zoom factor (1.02 = 2% closer per frame)
    smooth_zoom_out = False      # Flag to indicate if we're zooming out instead of in
    mouse_pos = (WIDTH // 2, HEIGHT // 2)  # Default to center of screen
    
    # Panning
    panning = False              # Flag to indicate active panning
    pan_direction = None         # Current pan direction (up, down, left, right)
    pan_speed = np.float64(0.02)             # Pan speed as a fraction of current view width/height
    # Track individual key states for diagonal panning
    key_pressed = {
        "up": False,
        "down": False,
        "left": False,
        "right": False
    }
    
    # Store the current Mandelbrot set
    current_pixels = None
    
    # Rectangle drawing variables
    drawing = False
    start_pos = None
    current_pos = None
    
    # UI visibility control
    show_ui_panels = False  # Initially hide the UI panels
    
    # Color palette settings
    color_mode = 1  # 0: Classic rainbow, 1: Fire palette, 2: Electric blue, 3: Twilight, 4: Neon, 5: Vintage
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
    
    def mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter, color_mode, color_shift):
        global cl_ctx, cl_queue, cl_program, cl_buffer_x, cl_buffer_y, HAVE_GPU
        
        # If OpenCL not initialized, try to do so or fall back to CPU
        if cl_ctx is None and not init_gpu():
            print("GPU acceleration unavailable, falling back to CPU")
            return mandelbrot_cpu(height, width, xmin, xmax, ymin, ymax, max_iter), None
        
        try:
            # Create input data for x and y coordinates
            x = np.linspace(xmin, xmax, width, dtype=np.float64)
            y = np.linspace(ymin, ymax, height, dtype=np.float64)
            
            # Check if buffers exist or need to be recreated due to size change
            if cl_buffer_x is None or cl_buffer_x.size != width * 8:  # 8 bytes per double
                if cl_buffer_x is not None:
                    cl_buffer_x.release()
                cl_buffer_x = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
            else:
                cl.enqueue_copy(cl_queue, cl_buffer_x, x)
                
            if cl_buffer_y is None or cl_buffer_y.size != height * 8:  # 8 bytes per double
                if cl_buffer_y is not None:
                    cl_buffer_y.release()
                cl_buffer_y = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
            else:
                cl.enqueue_copy(cl_queue, cl_buffer_y, y)
            
            # Output buffer for iteration counts
            iterations = np.zeros((height, width), dtype=np.int32)
            iterations_buffer = cl.Buffer(cl_ctx, mf.WRITE_ONLY, iterations.nbytes)
            
            # Output buffer for direct RGB values
            rgb_output = np.zeros((height * width * 3), dtype=np.uint8)
            rgb_buffer = cl.Buffer(cl_ctx, mf.WRITE_ONLY, rgb_output.nbytes)
            
            # Execute kernel
            try:
                mandelbrot_kernel = cl_program.mandelbrot
                mandelbrot_kernel.set_args(iterations_buffer, rgb_buffer,
                                      cl_buffer_x, cl_buffer_y,
                                      np.int32(width), np.int32(height),
                                      np.int32(max_iter),
                                      np.int32(color_mode),
                                      np.float64(color_shift))  # Changed from float32 to float64
                
                # Execute the kernel as a 2D grid
                global_size = (width, height)
                local_size = None  # Let OpenCL choose the work-group size
                cl.enqueue_nd_range_kernel(cl_queue, mandelbrot_kernel, global_size, local_size)
                cl_queue.finish()
            except cl.RuntimeError as e:
                error_msg = str(e)
                print(f"OpenCL kernel execution error: {error_msg}")
                if "build" in error_msg.lower():
                    print("There might be an issue with the OpenCL kernel code.")
                    print("Try restarting the application or updating your GPU drivers.")
                    HAVE_GPU = False  # Disable GPU for future calculations
                
                # Clean up
                iterations_buffer.release()
                rgb_buffer.release()
                
                # Fall back to CPU
                print("Falling back to CPU computation")
                return mandelbrot_cpu(height, width, xmin, xmax, ymin, ymax, max_iter), None
            
            # Read back RGB buffer directly from device
            cl.enqueue_copy(cl_queue, rgb_output, rgb_buffer)
            rgb_shaped = rgb_output.reshape((height, width, 3))
            
            # Read back iterations buffer for potential future use
            cl.enqueue_copy(cl_queue, iterations, iterations_buffer)
            
            # If transpose flag is set, transpose the output to match expected orientation
            if TRANSPOSE_NUMBA_OUTPUT:
                iterations = iterations.T
                rgb_shaped = np.transpose(rgb_shaped, (1, 0, 2))
            
            # Clean up temporary buffers
            iterations_buffer.release()
            rgb_buffer.release()
            
            return iterations, rgb_shaped
            
        except Exception as e:
            print(f"GPU computation failed: {e}")
            if isinstance(e, cl.RuntimeError) and "ambiguous call to 'pow'" in str(e):
                print("Error is related to pow() function ambiguity in the OpenCL kernel.")
                print("This is likely due to type casting issues in the GPU kernel code.")
                HAVE_GPU = False  # Disable GPU for future calculations in this session
            
            print("Falling back to CPU computation")
            return mandelbrot_cpu(height, width, xmin, xmax, ymin, ymax, max_iter), None

    def mandelbrot_cpu(h, w, x_min, x_max, y_min, y_max, max_iter):
        """Calculate the Mandelbrot set using CPU (either Numba or NumPy)"""
        # Use float64 for better precision
        # Set up the x and y ranges with the same orientation as GPU:
        # x increases from left to right: x_min at left, x_max at right
        # y now matches GPU implementation and increases from bottom to top
        x = np.linspace(x_min, x_max, w, dtype=np.float64)
        y = np.linspace(y_min, y_max, h, dtype=np.float64)  # Fixed typo: ymax -> y_max
        
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
            # This makes c[0,0] = x[0] + 1j * y[0] = bottom-left corner now
            # And maps screen coordinates directly to the complex plane with the same orientation as GPU
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

    def mandelbrot(w, h, xa, xb, ya, yb, max_iter, color_mode, color_shift):
        """Calculate Mandelbrot set using the selected implementation"""
        global colored_pixels
        
        # If adaptive resolution is enabled, apply scaling
        if render_scale < 1.0:
            # Calculate scaled dimensions (must be at least 1 pixel)
            scaled_w = max(1, int(w * render_scale))
            scaled_h = max(1, int(h * render_scale))
            
            # Calculate the Mandelbrot set at reduced resolution
            if USE_GPU:
                try:
                    iterations, rgb_array = mandelbrot_gpu(xa, xb, ya, yb, scaled_w, scaled_h, max_iter, color_mode, color_shift)
                    # OpenCL kernel already applies coloring, so we can use rgb_array directly
                    if rgb_array is not None:
                        colored_pixels = rgb_array
                        
                        # Upscale the colored pixels to the target resolution
                        if scaled_w != w or scaled_h != h:
                            # Upscale using NumPy's repeat function (faster than interpolation for this purpose)
                            scale_x = w // scaled_w
                            scale_y = h // scaled_h
                            colored_pixels = np.repeat(np.repeat(colored_pixels, scale_y, axis=0), scale_x, axis=1)
                            
                            # If dimensions don't align perfectly, crop or pad
                            colored_pixels = resize_array(colored_pixels, (h, w, 3))
                    else:     
                        # Upscale the colored pixels to the target resolution
                        if scaled_w != w or scaled_h != h:
                            scale_x = w // scaled_w
                            scale_y = h // scaled_h
                            colored_pixels = np.repeat(np.repeat(colored_pixels, scale_y, axis=0), scale_x, axis=1)
                            colored_pixels = resize_array(colored_pixels, (h, w, 3))
                        
                        return iterations
                except Exception as e:
                    print(f"GPU computation failed, falling back to CPU: {e}")
                    # Fall back to CPU if GPU computation fails
                    iterations = mandelbrot_cpu(scaled_h, scaled_w, xa, xb, ya, yb, max_iter)
                    
                    # Upscale the colored pixels to the target resolution
                    if scaled_w != w or scaled_h != h:
                        # Upscale using NumPy's repeat function (faster than interpolation for this purpose)
                        scale_x = w // scaled_w
                        scale_y = h // scaled_h
                        colored_pixels = np.repeat(np.repeat(colored_pixels, scale_y, axis=0), scale_x, axis=1)
                        
                        # If dimensions don't align perfectly, crop or pad
                        colored_pixels = resize_array(colored_pixels, (h, w, 3))
                    
                    return iterations
            else:
                iterations = mandelbrot_cpu(scaled_h, scaled_w, xa, xb, ya, yb, max_iter)

                # Upscale the colored pixels to the target resolution
                if scaled_w != w or scaled_h != h:
                    # Upscale using NumPy's repeat function (faster than interpolation for this purpose)
                    scale_x = w // scaled_w
                    scale_y = h // scaled_h
                    colored_pixels = np.repeat(np.repeat(colored_pixels, scale_y, axis=0), scale_x, axis=1)
                    
                    # If dimensions don't align perfectly, crop or pad
                    colored_pixels = resize_array(colored_pixels, (h, w, 3))
                
                return iterations
        
        # Full resolution calculation
        if USE_GPU:
            try:
                iterations, rgb_array = mandelbrot_gpu(xa, xb, ya, yb, w, h, max_iter, color_mode, color_shift)
                # Store the colored pixels for immediate use if not None
                if rgb_array is not None:
                    colored_pixels = rgb_array
                return iterations
            except Exception as e:
                print(f"GPU computation failed, falling back to CPU: {e}")
                # Fall back to CPU if GPU computation fails
                return mandelbrot_cpu(h, w, xa, xb, ya, yb, max_iter)
        else:
            # CPU calculation
            iterations = mandelbrot_cpu(h, w, xa, xb, ya, yb, max_iter)
            return iterations

    def create_smooth_colormap():
        """Create a lookup table for smooth color mapping"""
        cmap = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            # Normalized value in [0, 1]
            t = np.float64(i) / np.float64(255.0)
            
            # Calculate HSV
            h = t  # hue = normalized value
            s = np.float64(0.8)  # saturation
            v = np.float64(1.0) if t < np.float64(0.95) else (np.float64(1.0) - t) * np.float64(20)  # falloff for high values
            
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
        
        if mode == 0:  # Classic rainbow palette using sine waves
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
        
        elif mode == 1:  # Fire palette
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
        
        elif mode == 2:  # Electric blue
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
        
        elif mode == 3:  # Twilight palette (purple to orange)
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
        
        elif mode == 4:  # Neon palette with bright glows and dark backgrounds
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

        elif mode == 5:  # Vintage/Sepia
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

    def update_render_scale():
        """Update the render scale based on movement state"""
        global render_scale, target_render_scale, is_moving, last_movement_time
        
        # If using GPU, always use full resolution
        if USE_GPU:
            if render_scale < 1.0:
                render_scale = 1.0
                return True
            return False
        
        # If adaptive render scaling is disabled, always use full resolution
        if not adaptive_render_scale:
            if render_scale < 1.0:
                render_scale = 1.0
                return True
            return False
        
        current_time = pygame.time.get_ticks()
        
        # If we're moving, set scale to minimum
        if is_moving:
            target_render_scale = min_render_scale
            last_movement_time = current_time
            
            # When moving, immediately drop to low quality
            if render_scale != min_render_scale:
                render_scale = min_render_scale
                return True
        # If movement has stopped, immediately jump to full resolution
        else:
            if render_scale < 1.0:
                render_scale = 1.0
                return True
        
        return False

    def draw_ui_panel():
        """Draw UI panels with matching styling"""
        # If panels are hidden, return immediately
        if not show_ui_panels:
            return
            
        # Help panel (left)
        help_texts = [
            "Controls:",
            "M: Toggle zoom mode (smooth/selection)",
            f"Zoom Mode: {'Smooth' if smooth_zoom_mode else 'Rectangle Selection'}",
            "In Smooth Zoom Mode:",
            "  Left click (hold): Zoom in at cursor", 
            "  Right click (hold): Zoom out at cursor",
            "In Rectangle Mode:",
            "  Left click and drag: Select zoom area",
            "  Right click: Zoom out to previous view",
            "Panning:",
            "  W: Pan up",
            "  S: Pan down",
            "  A: Pan left",
            "  D: Pan right",
            "E (hold): Smooth zoom in toward cursor",
            "Q (hold): Smooth zoom out from cursor",
            "C: Change color mode",
            "Z/X: Shift colors left/right",
            "I/O: Increase/Decrease iterations", 
            "J/K: Decrease/Increase quality multiplier",
            "T: Toggle adaptive render scaling",
            "Y: Toggle high quality mode",
            "V: Toggle debug mode",
            "N: Toggle Numba/NumPy",
            "G: Toggle GPU acceleration",
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
        color_names = ['Rainbow', 'Fire', 'Electric Blue', 'Twilight', 'Neon', 'Vintage']
        
        # Add high quality indicator to iteration count
        quality_text = f"HQ {high_quality_multiplier}x" if high_quality_mode else "Standard"
        hq_iterations = max_iter * high_quality_multiplier if high_quality_mode else max_iter
        
        # Debug mode indicator
        debug_text = "Debug ON" if debug_coordinates else "Debug OFF"
        
        # Implementation indicator
        if HAVE_GPU and USE_GPU:
            impl_text = "GPU (OpenCL)"
        elif HAVE_NUMBA:
            if USE_NUMBA:
                impl_text = "NumPy" if force_numpy else "Numba"
            else:
                impl_text = "NumPy (Numba disabled in config)"
        else:
            impl_text = "NumPy (Numba not available)"
        
        # Render scale info
        render_scale_percent = int(render_scale * 100)
        adaptive_scale_text = "Enabled" if adaptive_render_scale else "Disabled"
        
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
        
        # Add render scale info only in CPU mode
        if not USE_GPU:
            settings_texts.insert(6, f"Render Scale: {render_scale_percent}% ({adaptive_scale_text})")
        
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
        
        # Check if we need to update the render scale
        scale_changed = update_render_scale()
        
        if current_pixels is None or scale_changed:
            # Calculate the effective iterations based on quality mode
            effective_iter = max_iter * high_quality_multiplier if high_quality_mode else max_iter
            current_pixels = mandelbrot(HEIGHT, WIDTH, x_min, x_max, y_min, y_max, effective_iter, color_mode, color_shift)
        
        # If colored_pixels is not already calculated by GPU, do it on CPU now
        if USE_GPU and HAVE_GPU and colored_pixels is not None:
            # GPU has already calculated the colors
            pass
        else:
            # CPU coloring
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
        global current_pixels, colored_pixels, base_surface, is_moving
        
        # Mark that we're actively moving (panning or zooming)
        is_moving = True
        
        # Force recalculation with new view parameters
        current_pixels = None
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
            print(f"High quality mode enabled: {effective_iter} iterations (multiplier: {high_quality_multiplier}x)")
        else:
            print(f"Standard quality mode: {max_iter} iterations")
        
        # Update the display
        update_mandelbrot()

    def adjust_quality_multiplier(increase=True):
        """Adjust the high quality multiplier up or down"""
        global high_quality_multiplier, current_pixels
        
        # Store the old value for reporting
        old_multiplier = high_quality_multiplier
        
        if increase:
            # Double the multiplier (maximum value is 320)
            high_quality_multiplier = min(high_quality_multiplier * 2, 320)
        else:
            # Halve the multiplier, but don't go below the minimum
            high_quality_multiplier = max(high_quality_multiplier // 2, min_quality_multiplier)
        
        # Only update if the value actually changed
        if old_multiplier != high_quality_multiplier:
            # Force recalculation if we're in high quality mode
            if high_quality_mode:
                current_pixels = None
                
            # Log the change
            effective_iter = max_iter * high_quality_multiplier if high_quality_mode else max_iter
            print(f"Quality multiplier {'increased' if increase else 'decreased'} to {high_quality_multiplier}x")
            print(f"Effective iterations: {effective_iter}")
            
            # Update the display if in high quality mode
            if high_quality_mode:
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
        rect_x1 = np.float64(min(start_pos[0], current_pos[0]))
        rect_x2 = np.float64(max(start_pos[0], current_pos[0]))
        rect_y1 = np.float64(min(start_pos[1], current_pos[1]))
        rect_y2 = np.float64(max(start_pos[1], current_pos[1]))
        
        # Find the center of the rectangle (as floating point for precision)
        center_x = (rect_x1 + rect_x2) / np.float64(2.0)
        center_y = (rect_y1 + rect_y2) / np.float64(2.0)
        
        # Get the longest dimension (width or height)
        rect_width = rect_x2 - rect_x1
        rect_height = rect_y2 - rect_y1
        zoom_length = max(rect_width, rect_height)
        
        # Ensure zoom_length is not zero to prevent division by zero later
        # Set a minimum size of 5 pixels
        zoom_length = max(np.float64(5.0), zoom_length)
        
        # Calculate the square bounds centered at the original center point
        # Store as floats to maintain precision
        square_x1 = center_x - zoom_length / np.float64(2.0)
        square_y1 = center_y - zoom_length / np.float64(2.0)
        square_x2 = center_x + zoom_length / np.float64(2.0)
        square_y2 = center_y + zoom_length / np.float64(2.0)
        
        # Ensure square is within screen bounds
        square_x1 = max(np.float64(0.0), square_x1)
        square_y1 = max(np.float64(0.0), square_y1)
        square_x2 = min(np.float64(WIDTH), square_x2)
        square_y2 = min(np.float64(HEIGHT), square_y2)
        
        # Recalculate center if square was adjusted
        center_x = (square_x1 + square_x2) / np.float64(2.0)
        center_y = (square_y1 + square_y2) / np.float64(2.0)
        
        # Map the center point from screen coordinates to complex plane
        # With our updated mandelbrot function, we directly map:
        # - x from 0 to WIDTH maps to x_min to x_max
        # - y from 0 to HEIGHT maps to y_max to y_min (inverted)
        
        # Linear mapping from screen to complex plane with higher precision
        complex_center_x = np.float64(x_min) + (np.float64(x_max) - np.float64(x_min)) * center_x / np.float64(WIDTH)
        complex_center_y = np.float64(y_min) + (np.float64(y_max) - np.float64(y_min)) * center_y / np.float64(HEIGHT)
        
        # Print debugging information to verify mappings
        if hasattr(globals(), 'debug_coordinates') and globals().get('debug_coordinates'):
            print(f"Screen point: ({center_x}, {center_y}) -> Complex point: ({complex_center_x}, {complex_center_y})")
            # Screen corners mapping
            top_left_complex = (x_min, y_max)
            bottom_right_complex = (x_max, y_min)
            print(f"Screen bounds: (0,0) -> ({WIDTH},{HEIGHT})")
            print(f"Complex bounds: {top_left_complex} -> {bottom_right_complex}")
        
        # Calculate the width in the complex plane corresponding to zoom_length
        complex_width = (np.float64(x_max) - np.float64(x_min)) * zoom_length / np.float64(WIDTH)
        
        # Ensure complex_width is not zero to prevent division by zero
        complex_width = max(complex_width, np.float64(1e-15))
        
        # Calculate zoom factor based on the ratio of current width to new width
        current_width = np.float64(x_max) - np.float64(x_min)
        zoom_factor = current_width / complex_width
        
        # Calculate the new boundaries in the complex plane
        new_x_min = complex_center_x - complex_width / np.float64(2.0)
        new_x_max = complex_center_x + complex_width / np.float64(2.0)
        
        # Maintain aspect ratio for y coordinates
        complex_height = complex_width
        new_y_min = complex_center_y - complex_height / np.float64(2.0)
        new_y_max = complex_center_y + complex_height / np.float64(2.0)
        
        # Check for floating-point precision issues
        if abs(new_x_max - new_x_min) < np.float64(1e-15) or abs(new_y_max - new_y_min) < np.float64(1e-15):
            print("Warning: Reached floating-point precision limit")
            return None
        
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

    def reset_view():
        """Reset to initial view"""
        global x_min, x_max, y_min, y_max, max_iter, current_pixels
        
        # Reset to initial parameters
        x_min, x_max = np.float64(-2), np.float64(1)
        y_min, y_max = np.float64(-1.5), np.float64(1.5)
        max_iter = 100
        
        # Force recalculation
        current_pixels = None
        
        # Update the display
        update_mandelbrot()
        
        print("View reset to initial state")

    def smooth_zoom_to_cursor(zoom_out=False):
        """Perform a smooth zoom step centered on the current mouse position"""
        global x_min, x_max, y_min, y_max, current_pixels
        
        # Get the current width and height in the complex plane
        width = x_max - x_min
        height = y_max - y_min
        
        # Map mouse position to complex coordinates with higher precision
        # Use numpy's float64 for better precision
        cx = np.float64(x_min) + np.float64(width) * np.float64(mouse_pos[0]) / np.float64(WIDTH)
        cy = np.float64(y_min) + np.float64(height) * np.float64(mouse_pos[1]) / np.float64(HEIGHT)
        
        # Calculate new bounds with a small zoom step
        # The zoom factor is applied relative to the cursor position
        if zoom_out:
            # When zooming out, multiply by the factor instead of dividing
            new_width = np.float64(width) * np.float64(smooth_zoom_factor)
            new_height = np.float64(height) * np.float64(smooth_zoom_factor)
        else:
            # Normal zoom in
            new_width = np.float64(width) / np.float64(smooth_zoom_factor)
            new_height = np.float64(height) / np.float64(smooth_zoom_factor)
        
        # Calculate new bounds while keeping the cursor point at the same relative position
        # We need to maintain the ratio between cursor position and the edge distances
        cursor_ratio_x = (cx - np.float64(x_min)) / np.float64(width)
        cursor_ratio_y = (cy - np.float64(y_min)) / np.float64(height)
        
        # Apply ratios to calculate new bounds with higher precision
        new_x_min = cx - new_width * cursor_ratio_x
        new_x_max = new_x_min + new_width
        new_y_min = cy - new_height * cursor_ratio_y
        new_y_max = new_y_min + new_height
        
        # Check for floating-point precision issues
        if abs(new_x_max - new_x_min) < np.float64(1e-15) or abs(new_y_max - new_y_min) < np.float64(1e-15):
            print("Warning: Reached floating-point precision limit")
            return
        
        # Apply the new bounds
        x_min, x_max = float(new_x_min), float(new_x_max)
        y_min, y_max = float(new_y_min), float(new_y_max)
        
        # Reset current pixels to force recalculation
        current_pixels = None
        
        # In debug mode, log the new coordinates
        if debug_coordinates:
            zoom_type = "out" if zoom_out else "in"
            print(f"Smooth zoom {zoom_type} to: ({cx:.6f}, {cy:.6f})")
            print(f"New bounds: x=[{x_min:.6f}, {x_max:.6f}], y=[{y_min:.6f}, {y_max:.6f}]")

    def pan_view(direction=None):
        """Pan the view in the specified direction or based on key states"""
        global x_min, x_max, y_min, y_max, current_pixels
        
        # Get current view dimensions
        width = x_max - x_min
        height = y_max - y_min
        
        # Calculate pan amount based on current view size
        pan_amount_x = width * pan_speed
        pan_amount_y = height * pan_speed
        
        # Check key states instead of single direction to support diagonal movement
        # Vertical movement
        if key_pressed["up"] and not key_pressed["down"]:
            y_min += pan_amount_y
            y_max += pan_amount_y
        elif key_pressed["down"] and not key_pressed["up"]:
            y_min -= pan_amount_y
            y_max -= pan_amount_y
        
        # Horizontal movement
        if key_pressed["left"] and not key_pressed["right"]:
            x_min -= pan_amount_x
            x_max -= pan_amount_x
        elif key_pressed["right"] and not key_pressed["left"]:
            x_min += pan_amount_x
            x_max += pan_amount_x
        
        # Force recalculation of the Mandelbrot set
        current_pixels = None
        
        # Print debug info if enabled
        if debug_coordinates:
            current_direction = []
            if key_pressed["up"]: current_direction.append("up")
            if key_pressed["down"]: current_direction.append("down")
            if key_pressed["left"]: current_direction.append("left")
            if key_pressed["right"]: current_direction.append("right")
            
            direction_str = "+".join(current_direction) if current_direction else "none"
            print(f"Panning {direction_str}: New bounds x=[{x_min:.6f}, {x_max:.6f}], y=[{y_min:.6f}, {y_max:.6f}]")

    def toggle_gpu_mode():
        """Toggle between GPU and CPU implementations"""
        global USE_GPU, current_pixels
        
        # Only toggle if GPU is available
        if HAVE_GPU:
            USE_GPU = not USE_GPU
            current_pixels = None  # Force recalculation
            
            # Update status in console
            if USE_GPU:
                print("Using GPU acceleration for Mandelbrot calculations")
            else:
                print("Using CPU implementation for Mandelbrot calculations")
                
            # Update the display
            update_mandelbrot()
        else:
            print("GPU acceleration not available (install PyOpenCL for GPU support)")

    # Add these global variables with the other global settings around line 130
    # Incremental rendering settings
    render_scale = np.float64(1.0)           # Current rendering scale (1.0 = full resolution)
    target_render_scale = np.float64(1.0)    # Target scale to gradually approach
    adaptive_render_scale = False # Toggle for adaptive render scaling during movement
    min_render_scale = np.float64(0.25)      # Minimum rendering scale during movement (0.25 = quarter resolution)
    is_moving = False            # Flag to track if we're currently panning or zooming
    last_movement_time = 0       # Time of last movement action
    scale_recovery_delay = 0     # No delay before increasing resolution

    # Print a message indicating successful initialization
    print("Mandelbrot Viewer successfully initialized...")
    
    # Main loop
    running = True
    clock = pygame.time.Clock()  # Add a clock to control the frame rate
    
    # Ensure we start with a rendered Mandelbrot
    current_pixels = None
    draw_mandelbrot()
    
    # Save history at regular intervals during smooth zoom
    smooth_zoom_history_counter = 0
    smooth_zoom_history_interval = 30  # Save every 30 frames of smooth zooming
    
    while running:
        # Check if we're in smooth zoom mode and E or Q key is still pressed
        # Get the mouse position before handling events
        if smooth_zooming:
            mouse_pos = pygame.mouse.get_pos()
            smooth_zoom_to_cursor(smooth_zoom_out)
            
            # Update the display
            update_mandelbrot()
            
            # Save zoom history periodically during smooth zoom
            smooth_zoom_history_counter += 1
            if smooth_zoom_history_counter >= smooth_zoom_history_interval:
                smooth_zoom_history_counter = 0
                # Save current view to history
                zoom_history.append((x_min, x_max, y_min, y_max, max_iter))
                # Limit history size
                if len(zoom_history) > 50:
                    zoom_history.pop(0)
        
        # Handle active panning
        if panning:
            pan_view()
            update_mandelbrot()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos  # Always update the mouse position
                
                if smooth_zoom_mode:
                    # In smooth zoom mode, mouse buttons directly control zooming
                    if event.button == 1:  # Left click - zoom in
                        # Save current view to history before starting smooth zoom
                        zoom_history.append((x_min, x_max, y_min, y_max, max_iter))
                        # Limit history size
                        if len(zoom_history) > 50:
                            zoom_history.pop(0)
                        smooth_zooming = True
                        smooth_zoom_out = False
                        smooth_zoom_history_counter = 0
                    elif event.button == 3:  # Right click - zoom out
                        # Save current view to history before starting smooth zoom
                        zoom_history.append((x_min, x_max, y_min, y_max, max_iter))
                        # Limit history size
                        if len(zoom_history) > 50:
                            zoom_history.pop(0)
                        smooth_zooming = True
                        smooth_zoom_out = True
                        smooth_zoom_history_counter = 0
                else:
                    # In rectangle selection mode, use original behavior
                    if event.button == 1:  # Left click - start drawing selection rectangle
                        drawing = True
                        start_pos = event.pos
                        current_pos = event.pos
                    elif event.button == 3:  # Right click - zoom out to previous view
                        zoom_out()
            elif event.type == pygame.MOUSEBUTTONUP:
                if smooth_zoom_mode:
                    # In smooth zoom mode, releasing either mouse button stops zooming
                    if event.button == 1 or event.button == 3:
                        smooth_zooming = False
                else:
                    # In rectangle selection mode, use original behavior
                    if event.button == 1 and drawing:  # Left click release - complete rectangle
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
                # Always update mouse position for smooth zoom
                mouse_pos = event.pos
                
                # Only draw selection rectangle in rectangle selection mode
                if not smooth_zoom_mode and drawing:
                    current_pos = event.pos
                    # Draw the selection rectangle without recalculating the Mandelbrot set
                    draw_selection_rectangle()
            elif event.type == pygame.KEYDOWN:
                # Cancel any active selection when pressing any key
                if drawing:
                    drawing = False
                    start_pos = None
                    current_pos = None
                
                if event.key == pygame.K_m:
                    # Toggle zoom mode (smooth vs rectangle selection)
                    smooth_zoom_mode = not smooth_zoom_mode
                    print(f"Zoom mode: {'Smooth zoom' if smooth_zoom_mode else 'Rectangle selection'}")
                    draw_mandelbrot()  # Redraw to update UI panel
                elif event.key == pygame.K_e:
                    # Start smooth zooming in when E key is pressed
                    smooth_zooming = True
                    smooth_zoom_out = False  # Ensure we're zooming in
                    mouse_pos = pygame.mouse.get_pos()
                    smooth_zoom_history_counter = 0
                    
                    # Save current view to history before starting smooth zoom
                    zoom_history.append((x_min, x_max, y_min, y_max, max_iter))
                    # Limit history size
                    if len(zoom_history) > 50:
                        zoom_history.pop(0)
                elif event.key == pygame.K_q:
                    # Start smooth zooming out when Q key is pressed
                    smooth_zooming = True
                    smooth_zoom_out = True  # We're zooming out
                    mouse_pos = pygame.mouse.get_pos()
                    smooth_zoom_history_counter = 0
                    
                    # Save current view to history before starting smooth zoom
                    zoom_history.append((x_min, x_max, y_min, y_max, max_iter))
                    # Limit history size
                    if len(zoom_history) > 50:
                        zoom_history.pop(0)
                elif event.key == pygame.K_c:
                    # Change color mode
                    color_mode = (color_mode + 1) % 6
                    # Force recalculation when using GPU to ensure the color mode is applied
                    if USE_GPU:
                        current_pixels = None
                    draw_mandelbrot()
                elif event.key == pygame.K_z:
                    # Shift colors left
                    color_shift = (color_shift - 0.1) % 1.0
                    # Force recalculation when using GPU to ensure the color shift is applied
                    if USE_GPU:
                        current_pixels = None
                    draw_mandelbrot()
                elif event.key == pygame.K_x:
                    # Shift colors right
                    color_shift = (color_shift + 0.1) % 1.0
                    # Force recalculation when using GPU to ensure the color shift is applied
                    if USE_GPU:
                        current_pixels = None
                    draw_mandelbrot()
                elif event.key == pygame.K_i:
                    # Increase max iterations
                    max_iter = min(max_iter * 2, 2000)
                    update_mandelbrot()
                elif event.key == pygame.K_o:  # Changed from D to O for decrease iterations
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
                elif event.key == pygame.K_y:  # Changed from Q to Y for quality mode
                    # Toggle quality mode
                    toggle_quality_mode()
                elif event.key == pygame.K_v:
                    # Toggle debug mode
                    globals()['debug_coordinates'] = not debug_coordinates
                    draw_mandelbrot()
                elif event.key == pygame.K_n:
                    # Toggle NumPy mode
                    toggle_numpy_mode()
                elif event.key == pygame.K_r:
                    # Reset view
                    reset_view()
                # Panning controls
                elif event.key == pygame.K_w:
                    key_pressed["down"] = True
                    panning = True
                elif event.key == pygame.K_s:
                    key_pressed["up"] = True
                    panning = True
                elif event.key == pygame.K_a:
                    key_pressed["left"] = True
                    panning = True
                elif event.key == pygame.K_d:
                    key_pressed["right"] = True
                    panning = True
                elif event.key == pygame.K_g:
                    # Toggle GPU mode
                    toggle_gpu_mode()
                elif event.key == pygame.K_t:
                    # Toggle adaptive render scaling (only in CPU mode)
                    if not USE_GPU:
                        globals()['adaptive_render_scale'] = not adaptive_render_scale
                        if adaptive_render_scale:
                            print("Adaptive render scaling enabled - reduces resolution during movement")
                        else:
                            print("Adaptive render scaling disabled - always renders at full resolution")
                            render_scale = 1.0  # Reset to full resolution
                        draw_mandelbrot()
                    else:
                        print("Adaptive render scaling is only available in CPU mode")
                elif event.key == pygame.K_j:
                    # Decrease quality multiplier
                    adjust_quality_multiplier(increase=False)
                elif event.key == pygame.K_k:
                    # Increase quality multiplier
                    adjust_quality_multiplier(increase=True)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_e or event.key == pygame.K_q:
                    # Stop smooth zooming when E or Q key is released
                    smooth_zooming = False
                    # Immediately render at full resolution
                    render_scale = 1.0
                    current_pixels = None
                    draw_mandelbrot()
                
                # Stop panning when W, S, A, or D key is released
                elif event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]:
                    if event.key == pygame.K_w:
                        key_pressed["down"] = False
                    elif event.key == pygame.K_s:
                        key_pressed["up"] = False
                    elif event.key == pygame.K_a:
                        key_pressed["left"] = False
                    elif event.key == pygame.K_d:
                        key_pressed["right"] = False
                    
                    # Stop panning only if all direction keys are released
                    if not any(key_pressed.values()):
                        panning = False
                        # Immediately render at full resolution
                        render_scale = 1.0
                        current_pixels = None
                        draw_mandelbrot()
        
        # Limit the frame rate
        clock.tick(30)
        
        # Check if we need to update due to resolution scaling
        if not is_moving and render_scale < 1.0:
            # Initiate a redraw if we need to increase resolution
            if update_render_scale():
                draw_mandelbrot()
        
        # Reset the movement flag at the end of each frame
        is_moving = smooth_zooming or panning
    
    print("Exiting Mandelbrot Viewer...")
    pygame.quit()

except Exception as e:
    print(f"Error: {e}")
    print(traceback.format_exc())
    pygame.quit()
    sys.exit(1)

def resize_array(arr, target_shape):
    """Resize an array to the target shape, either by cropping or padding"""
    result = np.zeros(target_shape, dtype=arr.dtype)
    
    # Get the minimum sizes in each dimension
    min_h = min(arr.shape[0], target_shape[0])
    min_w = min(arr.shape[1], target_shape[1])
    
    # Copy the data that fits
    if len(target_shape) == 3:  # For RGB arrays
        result[:min_h, :min_w, :] = arr[:min_h, :min_w, :]
    else:  # For 2D arrays
        result[:min_h, :min_w] = arr[:min_h, :min_w]
        
    return result