import pygame
import math
import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Initialize Pygame
pygame.init()

# Constants
WIDTH = 800
HEIGHT = 600
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
ITERATION_DELAY_TIME=1
START_ITERATION=2
MAX_ITERATIONS=8

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Koch Curve Generator (GPU Accelerated)")

# Set up font for debug info
font = pygame.font.Font(None, 24)

# CUDA kernel for computing Koch curve points
cuda_code = """
extern "C" {

__device__ void koch_points_gpu(float* points, int* point_count, 
                              float x1, float y1, float x2, float y2, 
                              int iteration, int max_points) {
    if (iteration == 0) {
        if (*point_count < max_points - 1) {
            points[*point_count * 2] = x1;
            points[*point_count * 2 + 1] = y1;
            *point_count += 1;
        }
        return;
    }
    
    float dx = (x2 - x1) / 3.0f;
    float dy = (y2 - y1) / 3.0f;
    
    float p1_third_x = x1 + dx;
    float p1_third_y = y1 + dy;
    float p2_third_x = x2 - dx;
    float p2_third_y = y2 - dy;
    
    float perp_x = -dy;
    float perp_y = dx;
    
    float length = sqrtf(dx*dx + dy*dy);
    float height = sqrtf(3.0f) * length / 2.0f;
    
    float mid_x = (x1 + x2) / 2.0f;
    float mid_y = (y1 + y2) / 2.0f;
    float peak_x = mid_x + perp_x * height / length;
    float peak_y = mid_y + perp_y * height / length;
    
    koch_points_gpu(points, point_count, x1, y1, p1_third_x, p1_third_y, iteration - 1, max_points);
    koch_points_gpu(points, point_count, p1_third_x, p1_third_y, peak_x, peak_y, iteration - 1, max_points);
    koch_points_gpu(points, point_count, peak_x, peak_y, p2_third_x, p2_third_y, iteration - 1, max_points);
    koch_points_gpu(points, point_count, p2_third_x, p2_third_y, x2, y2, iteration - 1, max_points);
}

__global__ void compute_koch_curve(float* points, int* point_count,
                                 float p1_x, float p1_y,
                                 float p2_x, float p2_y,
                                 float p3_x, float p3_y,
                                 int iteration, int max_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        *point_count = 0;
        // First side
        koch_points_gpu(points, point_count, p1_x, p1_y, p2_x, p2_y, iteration, max_points);
        // Second side
        koch_points_gpu(points, point_count, p2_x, p2_y, p3_x, p3_y, iteration, max_points);
        // Third side
        koch_points_gpu(points, point_count, p3_x, p3_y, p1_x, p1_y, iteration, max_points);
        // Add the closing point
        if (*point_count < max_points) {
            points[*point_count * 2] = p1_x;
            points[*point_count * 2 + 1] = p1_y;
            *point_count += 1;
        }
    }
}

}
"""

try:
    # Compile CUDA code with minimal options
    options = [
        '--allow-unsupported-compiler',
        '--gpu-architecture=sm_60'
    ]
    mod = SourceModule(cuda_code, options=options)
    compute_koch_curve_kernel = mod.get_function("compute_koch_curve")
except cuda.LogicError as e:
    print(f"CUDA compilation error: {e}")
    print("Falling back to CPU version...")
    USE_GPU = False
except Exception as e:
    print(f"Error: {e}")
    print("Falling back to CPU version...")
    USE_GPU = False
else:
    USE_GPU = True

def compute_koch_curve_cpu(p1, p2, p3, iteration):
    def koch_points(p1, p2, iteration):
        if iteration == 0:
            return [p1, p2]
        
        x1, y1 = p1
        x2, y2 = p2
        
        dx = (x2 - x1) / 3
        dy = (y2 - y1) / 3
        
        p1_third = (x1 + dx, y1 + dy)
        p2_third = (x2 - dx, y2 - dy)
        
        perp_x = -dy
        perp_y = dx
        
        length = math.sqrt(dx*dx + dy*dy)
        height = math.sqrt(3) * length / 2
        
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        peak_x = mid_x + perp_x * height / length
        peak_y = mid_y + perp_y * height / length
        
        points1 = koch_points(p1, p1_third, iteration - 1)
        points2 = koch_points(p1_third, (peak_x, peak_y), iteration - 1)
        points3 = koch_points((peak_x, peak_y), p2_third, iteration - 1)
        points4 = koch_points(p2_third, p2, iteration - 1)
        
        return points1[:-1] + points2[:-1] + points3[:-1] + points4
    
    points1 = koch_points(p1, p2, iteration)
    points2 = koch_points(p2, p3, iteration)
    points3 = koch_points(p3, p1, iteration)
    return np.array(points1[:-1] + points2[:-1] + points3[:-1] + [p1])

def compute_koch_curve_gpu(p1, p2, p3, iteration):
    # Calculate maximum possible points (4^iteration * 3 + 1)
    max_points = int(4 ** iteration * 3 + 1)
    
    # Allocate GPU memory with extra space to ensure we don't overflow
    points_gpu = cuda.mem_alloc(max_points * 2 * np.float32().nbytes)
    point_count_gpu = cuda.mem_alloc(np.int32().nbytes)
    
    # Prepare input parameters
    p1_x, p1_y = p1
    p2_x, p2_y = p2
    p3_x, p3_y = p3
    
    # Launch kernel
    compute_koch_curve_kernel(
        points_gpu, point_count_gpu,
        np.float32(p1_x), np.float32(p1_y),
        np.float32(p2_x), np.float32(p2_y),
        np.float32(p3_x), np.float32(p3_y),
        np.int32(iteration), np.int32(max_points),
        block=(1, 1, 1), grid=(1, 1)
    )
    
    # Get point count
    point_count = np.zeros(1, dtype=np.int32)
    cuda.memcpy_dtoh(point_count, point_count_gpu)
    
    # Get points
    points = np.zeros((point_count[0], 2), dtype=np.float32)
    cuda.memcpy_dtoh(points, points_gpu)
    
    # Ensure we have all points by checking if we need to add the closing point
    if len(points) > 0:
        # Add the closing point if it's not already there
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        # Ensure we have all three sides of the triangle
        if not np.array_equal(points[-1], points[0]):
            points = np.vstack([points, points[0]])
    
    return points

def main(wait_for_input=True):
    clock = pygame.time.Clock()
    current_iteration = START_ITERATION
    max_iterations = MAX_ITERATIONS
    target_time_per_iteration = 1.0
    start_time = time.time()
    delay_start_time = None
    
    # Calculate the initial triangle points
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    size = min(WIDTH, HEIGHT) * 0.4
    
    p1 = (center_x, center_y - size)
    p2 = (center_x - size * math.sqrt(3)/2, center_y + size/2)
    p3 = (center_x + size * math.sqrt(3)/2, center_y + size/2)
    
    current_points = []
    current_point_index = 0
    points_per_frame = 1
    
    running = True
    waiting_for_input = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and waiting_for_input:
                if event.key == pygame.K_SPACE:
                    waiting_for_input = False
                    current_iteration += 1
                    current_points = []
                    current_point_index = 0
        
        screen.fill(BLACK)
        
        # Generate points for the current iteration if needed
        if current_iteration <= max_iterations and not current_points:
            if current_iteration == 1:
                current_points = [p1, p2, p3, p1]
            else:
                # Use GPU or CPU to compute points
                if USE_GPU:
                    points = compute_koch_curve_gpu(p1, p2, p3, current_iteration - 1)
                else:
                    points = compute_koch_curve_cpu(p1, p2, p3, current_iteration - 1)
                current_points = [(float(x), float(y)) for x, y in points]
                current_points.append(p1)  # Close the shape
            
            # Calculate points_per_frame based on total points and target time
            total_points = len(current_points)
            frames_per_iteration = int(target_time_per_iteration * 60)
            points_per_frame = max(1, int(total_points / frames_per_iteration))
        
        # Draw the current points
        if current_points:
            # Draw the lines up to the current point
            for i in range(min(int(current_point_index), len(current_points) - 1)):
                pygame.draw.line(screen, WHITE, current_points[i], current_points[i+1], 1)
            
            # Draw the current point
            if current_point_index < len(current_points) - 1:
                pygame.draw.line(screen, WHITE, current_points[int(current_point_index)], 
                               current_points[int(current_point_index) + 1], 1)
            
            # Update the current point index
            current_point_index += points_per_frame
            
            # If we've reached the end of the current iteration
            if current_point_index >= len(current_points) - 1:
                current_point_index = len(current_points) - 1
                pygame.draw.line(screen, WHITE, current_points[-2], current_points[-1], 1)
                pygame.draw.polygon(screen, GREEN, current_points)
                if wait_for_input:
                    waiting_for_input = True
                else:
                    if delay_start_time is None:
                        delay_start_time = time.time()
                    if time.time() - delay_start_time >= ITERATION_DELAY_TIME:
                        current_iteration += 1
                        current_points = []
                        current_point_index = 0
                        delay_start_time = None
        
            if current_iteration > max_iterations:
                current_iteration = 1    

        # Draw debug information
        elapsed_time = time.time() - start_time
        debug_info = [
            f"Iteration: {current_iteration}/{max_iterations}",
            f"Points: {len(current_points) if current_points else 0}",
            f"Current Point: {current_point_index}/{len(current_points) if current_points else 0}",
            f"Time: {elapsed_time:.1f}s",
            f"Points per Frame: {points_per_frame}",
            f"{'GPU' if USE_GPU else 'CPU'} Accelerated"
        ]
        
        if waiting_for_input:
            debug_info.append("Press SPACE to continue to next iteration")
        elif current_iteration <= max_iterations:
            debug_info.append("Auto-advancing to next iteration")
        
        for i, text in enumerate(debug_info):
            text_surface = font.render(text, True, RED)
            screen.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

WAIT_FOR_INPUT = False

if __name__ == "__main__":
    main(WAIT_FOR_INPUT) 