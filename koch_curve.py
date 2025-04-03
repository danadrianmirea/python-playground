import pygame
import math
import time

# Initialize Pygame
pygame.init()

# Constants
WIDTH = 800
HEIGHT = 600
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Koch Curve Generator")

# Set up font for debug info
font = pygame.font.Font(None, 24)

def koch_points(p1, p2, iteration):
    if iteration == 0:
        return [p1, p2]
    
    # Calculate the points for the Koch curve
    x1, y1 = p1
    x2, y2 = p2
    
    # Calculate the points that divide the line into thirds
    dx = (x2 - x1) / 3
    dy = (y2 - y1) / 3
    
    # Points that divide the line into thirds
    p1_third = (x1 + dx, y1 + dy)
    p2_third = (x2 - dx, y2 - dy)
    
    # Calculate the perpendicular vector (rotate 90 degrees counterclockwise)
    perp_x = -dy
    perp_y = dx
    
    # Calculate the height of the equilateral triangle
    length = math.sqrt(dx*dx + dy*dy)
    height = math.sqrt(3) * length / 2
    
    # Calculate the peak point
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    peak_x = mid_x + perp_x * height / length
    peak_y = mid_y + perp_y * height / length
    
    # Recursively generate points for each segment
    points1 = koch_points(p1, p1_third, iteration - 1)
    points2 = koch_points(p1_third, (peak_x, peak_y), iteration - 1)
    points3 = koch_points((peak_x, peak_y), p2_third, iteration - 1)
    points4 = koch_points(p2_third, p2, iteration - 1)
    
    return points1[:-1] + points2[:-1] + points3[:-1] + points4

def main(wait_for_input=True):
    clock = pygame.time.Clock()
    current_iteration = 1  # Start from iteration 1
    max_iterations = 10
    target_time_per_iteration = 3.0  # Target time in seconds for each iteration
    start_time = time.time()
    delay_start_time = None  # Track when delay started
    
    # Calculate the initial triangle points
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    size = min(WIDTH, HEIGHT) * 0.4
    
    # Calculate the three points of the initial triangle
    p1 = (center_x, center_y - size)
    p2 = (center_x - size * math.sqrt(3)/2, center_y + size/2)
    p3 = (center_x + size * math.sqrt(3)/2, center_y + size/2)
    
    # Initialize the points list for the current iteration
    current_points = []
    current_point_index = 0
    points_per_frame = 1  # Initialize with a default value
    
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
                current_points = [p1, p2, p3, p1]  # Close the triangle
            else:
                # Generate points for the current iteration
                points1 = koch_points(p1, p2, current_iteration - 1)
                points2 = koch_points(p2, p3, current_iteration - 1)
                points3 = koch_points(p3, p1, current_iteration - 1)
                current_points = points1[:-1] + points2[:-1] + points3[:-1] + [p1]
            
            # Calculate points_per_frame based on total points and target time
            total_points = len(current_points)
            frames_per_iteration = int(target_time_per_iteration * 60)  # 60 FPS
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
                current_point_index = len(current_points) - 1  # Ensure we don't exceed bounds
                # Draw the final line in white
                pygame.draw.line(screen, WHITE, current_points[-2], current_points[-1], 1)
                # Fill the interior with green
                pygame.draw.polygon(screen, GREEN, current_points)
                if wait_for_input:
                    waiting_for_input = True
                else:
                    # Start the delay if not already started
                    if delay_start_time is None:
                        delay_start_time = time.time()
                    # Check if 3 seconds have passed
                    if time.time() - delay_start_time >= 3.0:
                        current_iteration += 1
                        current_points = []
                        current_point_index = 0
                        delay_start_time = None  # Reset delay timer

            if current_iteration > max_iterations:
                current_iteration = 1        

        # Draw debug information
        elapsed_time = time.time() - start_time
        debug_info = [
            f"Iteration: {current_iteration}/{max_iterations}",
            f"Points: {len(current_points) if current_points else 0}",
            f"Current Point: {current_point_index}/{len(current_points) if current_points else 0}",
            f"Time: {elapsed_time:.1f}s",
            f"Points per Frame: {points_per_frame}"
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

WAIT_FOR_INPUT=False

if __name__ == "__main__":
    main(WAIT_FOR_INPUT)  # You can change this to False to disable waiting 