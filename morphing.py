import pygame
import math
import random

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Morphing Shapes")

clock = pygame.time.Clock()
time = 0
running = True

def get_shape_point(t, shape_type, size, offset=0):
    """Get point on shape outline at time t"""
    if shape_type == 0:  # Circle
        angle = t * 2 * math.pi + offset
        return (size * math.cos(angle), size * math.sin(angle))
    elif shape_type == 1:  # Square
        angle = t * 2 * math.pi + offset
        x = size * math.copysign(1, math.cos(angle))
        y = size * math.copysign(1, math.sin(angle))
        return (x, y)
    elif shape_type == 2:  # Triangle
        angle = t * 2 * math.pi + offset
        # Triangle in polar coordinates
        r = size / (1 + 2 * math.cos(angle % (2*math.pi/3) - math.pi/3))
        return (r * math.cos(angle), r * math.sin(angle))
    elif shape_type == 3:  # Star
        angle = t * 2 * math.pi + offset
        r = size * (0.5 + 0.5 * math.cos(5 * angle))
        return (r * math.cos(angle), r * math.sin(angle))

shapes = []
for _ in range(20):
    x = random.randint(100, WIDTH - 100)
    y = random.randint(100, HEIGHT - 100)
    size = random.randint(30, 80)
    speed = random.uniform(0.5, 2)
    shape_type = random.randint(0, 3)
    shapes.append([x, y, size, speed, shape_type, random.uniform(0, 2*math.pi)])

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
    
    screen.fill((5, 5, 15))
    time += 0.01
    
    for i, (x, y, size, speed, shape_type, offset) in enumerate(shapes):
        # Morph between shapes slowly
        target_type = (shape_type + int(time * 0.05)) % 4
        morph_progress = (math.sin(time * 0.1 + i) + 1) / 2
        
        color = (int(128 + 127 * math.sin(time + i)), 
                 int(128 + 127 * math.sin(time * 1.2 + i * 2)), 
                 int(128 + 127 * math.sin(time * 0.8 + i * 3)))
        
        # Draw morphing shape
        points = []
        for j in range(20):
            t = j / 20
            # Interpolate between current and target shape
            p1 = get_shape_point(t, shape_type, size, offset + time * speed)
            p2 = get_shape_point(t, target_type, size, offset + time * speed)
            px = p1[0] * (1 - morph_progress) + p2[0] * morph_progress
            py = p1[1] * (1 - morph_progress) + p2[1] * morph_progress
            points.append((x + px, y + py))
        
        pygame.draw.polygon(screen, color, points, 2)
    
    pygame.display.flip()
    clock.tick(30)

pygame.quit()