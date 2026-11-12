import pygame
import math

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ocean Waves")

clock = pygame.time.Clock()
time = 0
running = True

# Wave settings
WAVE_COUNT = 8
AMPLITUDE = 50
FREQUENCY = 0.02
SPEED = 2

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
    
    screen.fill((10, 10, 30))
    time += 0.1
    
    # Draw waves
    for wave in range(WAVE_COUNT):
        phase_shift = wave * (math.pi / 4)
        # Color gradient from deep blue to cyan
        blue = 50 + wave * 20
        color = (0, 100 + wave * 10, min(255, blue))
        
        points = []
        for x in range(WIDTH):
            y = HEIGHT // 2 + (wave - WAVE_COUNT//2) * 15
            # Combine multiple sine waves for complex patterns
            wave1 = AMPLITUDE * math.sin(x * FREQUENCY + time * SPEED + phase_shift)
            wave2 = 30 * math.sin(x * FREQUENCY * 0.5 + time * 1.5 + phase_shift)
            y += wave1 + wave2
            points.append((x, int(y)))
        
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, 3)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()