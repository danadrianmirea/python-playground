import pygame
import random
import math

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fireworks Screensaver")
clock = pygame.time.Clock()

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.randint(30, 80)
        self.size = random.randint(2, 4)
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.05  # Gravity
        self.vx *= 0.99  # Air resistance
        self.vy *= 0.99
        self.life -= 1
        self.size *= 0.99
    
    def draw(self):
        alpha = max(0, self.life / 80)
        color = tuple(int(c * alpha) for c in self.color)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), int(max(1, self.size)))

particles = []
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
    
    # Create new fireworks
    if random.random() < 0.02:
        x = random.randint(100, WIDTH - 100)
        y = random.randint(100, HEIGHT - 200)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        for _ in range(100):
            particles.append(Particle(x, y, color))
    
    # Update and draw
    screen.fill((0, 0, 0))
    particles = [p for p in particles if p.life > 0 and p.size > 0.5]
    for p in particles:
        p.update()
        p.draw()
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()