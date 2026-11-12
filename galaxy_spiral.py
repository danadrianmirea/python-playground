import pygame
import random
import math

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Spiral Galaxy")

clock = pygame.time.Clock()

class Star:
    def __init__(self):
        self.angle = random.uniform(0, 2 * math.pi)
        self.distance = random.uniform(0, 300)
        self.size = random.uniform(1, 3)
        self.brightness = random.randint(50, 255)
        self.drift = random.uniform(0.001, 0.005)
        
    def update(self):
        self.angle += self.drift * (500 / (self.distance + 10))
        self.distance += random.uniform(-0.2, 0.2)
        if self.distance < 0:
            self.distance = 0
        if self.distance > 300:
            self.distance = 300
    
    def draw(self):
        x = WIDTH // 2 + self.distance * math.cos(self.angle)
        y = HEIGHT // 2 + self.distance * math.sin(self.angle)
        color = (self.brightness, self.brightness, min(255, int(self.brightness * 0.8)))
        pygame.draw.circle(screen, color, (int(x), int(y)), int(self.size))

stars = [Star() for _ in range(300)]
rotation = 0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
    
    screen.fill((0, 0, 10))
    
    # Rotate entire galaxy
    rotation += 0.002
    
    for star in stars:
        star.update()
        # Add rotation offset
        temp_angle = star.angle + rotation
        x = WIDTH // 2 + star.distance * math.cos(temp_angle)
        y = HEIGHT // 2 + star.distance * math.sin(temp_angle)
        color = (star.brightness, star.brightness, int(star.brightness * 0.8))
        pygame.draw.circle(screen, color, (int(x), int(y)), int(star.size))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()