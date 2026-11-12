import pygame
import random
import math

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Starfield Screensaver")

BLACK = (0, 0, 0)

class Star:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)
        self.z = random.randint(1, 100)  # Depth
        self.speed = random.uniform(1, 5)
        self.brightness = random.randint(100, 255)
    
    def update(self):
        self.z -= self.speed
        if self.z <= 0:
            self.z = 100
            self.x = random.randint(0, SCREEN_WIDTH)
            self.y = random.randint(0, SCREEN_HEIGHT)
    
    def draw(self, screen):
        # Calculate size based on depth (closer = bigger)
        size = int((100 / self.z) * 2)
        if size < 1:
            size = 1
        
        # Calculate brightness
        brightness = int((self.z / 100) * 255)
        color = (brightness, brightness, brightness)
        
        pygame.draw.circle(screen, color, (self.x, self.y), size)

stars = [Star() for _ in range(200)]

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    screen.fill(BLACK)
    
    for star in stars:
        star.update()
        star.draw(screen)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()