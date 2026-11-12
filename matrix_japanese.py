import pygame
import random

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Matrix - Katakana")

# Japanese Katakana characters (most common in Matrix)
CHARS = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"

# Try to use a Japanese font
try:
    font = pygame.font.SysFont("MS Gothic", 20)
except:
    font = pygame.font.Font(None, 22)

class Drop:
    def __init__(self, x):
        self.x = x
        self.y = random.randint(-200, -10)
        self.speed = random.uniform(1, 5)
        self.length = random.randint(10, 20)
        self.chars = [random.choice(CHARS) for _ in range(self.length)]
    
    def update(self):
        self.y += self.speed
        if self.y > HEIGHT + 50:
            self.__init__(self.x)
    
    def draw(self, screen):
        for i, char in enumerate(self.chars):
            y = self.y + i * 20
            if 0 <= y <= HEIGHT:
                brightness = max(0, 255 - i * 20)
                color = (0, brightness, 0) if i > 0 else (0, 255, 0)
                if i == 0 and random.random() < 0.1:
                    color = (0, 255, 100)  # Bright head
                screen.blit(font.render(char, True, color), (self.x, y))

# Create drops
drops = [Drop(x) for x in range(0, WIDTH, 20)]

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            drops = [Drop(x) for x in range(0, WIDTH, 20)]
    
    screen.fill((0, 0, 0))
    for drop in drops:
        drop.update()
        drop.draw(screen)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()