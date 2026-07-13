import pygame
import random
import math

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Bouncing Shapes Screensaver")

BLACK = (0, 0, 0)

class Shape:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)
        self.size = random.randint(20, 60)
        self.speed_x = random.uniform(-2, 2)
        self.speed_y = random.uniform(-2, 2)
        self.color = (random.randint(50, 255), 
                      random.randint(50, 255), 
                      random.randint(50, 255))
        self.rotation = 0
        self.rotation_speed = random.uniform(-2, 2)
        self.shape_type = random.choice(['circle', 'square', 'triangle'])
    
    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y
        self.rotation += self.rotation_speed
        
        # Bounce and change color
        if self.x <= 0 or self.x >= SCREEN_WIDTH:
            self.speed_x *= -1
            self.color = (random.randint(50, 255), 
                          random.randint(50, 255), 
                          random.randint(50, 255))
        if self.y <= 0 or self.y >= SCREEN_HEIGHT:
            self.speed_y *= -1
            self.color = (random.randint(50, 255), 
                          random.randint(50, 255), 
                          random.randint(50, 255))
    
    def draw(self, screen):
        if self.shape_type == 'circle':
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.size))
        elif self.shape_type == 'square':
            rect = pygame.Rect(self.x - self.size/2, self.y - self.size/2, self.size, self.size)
            pygame.draw.rect(screen, self.color, rect)
        else:  # triangle
            points = [
                (self.x, self.y - self.size),
                (self.x - self.size, self.y + self.size),
                (self.x + self.size, self.y + self.size)
            ]
            # Rotate points
            center = (self.x, self.y)
            rotated_points = []
            for point in points:
                dx = point[0] - center[0]
                dy = point[1] - center[1]
                angle = math.radians(self.rotation)
                new_x = dx * math.cos(angle) - dy * math.sin(angle) + center[0]
                new_y = dx * math.sin(angle) + dy * math.cos(angle) + center[1]
                rotated_points.append((new_x, new_y))
            pygame.draw.polygon(screen, self.color, rotated_points)

shapes = [Shape() for _ in range(15)]

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                shapes = [Shape() for _ in range(15)]
    
    screen.fill(BLACK)
    
    for shape in shapes:
        shape.update()
        shape.draw(screen)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()