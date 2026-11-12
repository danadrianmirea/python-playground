import pygame
import random
import math

pygame.init()
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Lava Lamp")
clock = pygame.time.Clock()

class Blob:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = random.randint(15, 35)
        self.target_y = random.randint(100, HEIGHT - 100)
        self.speed = random.uniform(0.5, 1.5)
        self.direction = 1  # 1=up, -1=down
        self.color = random.choice([
            (255, 100, 0), (255, 50, 0), (200, 50, 0),
            (255, 150, 50), (255, 200, 50)
        ])
        self.shape_offset = 0

    def update(self):
        # Move toward target
        if abs(self.y - self.target_y) < 5:
            # Change target
            self.target_y = random.randint(100, HEIGHT - 100)
            self.direction *= -1
            self.speed = random.uniform(0.3, 1.2)

        if self.direction > 0:
            self.y += self.speed
        else:
            self.y -= self.speed

        # Slight horizontal drift
        self.x += math.sin(pygame.time.get_ticks() * 0.001 + self.shape_offset) * 0.3
        self.x = max(self.radius, min(WIDTH - self.radius, self.x))

    def draw(self, screen):
        # Draw blob with gradient effect
        for i in range(self.radius, 0, -5):
            alpha = int(255 * (i / self.radius))
            radius = i
            color = (min(255, self.color[0] + int(50 * (1 - i/self.radius))),
                    self.color[1],
                    self.color[2])
            pygame.draw.circle(screen, color, (int(self.x), int(self.y)), radius)

        # Highlight
        highlight_x = self.x - self.radius * 0.3
        highlight_y = self.y - self.radius * 0.3
        pygame.draw.circle(screen, (255, 255, 200, 50), 
                          (int(highlight_x), int(highlight_y)), 
                          int(self.radius * 0.3))

# Create blobs
blobs = []
for _ in range(8):
    x = random.randint(30, WIDTH - 30)
    y = random.randint(100, HEIGHT - 100)
    blobs.append(Blob(x, y))

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    # Background (lamp gradient)
    for y in range(HEIGHT):
        # Gradient from dark blue to black to dark red
        if y < HEIGHT // 2:
            progress = y / (HEIGHT // 2)
            r = int(10 * progress)
            g = int(10 * progress)
            b = int(50 * (1 - progress) + 10)
        else:
            progress = (y - HEIGHT // 2) / (HEIGHT // 2)
            r = int(10 + 30 * progress)
            g = int(10 * (1 - progress))
            b = int(10 * (1 - progress))
        pygame.draw.line(screen, (r, g, b), (0, y), (WIDTH, y))

    # Update and draw blobs
    for blob in blobs:
        blob.update()
        blob.draw(screen)

    # Glass reflections
    pygame.draw.rect(screen, (50, 50, 100, 20), (0, 0, WIDTH, 10))
    pygame.draw.rect(screen, (50, 50, 100, 20), (0, HEIGHT - 10, WIDTH, 10))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()