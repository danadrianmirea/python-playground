import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
GRAVITY = 0.2
THRUST = 0.3
ROTATION_SPEED = 3
FUEL_CONSUMPTION = 0.5

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Moon Lander")
clock = pygame.time.Clock()

class Lander:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = 100
        self.velocity_x = 0
        self.velocity_y = 0
        self.angle = 0
        self.fuel = 100
        self.landed = False
        self.crashed = False
        self.width = 20
        self.height = 30

    def update(self, thrusting, rotating_left, rotating_right):
        if not self.landed and not self.crashed:
            # Apply gravity
            self.velocity_y += GRAVITY

            # Apply thrust if fuel available
            if thrusting and self.fuel > 0:
                self.velocity_x += math.sin(math.radians(self.angle)) * THRUST
                self.velocity_y -= math.cos(math.radians(self.angle)) * THRUST
                self.fuel = max(0, self.fuel - FUEL_CONSUMPTION)

            # Handle rotation
            if rotating_left:
                self.angle = (self.angle + ROTATION_SPEED) % 360
            if rotating_right:
                self.angle = (self.angle - ROTATION_SPEED) % 360

            # Update position
            self.x += self.velocity_x
            self.y += self.velocity_y

            # Check for landing or crash
            if self.y + self.height >= HEIGHT - 50:  # Landing platform height
                if abs(self.velocity_x) < 2 and abs(self.velocity_y) < 2 and abs(self.angle) < 5:
                    self.landed = True
                else:
                    self.crashed = True

            # Screen boundaries
            self.x = max(0, min(WIDTH - self.width, self.x))
            if self.y < 0:
                self.y = 0
                self.velocity_y = 0

    def draw(self, screen):
        # Draw lander
        points = [
            (self.x + self.width//2, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]
        
        # Rotate points around center
        center_x = self.x + self.width//2
        center_y = self.y + self.height//2
        rotated_points = []
        for x, y in points:
            # Translate to origin
            x -= center_x
            y -= center_y
            # Rotate
            new_x = x * math.cos(math.radians(self.angle)) - y * math.sin(math.radians(self.angle))
            new_y = x * math.sin(math.radians(self.angle)) + y * math.cos(math.radians(self.angle))
            # Translate back
            rotated_points.append((new_x + center_x, new_y + center_y))
        
        pygame.draw.polygon(screen, WHITE, rotated_points)

        # Draw flame if thrusting
        if pygame.key.get_pressed()[pygame.K_UP] and self.fuel > 0:
            flame_points = [
                (self.x + self.width//2, self.y + self.height),
                (self.x + self.width//2 + 5, self.y + self.height + 15),
                (self.x + self.width//2 - 5, self.y + self.height + 15)
            ]
            pygame.draw.polygon(screen, RED, flame_points)

def draw_terrain(screen):
    # Simple terrain with a landing platform
    pygame.draw.rect(screen, GRAY, (0, HEIGHT - 50, WIDTH, 50))
    pygame.draw.rect(screen, GREEN, (WIDTH//2 - 50, HEIGHT - 50, 100, 5))

def draw_hud(screen, lander):
    font = pygame.font.SysFont(None, 24)
    
    # Fuel gauge
    fuel_text = font.render(f"Fuel: {lander.fuel:.1f}", True, WHITE)
    screen.blit(fuel_text, (10, 10))
    
    # Velocity
    velocity_text = font.render(f"Velocity X: {lander.velocity_x:.1f} Y: {lander.velocity_y:.1f}", True, WHITE)
    screen.blit(velocity_text, (10, 40))
    
    # Angle
    angle_text = font.render(f"Angle: {lander.angle:.1f}°", True, WHITE)
    screen.blit(angle_text, (10, 70))
    
    # Game state
    if lander.landed:
        success_text = font.render("LANDING SUCCESSFUL!", True, GREEN)
        screen.blit(success_text, (WIDTH//2 - 100, HEIGHT//2))
    elif lander.crashed:
        crash_text = font.render("CRASHED!", True, RED)
        screen.blit(crash_text, (WIDTH//2 - 50, HEIGHT//2))

def main():
    lander = Lander()
    running = True

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset game
                    lander = Lander()

        # Get key states
        keys = pygame.key.get_pressed()
        thrusting = keys[pygame.K_UP] or keys[pygame.K_w]  # Up arrow or W
        rotating_left = keys[pygame.K_LEFT] or keys[pygame.K_a]  # Left arrow or A
        rotating_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]  # Right arrow or D

        # Update game state
        lander.update(thrusting, rotating_left, rotating_right)

        # Draw everything
        screen.fill(BLACK)
        draw_terrain(screen)
        lander.draw(screen)
        draw_hud(screen, lander)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main() 