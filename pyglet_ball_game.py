import pygame
import math

# Initialize Pygame
pygame.init()

# Create a window
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Ball Game")

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Ball properties
ball_x = WINDOW_WIDTH // 2  # Start in center
ball_y = WINDOW_HEIGHT // 2
ball_radius = 20
ball_speed = 300  # Speed in pixels per second

# Gun properties
current_angle = 0
edge_x = 0
edge_y = 0
gun_end_x = 0
gun_end_y = 0

# Bullet properties
bullets = []  # List to store active bullets
bullet_speed = 500  # Speed in pixels per second
bullet_radius = 4
fire_rate = 0.1  # Time between shots in seconds

# Shooting state
auto_fire = False
shoot_timer = 0

# Movement tracking
key_states = {
    pygame.K_w: False,
    pygame.K_s: False,
    pygame.K_a: False,
    pygame.K_d: False
}

class Bullet:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
    
    def update(self, dt):
        self.x += math.cos(self.angle) * bullet_speed * dt
        self.y += math.sin(self.angle) * bullet_speed * dt
    
    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), bullet_radius)
    
    def is_off_screen(self):
        return (self.x < 0 or self.x > WINDOW_WIDTH or 
                self.y < 0 or self.y > WINDOW_HEIGHT)

# Game loop
running = True
clock = pygame.time.Clock()

while running:
    # Get delta time
    dt = clock.get_time() / 1000.0  # Convert milliseconds to seconds
    current_time = pygame.time.get_ticks()
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                auto_fire = True
                # Force immediate first shot
                shoot_timer = current_time - (fire_rate * 1000 + 1)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left release
                auto_fire = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                auto_fire = True
                # Force immediate first shot
                shoot_timer = current_time - (fire_rate * 1000 + 1)
            elif event.key in key_states:
                key_states[event.key] = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                auto_fire = False
            elif event.key in key_states:
                key_states[event.key] = False
    
    # Get mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()
    
    # Handle movement
    move_dx = 0
    move_dy = 0
    
    if key_states[pygame.K_w]:
        move_dy -= ball_speed * dt
    if key_states[pygame.K_s]:
        move_dy += ball_speed * dt
    if key_states[pygame.K_a]:
        move_dx -= ball_speed * dt
    if key_states[pygame.K_d]:
        move_dx += ball_speed * dt
    
    # Calculate new position
    new_x = ball_x + move_dx
    new_y = ball_y + move_dy
    
    # Check boundaries and update position
    if new_x - ball_radius >= 0 and new_x + ball_radius <= WINDOW_WIDTH:
        ball_x = new_x
    if new_y - ball_radius >= 0 and new_y + ball_radius <= WINDOW_HEIGHT:
        ball_y = new_y
    
    # Update gun angle and position
    aim_dx = mouse_x - ball_x
    aim_dy = mouse_y - ball_y
    current_angle = math.atan2(aim_dy, aim_dx)
    
    # Calculate the point on the circle's edge where the gun should be
    edge_x = ball_x + ball_radius * math.cos(current_angle)
    edge_y = ball_y + ball_radius * math.sin(current_angle)
    
    # Calculate the end point of the gun line
    gun_end_x = edge_x + ball_radius * 0.6 * math.cos(current_angle)
    gun_end_y = edge_y + ball_radius * 0.6 * math.sin(current_angle)
    
    # Handle shooting
    if auto_fire and current_time - shoot_timer >= fire_rate * 1000:
        # Create new bullet
        bullets.append(Bullet(edge_x, edge_y, current_angle))
        shoot_timer = current_time  # Reset timer
    
    # Update bullets
    for bullet in bullets[:]:
        bullet.update(dt)
        if bullet.is_off_screen():
            bullets.remove(bullet)
    
    # Draw everything
    screen.fill(BLACK)
    
    # Draw ball
    pygame.draw.circle(screen, RED, (int(ball_x), int(ball_y)), ball_radius)
    
    # Draw gun line
    pygame.draw.line(screen, GREEN, 
                    (int(edge_x), int(edge_y)), 
                    (int(gun_end_x), int(gun_end_y)), 2)
    
    # Draw bullets
    for bullet in bullets:
        bullet.draw(screen)
    
    # Update display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)

pygame.quit() 