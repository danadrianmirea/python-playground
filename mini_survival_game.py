import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Create a window
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Mini Survival Game")

# Hide the default cursor
pygame.mouse.set_visible(False)

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Game state
game_active = True
game_over_timer = 0
game_over_delay = 2000  # ms before restart after game over

# Player properties
player_x = WINDOW_WIDTH // 2  # Start in center
player_y = WINDOW_HEIGHT // 2
player_radius = 20
player_speed = 300  # Speed in pixels per second

# Gun properties
current_angle = 0
edge_x = 0
edge_y = 0
gun_end_x = 0
gun_end_y = 0

# Bullet properties
bullets = []  # List to store active bullets
bullet_speed = 300  # Speed in pixels per second
bullet_radius = 4
fire_rate = 0.3  # Time between shots in seconds

# Enemy properties
enemies = []  # List to store active enemies
enemy_radius = 18
enemy_speed = 100  # Speed in pixels per second
enemy_spawn_rate = 2.0  # Time between spawns in seconds
max_enemies = 10  # Maximum number of enemies at once
last_enemy_spawn = 0

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

# Crosshair properties
CROSSHAIR_SIZE = 20  # Size of the crosshair
CROSSHAIR_THICKNESS = 2  # Thickness of the lines
CROSSHAIR_GAP = 4  # Gap in the center

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

class Enemy:
    def __init__(self, player_x, player_y):
        # Spawn from outside the screen
        side = random.randint(0, 3)  # 0: top, 1: right, 2: bottom, 3: left
        
        if side == 0:  # top
            self.x = random.randint(0, WINDOW_WIDTH)
            self.y = -enemy_radius
        elif side == 1:  # right
            self.x = WINDOW_WIDTH + enemy_radius
            self.y = random.randint(0, WINDOW_HEIGHT)
        elif side == 2:  # bottom
            self.x = random.randint(0, WINDOW_WIDTH)
            self.y = WINDOW_HEIGHT + enemy_radius
        else:  # left
            self.x = -enemy_radius
            self.y = random.randint(0, WINDOW_HEIGHT)
        
    def update(self, dt, player_x, player_y):
        # Calculate direction towards player
        dx = player_x - self.x
        dy = player_y - self.y
        distance = max(0.1, math.sqrt(dx * dx + dy * dy))  # Avoid division by zero
        
        # Normalize and scale by speed
        self.x += (dx / distance) * enemy_speed * dt
        self.y += (dy / distance) * enemy_speed * dt
    
    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), enemy_radius)
    
    def collides_with_player(self, player_x, player_y, player_radius):
        distance = math.sqrt((self.x - player_x)**2 + (self.y - player_y)**2)
        return distance < (enemy_radius + player_radius)
    
    def collides_with_bullet(self, bullet):
        distance = math.sqrt((self.x - bullet.x)**2 + (self.y - bullet.y)**2)
        return distance < (enemy_radius + bullet_radius)

def spawn_enemy():
    if len(enemies) < max_enemies:
        enemies.append(Enemy(player_x, player_y))

def reset_game():
    global player_x, player_y, bullets, enemies, game_active
    player_x = WINDOW_WIDTH // 2
    player_y = WINDOW_HEIGHT // 2
    bullets = []
    enemies = []
    game_active = True

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
        elif event.type == pygame.MOUSEBUTTONDOWN and game_active:
            if event.button == 1:  # Left click
                auto_fire = True
                # Force immediate first shot
                shoot_timer = current_time - (fire_rate * 1000 + 1)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left release
                auto_fire = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_active:
                auto_fire = True
                # Force immediate first shot
                shoot_timer = current_time - (fire_rate * 1000 + 1)
            elif event.key in key_states and game_active:
                key_states[event.key] = True
            elif event.key == pygame.K_r and not game_active:
                reset_game()
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                auto_fire = False
            elif event.key in key_states:
                key_states[event.key] = False
    
    if game_active:
        # Get mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Handle movement
        move_dx = 0
        move_dy = 0
        
        if key_states[pygame.K_w]:
            move_dy -= player_speed * dt
        if key_states[pygame.K_s]:
            move_dy += player_speed * dt
        if key_states[pygame.K_a]:
            move_dx -= player_speed * dt
        if key_states[pygame.K_d]:
            move_dx += player_speed * dt
        
        # Calculate new position
        new_x = player_x + move_dx
        new_y = player_y + move_dy
        
        # Check boundaries and update position
        if new_x - player_radius >= 0 and new_x + player_radius <= WINDOW_WIDTH:
            player_x = new_x
        if new_y - player_radius >= 0 and new_y + player_radius <= WINDOW_HEIGHT:
            player_y = new_y
        
        # Update gun angle and position
        aim_dx = mouse_x - player_x
        aim_dy = mouse_y - player_y
        current_angle = math.atan2(aim_dy, aim_dx)
        
        # Calculate the point on the circle's edge where the gun should be
        edge_x = player_x + player_radius * math.cos(current_angle)
        edge_y = player_y + player_radius * math.sin(current_angle)
        
        # Calculate the end point of the gun line
        gun_end_x = edge_x + player_radius * 0.6 * math.cos(current_angle)
        gun_end_y = edge_y + player_radius * 0.6 * math.sin(current_angle)
        
        # Handle shooting
        if auto_fire and current_time - shoot_timer >= fire_rate * 1000:
            # Create new bullet
            bullets.append(Bullet(edge_x, edge_y, current_angle))
            shoot_timer = current_time  # Reset timer
        
        # Enemy spawning
        if current_time - last_enemy_spawn >= enemy_spawn_rate * 1000:
            spawn_enemy()
            last_enemy_spawn = current_time
        
        # Update bullets
        for bullet in bullets[:]:
            bullet.update(dt)
            if bullet.is_off_screen():
                bullets.remove(bullet)
        
        # Update enemies and check for collisions
        for enemy in enemies[:]:
            enemy.update(dt, player_x, player_y)
            
            # Check for enemy-player collision
            if enemy.collides_with_player(player_x, player_y, player_radius):
                game_active = False
                game_over_timer = current_time
                break
            
            # Check for enemy-bullet collisions
            for bullet in bullets[:]:
                if enemy.collides_with_bullet(bullet):
                    if bullet in bullets:  # Make sure bullet wasn't already removed
                        bullets.remove(bullet)
                    if enemy in enemies:  # Make sure enemy wasn't already removed
                        enemies.remove(enemy)
                    spawn_enemy()  # Spawn a new enemy
                    break
    else:
        # Game over state
        if current_time - game_over_timer >= game_over_delay:
            # Automatically restart after delay
            reset_game()
    
    # Draw everything
    screen.fill(BLACK)
    
    if game_active:
        # Draw player
        pygame.draw.circle(screen, BLUE, (int(player_x), int(player_y)), player_radius)
        
        # Draw gun line
        pygame.draw.line(screen, GREEN, 
                        (int(edge_x), int(edge_y)), 
                        (int(gun_end_x), int(gun_end_y)), 2)
        
        # Draw bullets
        for bullet in bullets:
            bullet.draw(screen)
            
        # Draw enemies
        for enemy in enemies:
            enemy.draw(screen)
            
        # Draw crosshair
        # Horizontal line
        pygame.draw.line(screen, WHITE, 
                        (mouse_x - CROSSHAIR_SIZE//2, mouse_y),
                        (mouse_x - CROSSHAIR_GAP//2, mouse_y),
                        CROSSHAIR_THICKNESS)
        pygame.draw.line(screen, WHITE,
                        (mouse_x + CROSSHAIR_GAP//2, mouse_y),
                        (mouse_x + CROSSHAIR_SIZE//2, mouse_y),
                        CROSSHAIR_THICKNESS)
        # Vertical line
        pygame.draw.line(screen, WHITE,
                        (mouse_x, mouse_y - CROSSHAIR_SIZE//2),
                        (mouse_x, mouse_y - CROSSHAIR_GAP//2),
                        CROSSHAIR_THICKNESS)
        pygame.draw.line(screen, WHITE,
                        (mouse_x, mouse_y + CROSSHAIR_GAP//2),
                        (mouse_x, mouse_y + CROSSHAIR_SIZE//2),
                        CROSSHAIR_THICKNESS)
    else:
        # Draw game over screen
        font = pygame.font.SysFont(None, 72)
        game_over_text = font.render("GAME OVER", True, RED)
        restart_text = font.render("Press R to restart", True, WHITE)
        
        text_rect = game_over_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 40))
        restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 40))
        
        screen.blit(game_over_text, text_rect)
        screen.blit(restart_text, restart_rect)
    
    # Update display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)

pygame.quit() 