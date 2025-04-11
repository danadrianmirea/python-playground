import pygame
import math
import sys
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Solar System")

# Colors
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
ORANGE = (255, 165, 0)
RED = (255, 0, 0)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (0, 0, 139)
GRAY = (128, 128, 128)

# Simulation settings
SIMULATION_SPEED = 1.5  # Base speed multiplier for the simulation

# View settings (similar to DLA)
ZOOM_FACTOR = 1.2
MIN_ZOOM = 0.1
MAX_ZOOM = 10.0
view_zoom = 0.83
view_offset_x = 227
view_offset_y = 65
is_panning = False
last_mouse_pos = None

# Celestial body class
class CelestialBody:
    def __init__(self, x, y, radius, color, mass, orbital_radius=None, orbital_speed=None, name=None):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mass = mass
        self.orbital_radius = orbital_radius
        self.orbital_speed = orbital_speed
        self.angle = 0  # For orbital motion
        self.name = name

    def update(self, dt):
        if self.orbital_radius and self.orbital_speed:
            # Update orbital position
            self.angle += self.orbital_speed * dt
            self.x = math.cos(self.angle) * self.orbital_radius
            self.y = math.sin(self.angle) * self.orbital_radius

    def draw(self, screen):
        # Convert world coordinates to screen coordinates
        screen_x = self.x * view_zoom + WIDTH/2 - view_offset_x * view_zoom
        screen_y = self.y * view_zoom + HEIGHT/2 - view_offset_y * view_zoom
        scaled_radius = max(1, int(self.radius * view_zoom))
        
        # Draw the celestial body
        pygame.draw.circle(screen, self.color, (int(screen_x), int(screen_y)), scaled_radius)
        
        # Draw the label if name is provided
        if self.name:
            # Create font with size that scales with zoom but has a minimum size
            font_size = max(12, int(16 * view_zoom))
            font = pygame.font.SysFont('Arial', font_size)
            text = font.render(self.name, True, WHITE)
            
            # Position the label above the planet
            text_rect = text.get_rect(center=(int(screen_x), int(screen_y) - scaled_radius - 10))
            screen.blit(text, text_rect)

# Create celestial bodies
sun = CelestialBody(0, 0, 30, YELLOW, 1.989e30)  # Sun at center

# Create planets with their relative orbital speeds
mercury = CelestialBody(60, 0, 4, GRAY, 3.285e23, 60, 4.17 * SIMULATION_SPEED, "Mercury")  # Mercury
venus = CelestialBody(90, 0, 6, ORANGE, 4.867e24, 90, 1.61 * SIMULATION_SPEED, "Venus")  # Venus
earth = CelestialBody(150, 0, 10, GREEN, 5.972e24, 150, 1.0 * SIMULATION_SPEED, "Earth")  # Earth
mars = CelestialBody(200, 0, 8, RED, 6.39e23, 200, 0.53 * SIMULATION_SPEED, "Mars")  # Mars
jupiter = CelestialBody(300, 0, 20, ORANGE, 1.898e27, 300, 0.084 * SIMULATION_SPEED, "Jupiter")  # Jupiter
saturn = CelestialBody(400, 0, 18, YELLOW, 5.683e26, 400, 0.034 * SIMULATION_SPEED, "Saturn")  # Saturn
uranus = CelestialBody(500, 0, 15, LIGHT_BLUE, 8.681e25, 500, 0.012 * SIMULATION_SPEED, "Uranus")  # Uranus
neptune = CelestialBody(600, 0, 15, DARK_BLUE, 1.024e26, 600, 0.006 * SIMULATION_SPEED, "Neptune")  # Neptune

# Function to convert screen coordinates to world coordinates
def screen_to_world(screen_x, screen_y):
    world_x = (screen_x - WIDTH/2 + view_offset_x * view_zoom) / view_zoom
    world_y = (screen_y - HEIGHT/2 + view_offset_y * view_zoom) / view_zoom
    return world_x, world_y

# Function to draw the solar system
def draw_solar_system():
    screen.fill(BLACK)
    
    # Draw celestial bodies
    sun.draw(screen)
    mercury.draw(screen)
    venus.draw(screen)
    earth.draw(screen)
    mars.draw(screen)
    jupiter.draw(screen)
    saturn.draw(screen)
    uranus.draw(screen)
    neptune.draw(screen)
    
    # Create font for UI text
    font = pygame.font.SysFont('Arial', 16)
    
    # Draw view information
    view_info = [
        f"Zoom: {view_zoom:.2f}x",
        f"View Offset X: {view_offset_x:.1f}",
        f"View Offset Y: {view_offset_y:.1f}"
    ]
    
    # Draw controls information
    controls_info = [
        "Controls:",
        "Left Click + Drag: Pan View",
        "Mouse Wheel: Zoom In/Out",
        "ESC: Quit"
    ]
    
    # Draw view information
    y_offset = 10
    for info in view_info:
        text = font.render(info, True, WHITE)
        screen.blit(text, (10, y_offset))
        y_offset += 20
    
    # Draw controls information
    y_offset = 10
    for info in controls_info:
        text = font.render(info, True, WHITE)
        screen.blit(text, (WIDTH - 200, y_offset))
        y_offset += 20
    
    pygame.display.flip()

# Main simulation loop
def run_simulation():
    global view_zoom, view_offset_x, view_offset_y, is_panning, last_mouse_pos
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    is_panning = True
                    last_mouse_pos = event.pos
                elif event.button == 4:  # Mouse wheel up
                    # Zoom in at mouse position
                    mouse_x, mouse_y = event.pos
                    world_x, world_y = screen_to_world(mouse_x, mouse_y)
                    old_zoom = view_zoom
                    view_zoom = min(view_zoom * ZOOM_FACTOR, MAX_ZOOM)
                    # Adjust offset to keep mouse position fixed
                    view_offset_x += (world_x - (mouse_x - WIDTH/2 + view_offset_x * old_zoom) / old_zoom)
                    view_offset_y += (world_y - (mouse_y - HEIGHT/2 + view_offset_y * old_zoom) / old_zoom)
                elif event.button == 5:  # Mouse wheel down
                    # Zoom out at mouse position
                    mouse_x, mouse_y = event.pos
                    world_x, world_y = screen_to_world(mouse_x, mouse_y)
                    old_zoom = view_zoom
                    view_zoom = max(view_zoom / ZOOM_FACTOR, MIN_ZOOM)
                    # Adjust offset to keep mouse position fixed
                    view_offset_x += (world_x - (mouse_x - WIDTH/2 + view_offset_x * old_zoom) / old_zoom)
                    view_offset_y += (world_y - (mouse_y - HEIGHT/2 + view_offset_y * old_zoom) / old_zoom)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click release
                    is_panning = False
            elif event.type == pygame.MOUSEMOTION:
                if is_panning and last_mouse_pos is not None:
                    # Calculate the movement in world coordinates
                    dx = (event.pos[0] - last_mouse_pos[0]) / view_zoom
                    dy = (event.pos[1] - last_mouse_pos[1]) / view_zoom
                    view_offset_x -= dx
                    view_offset_y -= dy
                    last_mouse_pos = event.pos
        
        # Update celestial bodies
        mercury.update(dt)
        venus.update(dt)
        earth.update(dt)
        mars.update(dt)
        jupiter.update(dt)
        saturn.update(dt)
        uranus.update(dt)
        neptune.update(dt)
        
        # Draw the solar system
        draw_solar_system()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation() 