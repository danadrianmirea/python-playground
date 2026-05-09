import pygame
import random
import math
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
FPS = 60
AIR_RESISTANCE = 0.995
MAX_PARTICLES = 5000

# Mutable settings (can be changed at runtime)
gravity = 0.15

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (100, 150, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 100)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Particle System Simulator")
clock = pygame.time.Clock()

class Particle:
    def __init__(self, x=None, y=None, particle_type=None):
        if x is None:
            self.pos = [random.uniform(0, WIDTH), random.uniform(0, HEIGHT)]
        else:
            self.pos = [x, y]
        
        if particle_type is None:
            self.particle_type = random.choice(['fire', 'smoke', 'energy', 'normal'])
        else:
            self.particle_type = particle_type
        
        # Velocity
        self.vel = [random.uniform(-3, 3), random.uniform(-3, 3)]
        
        # Acceleration (for forces)
        self.acc = [0, 0]
        
        # Size based on type
        if self.particle_type == 'fire':
            self.size = random.uniform(2, 6)
            self.color = (random.randint(200, 255), random.randint(100, 200), 0)
        elif self.particle_type == 'smoke':
            self.size = random.uniform(4, 10)
            self.color = (random.randint(50, 100), random.randint(50, 100), random.randint(50, 100))
        elif self.particle_type == 'energy':
            self.size = random.uniform(1, 4)
            self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        else:
            self.size = random.uniform(2, 5)
            self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Life (for fading particles)
        self.life = 1.0
        self.decay = random.uniform(0.001, 0.005)
        
        # Mass (affects how forces affect it)
        self.mass = self.size * 0.5
    
    def apply_force(self, force_x, force_y):
        """Apply force to particle (F = ma, so a = F/m)."""
        self.acc[0] += force_x / self.mass
        self.acc[1] += force_y / self.mass
    
    def update(self):
        """Update particle position and apply physics."""
        # Apply gravity
        self.acc[1] += gravity
        
        # Apply air resistance
        self.vel[0] *= AIR_RESISTANCE
        self.vel[1] *= AIR_RESISTANCE
        
        # Update velocity
        self.vel[0] += self.acc[0]
        self.vel[1] += self.acc[1]
        
        # Update position
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        # Boundary collision
        if self.pos[0] < 0:
            self.pos[0] = 0
            self.vel[0] *= -0.8
        if self.pos[0] > WIDTH:
            self.pos[0] = WIDTH
            self.vel[0] *= -0.8
        if self.pos[1] < 0:
            self.pos[1] = 0
            self.vel[1] *= -0.8
        if self.pos[1] > HEIGHT:
            self.pos[1] = HEIGHT
            self.vel[1] *= -0.8
        
        # Decay life
        self.life -= self.decay
    
    def draw(self, surface):
        """Draw particle on surface."""
        if self.life <= 0:
            return
        
        # Draw particle with transparency
        alpha = int(255 * self.life)
        color_with_alpha = (self.color[0], self.color[1], self.color[2], alpha)
        
        # Create a surface for alpha-blended drawing
        particle_surf = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
        pygame.draw.circle(particle_surf, color_with_alpha, (int(self.size), int(self.size)), int(self.size))
        surface.blit(particle_surf, (int(self.pos[0] - self.size), int(self.pos[1] - self.size)))
        
        # Add glow effect for some particles
        if self.particle_type == 'energy':
            glow_size = int(self.size * 2)
            pygame.draw.circle(surface, (255, 255, 255), (int(self.pos[0]), int(self.pos[1])), glow_size, 1)
    
    def fade(self, amount):
        """Reduce life by amount."""
        self.life -= amount


class ParticleSystem:
    """Manages all particles in the simulation."""
    
    def __init__(self):
        """Initialize particle system."""
        self.particles = []
        self.mouse_pos = [0, 0]
        self.mouse_down = False
        self.particle_count = 0
        self.max_particles = MAX_PARTICLES
    
    def spawn_particles(self, x, y, count=10, force=5):
        """Spawn multiple particles at position."""
        for _ in range(count):
            if len(self.particles) < self.max_particles:
                # Add explosion force
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, force)
                self.particles.append(Particle(x, y))
                self.particles[-1].vel[0] += math.cos(angle) * speed
                self.particles[-1].vel[1] += math.sin(angle) * speed
    
    def add_particle(self, x, y):
        """Add a single particle."""
        if len(self.particles) < self.max_particles:
            self.particles.append(Particle(x, y))
    
    def clear(self):
        """Clear all particles."""
        self.particles = []
    
    def update(self):
        """Update all particles."""
        for particle in self.particles[:]:
            particle.update()
        # Filter out dead particles efficiently
        self.particles = [p for p in self.particles if p.life > 0]
    
    def draw(self, surface):
        """Draw all particles."""
        for particle in self.particles:
            particle.draw(surface)
    
    def get_particle_count(self):
        """Return current particle count."""
        return len(self.particles)
    
    def get_stats(self):
        """Get system statistics."""
        return {
            'count': len(self.particles),
            'max': self.max_particles
        }


# Create particle system
system = ParticleSystem()

# UI controls
show_stats = True
show_grid = False
auto_spawn = False
spawn_timer = 0


def draw_ui(surface):
    """Draw UI overlay."""
    # Title
    font = pygame.font.Font(None, 24)
    title = font.render("Particle System Simulator", True, WHITE)
    surface.blit(title, (10, 10))
    
    # Instructions
    instructions = [
        "Mouse: Click to spawn particles",
        "Mouse Wheel: Adjust gravity",
        "Space: Pause/Resume",
        "C: Clear all particles",
        "R: Reset simulation",
        "G: Toggle grid",
        "S: Toggle auto-spawn",
    ]
    
    for i, text in enumerate(instructions):
        surface.blit(font.render(text, True, (200, 200, 200)), (10, 35 + i * 20))
    
    # Stats
    if show_stats:
        stats = system.get_stats()
        stats_text = f"Particles: {stats['count']}/{stats['max']}"
        surface.blit(font.render(stats_text, True, YELLOW), (10, HEIGHT - 25))


def draw_grid(surface):
    """Draw background grid."""
    if not show_grid:
        return
    
    color = (30, 30, 30)
    for x in range(0, WIDTH, 50):
        pygame.draw.line(surface, color, (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, 50):
        pygame.draw.line(surface, color, (0, y), (WIDTH, y), 1)


def main():
    """Main game loop."""
    global show_grid, auto_spawn, spawn_timer
    running = True
    paused = False
    
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    system.mouse_down = True
                    # Spawn particles on click
                    system.spawn_particles(event.pos[0], event.pos[1], count=5, force=8)
                
                elif event.button == 3:  # Right click
                    system.clear()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                system.mouse_down = False
            
            elif event.type == pygame.MOUSEMOTION:
                system.mouse_pos = event.pos
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_c:
                    system.clear()
                elif event.key == pygame.K_r:
                    system.clear()
                    # Add initial particles
                    for _ in range(100):
                        system.add_particle(random.randint(0, WIDTH), random.randint(0, HEIGHT))
                elif event.key == pygame.K_g:
                    show_grid = not show_grid
                elif event.key == pygame.K_s:
                    auto_spawn = not auto_spawn
            
            elif event.type == pygame.MOUSEWHEEL:
                # Adjust gravity (event.y: positive = scroll up, negative = scroll down)
                global gravity
                gravity += event.y * 0.01
                gravity = max(0.05, min(0.5, gravity))
        
        # Auto-spawn (when auto-spawn mode is on)
        if auto_spawn and not paused:
            spawn_timer += 1
            if spawn_timer >= 10:
                system.spawn_particles(
                    random.randint(0, WIDTH),
                    random.randint(0, HEIGHT),
                    count=3,
                    force=4
                )
                spawn_timer = 0
        
        # Continuous spawn while mouse is held down
        if system.mouse_down and not paused:
            system.spawn_particles(system.mouse_pos[0], system.mouse_pos[1], count=2, force=5)
        
        # Update
        if not paused:
            system.update()
        
        # Draw
        screen.fill(BLACK)
        draw_grid(screen)
        system.draw(screen)
        draw_ui(screen)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()