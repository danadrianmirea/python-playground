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
            # Fire: warm gradient from yellow to red
            r = random.randint(220, 255)
            g = random.randint(80, 200)
            b = random.randint(0, 50)
            self.color = (r, g, b)
        elif self.particle_type == 'smoke':
            self.size = random.uniform(4, 10)
            # Smoke: subtle purple-blue-gray tones
            base = random.randint(60, 140)
            self.color = (
                random.randint(base - 20, base + 20),
                random.randint(base - 30, base),
                random.randint(base + 10, base + 60)
            )
        elif self.particle_type == 'energy':
            self.size = random.uniform(1, 4)
            # Energy: bright neon colors with high saturation
            hue = random.uniform(0, 360)
            self.color = self._hsv_to_rgb(hue, 0.9, 1.0)
        else:
            self.size = random.uniform(2, 5)
            # Normal: vibrant random colors with good saturation
            hue = random.uniform(0, 360)
            saturation = random.uniform(0.6, 1.0)
            value = random.uniform(0.7, 1.0)
            self.color = self._hsv_to_rgb(hue, saturation, value)
        
        # Life (for fading particles)
        self.life = 1.0
        self.decay = random.uniform(0.001, 0.005)
        
        # Mass (affects how forces affect it)
        self.mass = self.size * 0.5
        
        # Color shift rate (for animated colors)
        self.color_shift = random.uniform(-2, 2)
        self.hue = random.uniform(0, 360)
    
    @staticmethod
    def _hsv_to_rgb(h, s, v):
        """Convert HSV to RGB. h in [0,360], s,v in [0,1]."""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
    
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
        
        # Animate color - shift hue over time for a rainbow effect
        if self.particle_type in ('energy', 'normal'):
            self.hue = (self.hue + self.color_shift) % 360
            self.color = self._hsv_to_rgb(self.hue, 0.9, 0.8 + 0.2 * self.life)
    
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
            glow_alpha = int(100 * self.life)
            glow_color = (self.color[0], self.color[1], self.color[2], glow_alpha)
            glow_surf = pygame.Surface((int(self.size * 4), int(self.size * 4)), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (int(self.size * 2), int(self.size * 2)), glow_size)
            surface.blit(glow_surf, (int(self.pos[0] - self.size * 2), int(self.pos[1] - self.size * 2)))
        
        # Add a bright core for fire particles
        if self.particle_type == 'fire' and self.life > 0.5:
            core_size = max(1, int(self.size * 0.4))
            core_alpha = int(200 * self.life)
            core_color = (255, 255, 200, core_alpha)
            core_surf = pygame.Surface((int(core_size * 2), int(core_size * 2)), pygame.SRCALPHA)
            pygame.draw.circle(core_surf, core_color, (core_size, core_size), core_size)
            surface.blit(core_surf, (int(self.pos[0] - core_size), int(self.pos[1] - core_size)))
    
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