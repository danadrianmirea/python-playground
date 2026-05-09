import pygame
import random
import math
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fire Particle System")
clock = pygame.time.Clock()


class FireParticle:
    """A single fire particle with realistic flame behavior."""

    def __init__(self, x, y, particle_type='flame'):
        self.pos = [x, y]
        self.particle_type = particle_type  # 'flame', 'ember', 'smoke'

        if particle_type == 'flame':
            # Flame: fast, small, bright
            self.size = random.uniform(3, 8)
            self.vel = [random.uniform(-0.8, 0.8), random.uniform(-4, -1.5)]
            self.life = random.uniform(0.6, 1.0)
            self.decay = random.uniform(0.008, 0.025)
            # Fire colors: yellow -> orange -> red
            hue_choice = random.random()
            if hue_choice < 0.3:
                # Bright yellow-white core
                self.color = (255, random.randint(220, 255), random.randint(150, 220))
            elif hue_choice < 0.7:
                # Orange
                self.color = (random.randint(220, 255), random.randint(120, 200), random.randint(0, 60))
            else:
                # Deep red
                self.color = (random.randint(180, 240), random.randint(40, 100), random.randint(0, 30))
            self.turbulence = random.uniform(0.3, 1.0)

        elif particle_type == 'ember':
            # Ember: tiny bright sparks that fly up
            self.size = random.uniform(1, 3)
            self.vel = [random.uniform(-1.5, 1.5), random.uniform(-5, -2)]
            self.life = random.uniform(0.3, 0.8)
            self.decay = random.uniform(0.01, 0.04)
            # Embers are bright yellow-white
            brightness = random.randint(200, 255)
            self.color = (brightness, brightness, random.randint(100, 200))
            self.turbulence = random.uniform(0.5, 1.5)

        else:  # smoke
            # Smoke: slow, large, dark, rises gently
            self.size = random.uniform(5, 15)
            self.vel = [random.uniform(-0.3, 0.3), random.uniform(-0.8, -0.3)]
            self.life = random.uniform(0.8, 1.0)
            self.decay = random.uniform(0.003, 0.008)
            # Smoke: dark gray with slight color tint
            base = random.randint(30, 80)
            tint = random.choice(['gray', 'blue', 'purple'])
            if tint == 'gray':
                self.color = (base, base, base)
            elif tint == 'blue':
                self.color = (base - 10, base, base + 20)
            else:  # purple
                self.color = (base + 10, base, base + 20)
            self.turbulence = random.uniform(0.1, 0.4)

        # For wind/turbulence effects
        self.acc = [0, 0]
        self.mass = self.size * 0.3

    def update(self, wind=0.0):
        """Update particle position with turbulence and wind."""
        # Add turbulence (random flickering)
        self.acc[0] += random.uniform(-self.turbulence, self.turbulence) * 0.1
        self.acc[1] += random.uniform(-0.1, 0.1) * 0.05

        # Add wind
        self.acc[0] += wind * 0.01

        # Apply acceleration
        self.vel[0] += self.acc[0]
        self.vel[1] += self.acc[1]

        # Dampen velocity
        self.vel[0] *= 0.98
        self.vel[1] *= 0.98

        # Update position
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        # Decay life
        self.life -= self.decay

        # Reset acceleration
        self.acc = [0, 0]

        # As flame particles age, shift color toward red/dark
        if self.particle_type == 'flame':
            age_factor = 1.0 - self.life  # 0 = new, 1 = dying
            r, g, b = self.color
            # Shift toward red then dark
            r = min(255, int(r * (1 + age_factor * 0.3)))
            g = max(0, int(g * (1 - age_factor * 0.7)))
            b = max(0, int(b * (1 - age_factor * 0.9)))
            self.color = (r, g, b)

        # Smoke expands as it rises
        if self.particle_type == 'smoke':
            self.size += 0.05

    def draw(self, surface):
        """Draw the particle with appropriate effects."""
        if self.life <= 0:
            return

        alpha = int(255 * self.life)

        if self.particle_type == 'flame':
            # Flame: draw with a soft glow
            # Main flame body
            flame_surf = pygame.Surface((int(self.size * 3), int(self.size * 3)), pygame.SRCALPHA)
            color_with_alpha = (self.color[0], self.color[1], self.color[2], alpha)
            pygame.draw.circle(flame_surf, color_with_alpha,
                               (int(self.size * 1.5), int(self.size * 1.5)), int(self.size))
            surface.blit(flame_surf, (int(self.pos[0] - self.size * 1.5), int(self.pos[1] - self.size * 1.5)))

            # Outer glow
            glow_alpha = int(60 * self.life)
            glow_color = (self.color[0], self.color[1], self.color[2], glow_alpha)
            glow_surf = pygame.Surface((int(self.size * 5), int(self.size * 5)), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color,
                               (int(self.size * 2.5), int(self.size * 2.5)), int(self.size * 1.5))
            surface.blit(glow_surf, (int(self.pos[0] - self.size * 2.5), int(self.pos[1] - self.size * 2.5)))

        elif self.particle_type == 'ember':
            # Ember: tiny bright dot with a small glow
            ember_surf = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
            color_with_alpha = (self.color[0], self.color[1], self.color[2], alpha)
            pygame.draw.circle(ember_surf, color_with_alpha,
                               (int(self.size), int(self.size)), max(1, int(self.size)))
            surface.blit(ember_surf, (int(self.pos[0] - self.size), int(self.pos[1] - self.size)))

        else:  # smoke
            # Smoke: soft, semi-transparent circles
            smoke_surf = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
            smoke_alpha = int(80 * self.life)
            smoke_color = (self.color[0], self.color[1], self.color[2], smoke_alpha)
            pygame.draw.circle(smoke_surf, smoke_color,
                               (int(self.size), int(self.size)), int(self.size))
            surface.blit(smoke_surf, (int(self.pos[0] - self.size), int(self.pos[1] - self.size)))


class FireSystem:
    """Manages the entire fire simulation."""

    def __init__(self):
        self.particles = []
        self.max_particles = 3000
        self.wind = 0.0
        self.target_wind = 0.0
        self.intensity = 1.0  # Multiplier for spawn rate

        # Fire source configuration
        self.fire_sources = []
        # Default: a campfire at the bottom center
        self.add_fire_source(WIDTH // 2, HEIGHT - 50, width=120, intensity=1.0)

    def add_fire_source(self, x, y, width=80, intensity=1.0):
        """Add a fire source at position with given width."""
        self.fire_sources.append({
            'x': x,
            'y': y,
            'width': width,
            'intensity': intensity
        })

    def clear_sources(self):
        """Remove all fire sources."""
        self.fire_sources = []

    def spawn_particles(self):
        """Spawn new particles from all fire sources."""
        for source in self.fire_sources:
            spawn_count = int(3 * source['intensity'] * self.intensity)
            for _ in range(spawn_count):
                if len(self.particles) >= self.max_particles:
                    break

                # Random position within the fire source width
                offset_x = random.uniform(-source['width'] // 2, source['width'] // 2)
                x = source['x'] + offset_x
                y = source['y'] + random.uniform(-5, 5)

                # Determine particle type
                roll = random.random()
                if roll < 0.6:
                    p_type = 'flame'
                elif roll < 0.85:
                    p_type = 'ember'
                else:
                    p_type = 'smoke'

                self.particles.append(FireParticle(x, y, p_type))

    def update(self):
        """Update all particles and wind."""
        # Smooth wind changes
        self.wind += (self.target_wind - self.wind) * 0.01

        # Spawn new particles
        self.spawn_particles()

        # Update existing particles
        for particle in self.particles[:]:
            particle.update(self.wind)

        # Remove dead particles
        self.particles = [p for p in self.particles if p.life > 0]

    def draw(self, surface):
        """Draw all particles."""
        # Draw smoke first (behind flames)
        for particle in self.particles:
            if particle.particle_type == 'smoke':
                particle.draw(surface)

        # Draw flames and embers on top
        for particle in self.particles:
            if particle.particle_type != 'smoke':
                particle.draw(surface)

    def get_stats(self):
        """Get system statistics."""
        flame_count = sum(1 for p in self.particles if p.particle_type == 'flame')
        ember_count = sum(1 for p in self.particles if p.particle_type == 'ember')
        smoke_count = sum(1 for p in self.particles if p.particle_type == 'smoke')
        return {
            'total': len(self.particles),
            'flames': flame_count,
            'embers': ember_count,
            'smoke': smoke_count,
            'max': self.max_particles,
            'wind': self.wind,
            'intensity': self.intensity
        }


def draw_ui(surface, system):
    """Draw UI overlay."""
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)

    # Title
    title = font.render("Fire Particle System", True, (255, 200, 100))
    surface.blit(title, (10, 10))

    # Instructions
    instructions = [
        "Left Click: Add fire source",
        "Right Click: Clear all fire sources",
        "Scroll Up/Down: Change intensity",
        "W: Increase wind (right)",
        "Q: Decrease wind (left)",
        "Space: Pause/Resume",
        "C: Clear all particles",
        "R: Reset to default campfire",
    ]

    for i, text in enumerate(instructions):
        surface.blit(small_font.render(text, True, (180, 180, 180)), (10, 35 + i * 18))

    # Stats
    stats = system.get_stats()
    stats_lines = [
        f"Particles: {stats['total']}/{stats['max']}",
        f"  Flames: {stats['flames']}  Embers: {stats['embers']}  Smoke: {stats['smoke']}",
        f"Wind: {stats['wind']:.1f}",
        f"Intensity: {stats['intensity']:.1f}",
    ]
    for i, text in enumerate(stats_lines):
        surface.blit(small_font.render(text, True, (255, 200, 100)), (10, HEIGHT - 80 + i * 18))

    # Draw fire source indicators
    for source in system.fire_sources:
        # Draw a subtle ground glow at each fire source
        glow_surf = pygame.Surface((source['width'] + 40, 20), pygame.SRCALPHA)
        glow_color = (255, 150, 50, 30)
        pygame.draw.ellipse(glow_surf, glow_color, (0, 0, source['width'] + 40, 20))
        surface.blit(glow_surf, (source['x'] - source['width'] // 2 - 20, source['y'] - 5))


def main():
    """Main game loop."""
    system = FireSystem()
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click - add fire source
                    system.add_fire_source(event.pos[0], event.pos[1], width=80, intensity=1.0)
                elif event.button == 3:  # Right click - clear sources
                    system.clear_sources()

            elif event.type == pygame.MOUSEWHEEL:
                # Adjust intensity
                system.intensity += event.y * 0.2
                system.intensity = max(0.1, min(3.0, system.intensity))

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_c:
                    system.particles = []
                elif event.key == pygame.K_r:
                    system.clear_sources()
                    system.add_fire_source(WIDTH // 2, HEIGHT - 50, width=120, intensity=1.0)
                    system.wind = 0.0
                    system.target_wind = 0.0
                    system.intensity = 1.0
                elif event.key == pygame.K_w:
                    system.target_wind += 0.5
                elif event.key == pygame.K_q:
                    system.target_wind -= 0.5

        # Update
        if not paused:
            system.update()

        # Draw
        screen.fill(BLACK)

        # Draw a subtle ground
        ground_surf = pygame.Surface((WIDTH, 60), pygame.SRCALPHA)
        ground_color = (20, 15, 10, 255)
        pygame.draw.rect(ground_surf, ground_color, (0, 0, WIDTH, 60))
        screen.blit(ground_surf, (0, HEIGHT - 60))

        system.draw(screen)
        draw_ui(screen, system)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
