import pygame
import math
import pymunk
import random

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Colors
GREEN = (34, 139, 34)  # Pool table green
BROWN = (139, 69, 19)  # Table border
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREEN_BALL = (0, 255, 0)
BROWN_BALL = (139, 69, 19)

# Ball properties
BALL_RADIUS = 15
BALL_MASS = 1.0
FRICTION = 0.3
ELASTICITY = 0.8

class Ball:
    def __init__(self, space, x, y, color, number=0, is_striped=False):
        self.color = color
        self.number = number
        self.is_striped = is_striped
        
        # Create pymunk body and shape
        moment = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS)
        self.body = pymunk.Body(BALL_MASS, moment)
        self.body.position = x, y
        
        self.shape = pymunk.Circle(self.body, BALL_RADIUS)
        self.shape.elasticity = ELASTICITY
        self.shape.friction = FRICTION
        
        space.add(self.body, self.shape)

    def draw(self, screen):
        # Get position from pymunk body
        x = int(self.body.position.x)
        y = int(self.body.position.y)
        
        # Draw the ball
        pygame.draw.circle(screen, self.color, (x, y), BALL_RADIUS)
        
        # Draw the number
        font = pygame.font.Font(None, 24)
        text = font.render(str(self.number), True, WHITE if self.color == BLACK else BLACK)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)
        
        # Draw stripe for striped balls
        if self.is_striped:
            stripe_width = 8
            stripe_height = BALL_RADIUS * 2
            stripe_rect = pygame.Rect(
                x - stripe_width//2,
                y - stripe_height//2,
                stripe_width,
                stripe_height
            )
            pygame.draw.rect(screen, WHITE, stripe_rect)

class PoolGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("2D Pool Game")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Create pymunk space
        self.space = pymunk.Space()
        self.space.damping = 0.9  # Add some drag to the space
        
        # Create table boundaries
        self.create_table_boundaries()
        
        self.balls = []
        self.cue_ball = None
        self.setup_balls()

    def create_table_boundaries(self):
        wall_thickness = 20
        walls = [
            # Left wall
            [(wall_thickness/2, WINDOW_HEIGHT/2), (wall_thickness, WINDOW_HEIGHT)],
            # Right wall
            [(WINDOW_WIDTH - wall_thickness/2, WINDOW_HEIGHT/2), (wall_thickness, WINDOW_HEIGHT)],
            # Top wall
            [(WINDOW_WIDTH/2, wall_thickness/2), (WINDOW_WIDTH, wall_thickness)],
            # Bottom wall
            [(WINDOW_WIDTH/2, WINDOW_HEIGHT - wall_thickness/2), (WINDOW_WIDTH, wall_thickness)]
        ]
        
        for pos, size in walls:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = pos
            shape = pymunk.Poly.create_box(body, size)
            shape.elasticity = 0.8
            shape.friction = 0.5
            self.space.add(body, shape)

    def setup_balls(self):
        # Create cue ball
        self.cue_ball = Ball(self.space, WINDOW_WIDTH // 4, WINDOW_HEIGHT // 2, WHITE)
        self.balls.append(self.cue_ball)

        # Ball colors for solid balls (1-7)
        solid_colors = [YELLOW, BLUE, RED, PURPLE, ORANGE, GREEN_BALL, BROWN_BALL]
        
        # Ball colors for striped balls (9-15)
        striped_colors = [YELLOW, BLUE, RED, PURPLE, ORANGE, GREEN_BALL, BROWN_BALL]
        
        # Setup the rack position
        start_x = WINDOW_WIDTH * 3 // 4
        start_y = WINDOW_HEIGHT // 2
        spacing = BALL_RADIUS * 2.2  # Slightly larger spacing to prevent initial overlap

        # Create the rack of balls
        rack = []
        
        # Add the 8-ball in the center
        rack.append(Ball(self.space, start_x, start_y, BLACK, 8))
        
        # Add solid balls (1-7)
        for i in range(7):
            row = i // 3
            col = i % 3
            x = start_x + (row * spacing)
            y = start_y + (col * spacing) - (row * spacing / 2)
            rack.append(Ball(self.space, x, y, solid_colors[i], i + 1))
        
        # Add striped balls (9-15)
        for i in range(7):
            row = (i + 7) // 3
            col = (i + 7) % 3
            x = start_x + (row * spacing)
            y = start_y + (col * spacing) - (row * spacing / 2)
            rack.append(Ball(self.space, x, y, striped_colors[i], i + 9, is_striped=True))
        
        # Sort balls by number
        rack.sort(key=lambda x: x.number)
        self.balls.extend(rack)

    def draw(self):
        self.screen.fill(GREEN)
        
        # Draw table border
        pygame.draw.rect(self.screen, BROWN, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 20)
        
        # Draw balls
        for ball in self.balls:
            ball.draw(self.screen)

        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle shooting
                    if event.button == 1:  # Left click
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        cue_pos = self.cue_ball.body.position
                        dx = mouse_x - cue_pos.x
                        dy = mouse_y - cue_pos.y
                        distance = math.sqrt(dx * dx + dy * dy)
                        if distance > 0:
                            # Apply impulse to cue ball
                            force = 5000
                            impulse = (dx / distance * force, dy / distance * force)
                            self.cue_ball.body.apply_impulse_at_local_point(impulse)

            # Update physics
            self.space.step(1/FPS)
            
            # Draw everything
            self.draw()
            
            # Cap the framerate
            self.clock.tick(FPS)

        pygame.quit()

if __name__ == "__main__":
    game = PoolGame()
    game.run() 