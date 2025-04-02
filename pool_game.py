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

# Cue stick properties
CUE_LENGTH = 200
CUE_WIDTH = 8
CUE_COLOR = (139, 69, 19)  # Brown
CUE_TIP_COLOR = (255, 255, 255)  # White
POWER_METER_WIDTH = 200
POWER_METER_HEIGHT = 20
POWER_METER_COLOR = (255, 0, 0)  # Red
POWER_METER_BG = (200, 200, 200)  # Gray

# Shot properties
MAX_SHOT_POWER = 8000  # Maximum force that can be applied
MIN_SHOT_POWER = 1000  # Minimum force that can be applied

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
        
        # Shooting mechanics
        self.aiming = False
        self.power = 0
        self.power_increasing = True
        self.max_power = 100
        self.power_speed = 2

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

    def draw_cue_stick(self, screen, mouse_pos):
        cue_pos = self.cue_ball.body.position
        dx = mouse_pos[0] - cue_pos.x
        dy = mouse_pos[1] - cue_pos.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 0:
            # Calculate the direction vector
            dir_x = dx / distance
            dir_y = dy / distance
            
            # Calculate the cue stick position
            cue_start_x = cue_pos.x - dir_x * (CUE_LENGTH + BALL_RADIUS)
            cue_start_y = cue_pos.y - dir_y * (CUE_LENGTH + BALL_RADIUS)
            cue_end_x = cue_pos.x - dir_x * BALL_RADIUS
            cue_end_y = cue_pos.y - dir_y * BALL_RADIUS
            
            # Draw the cue stick
            pygame.draw.line(screen, CUE_COLOR, 
                           (int(cue_start_x), int(cue_start_y)),
                           (int(cue_end_x), int(cue_end_y)),
                           CUE_WIDTH)
            
            # Draw the cue tip
            pygame.draw.circle(screen, CUE_TIP_COLOR,
                             (int(cue_end_x), int(cue_end_y)),
                             CUE_WIDTH // 2)

    def draw_power_meter(self, screen):
        # Draw power meter background
        meter_x = WINDOW_WIDTH // 2 - POWER_METER_WIDTH // 2
        meter_y = WINDOW_HEIGHT - 40
        pygame.draw.rect(screen, POWER_METER_BG,
                        (meter_x, meter_y, POWER_METER_WIDTH, POWER_METER_HEIGHT))
        
        # Draw power level
        power_width = int(POWER_METER_WIDTH * (self.power / self.max_power))
        pygame.draw.rect(screen, POWER_METER_COLOR,
                        (meter_x, meter_y, power_width, POWER_METER_HEIGHT))
        
        # Draw power meter border
        pygame.draw.rect(screen, BLACK,
                        (meter_x, meter_y, POWER_METER_WIDTH, POWER_METER_HEIGHT), 2)
        
        # Draw power percentage text
        font = pygame.font.Font(None, 24)
        text = font.render(f"{int(self.power)}%", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, meter_y - 20))
        screen.blit(text, text_rect)

    def draw(self):
        self.screen.fill(GREEN)
        
        # Draw table border
        pygame.draw.rect(self.screen, BROWN, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 20)
        
        # Draw balls
        for ball in self.balls:
            ball.draw(self.screen)
            
        # Draw targeting line and cue stick
        mouse_pos = pygame.mouse.get_pos()
        cue_pos = self.cue_ball.body.position
        
        # Draw targeting line (dashed)
        dx = mouse_pos[0] - cue_pos.x
        dy = mouse_pos[1] - cue_pos.y
        distance = math.sqrt(dx * dx + dy * dy)
        if distance > 0:
            dir_x = dx / distance
            dir_y = dy / distance
            line_length = 1000  # Long enough to reach any point on screen
            end_x = cue_pos.x + dir_x * line_length
            end_y = cue_pos.y + dir_y * line_length
            
            # Draw dashed line
            dash_length = 10
            gap_length = 5
            current_x = cue_pos.x
            current_y = cue_pos.y
            while math.sqrt((current_x - cue_pos.x)**2 + (current_y - cue_pos.y)**2) < line_length:
                # Draw dash
                pygame.draw.line(self.screen, (255, 255, 255, 128),
                               (int(current_x), int(current_y)),
                               (int(current_x + dir_x * dash_length),
                                int(current_y + dir_y * dash_length)), 1)
                # Move to next dash position
                current_x += dir_x * (dash_length + gap_length)
                current_y += dir_y * (dash_length + gap_length)
        
        self.draw_cue_stick(self.screen, mouse_pos)
            
        # Draw power meter
        self.draw_power_meter(self.screen)

        pygame.display.flip()

    def update_power(self):
        if not self.aiming:
            return
            
        if self.power_increasing:
            self.power += self.power_speed
            if self.power >= self.max_power:
                self.power = self.max_power
                self.power_increasing = False
        else:
            self.power -= self.power_speed
            if self.power <= 0:
                self.power = 0
                self.power_increasing = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Start aiming
                    if event.button == 1:  # Left click
                        self.aiming = True
                        self.power = 0
                        self.power_increasing = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    # Shoot the ball
                    if event.button == 1:  # Left click
                        if self.aiming:
                            mouse_x, mouse_y = pygame.mouse.get_pos()
                            cue_pos = self.cue_ball.body.position
                            dx = mouse_x - cue_pos.x
                            dy = mouse_y - cue_pos.y
                            distance = math.sqrt(dx * dx + dy * dy)
                            if distance > 0:
                                # Calculate force based on power (between MIN and MAX)
                                power_factor = self.power / self.max_power
                                force = MIN_SHOT_POWER + (MAX_SHOT_POWER - MIN_SHOT_POWER) * power_factor
                                impulse = (dx / distance * force, dy / distance * force)
                                self.cue_ball.body.apply_impulse_at_local_point(impulse)
                        self.aiming = False
                        self.power = 0

            # Update power meter while aiming
            self.update_power()

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