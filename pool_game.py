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
FRICTION = 0.7
ELASTICITY = 0.6

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
MAX_SHOT_POWER = 4000  # Maximum force that can be applied
MIN_SHOT_POWER = 500  # Minimum force that can be applied

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
        self.space.damping = 0.85
        
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
        
        # Standard 8-ball rack formation (rotated 180 degrees counterclockwise)
        # Column 1 (rightmost) - 5 balls
        for i in range(5):
            x = start_x + 2 * spacing
            y = start_y + (i - 2) * spacing
            if i == 0:
                rack.append(Ball(self.space, x, y, solid_colors[0], 1))  # Yellow
            elif i == 1:
                rack.append(Ball(self.space, x, y, striped_colors[0], 9, is_striped=True))  # Yellow stripe
            elif i == 2:
                rack.append(Ball(self.space, x, y, solid_colors[1], 2))  # Blue
            elif i == 3:
                rack.append(Ball(self.space, x, y, striped_colors[1], 10, is_striped=True))  # Blue stripe
            elif i == 4:
                rack.append(Ball(self.space, x, y, solid_colors[2], 3))  # Red

        # Column 2 - 4 balls
        for i in range(4):
            x = start_x + spacing
            y = start_y + (i - 1.5) * spacing
            if i == 0:
                rack.append(Ball(self.space, x, y, striped_colors[2], 11, is_striped=True))  # Red stripe
            elif i == 1:
                rack.append(Ball(self.space, x, y, solid_colors[3], 4))  # Purple
            elif i == 2:
                rack.append(Ball(self.space, x, y, striped_colors[3], 12, is_striped=True))  # Purple stripe
            elif i == 3:
                rack.append(Ball(self.space, x, y, solid_colors[4], 5))  # Orange

        # Column 3 (center) - 3 balls
        for i in range(3):
            x = start_x
            y = start_y + (i - 1) * spacing
            if i == 0:
                rack.append(Ball(self.space, x, y, striped_colors[4], 13, is_striped=True))  # Orange stripe
            elif i == 1:
                rack.append(Ball(self.space, x, y, BLACK, 8))  # 8-ball in center
            elif i == 2:
                rack.append(Ball(self.space, x, y, solid_colors[5], 6))  # Green

        # Column 4 - 2 balls
        for i in range(2):
            x = start_x - spacing
            y = start_y + (i - 0.5) * spacing
            if i == 0:
                rack.append(Ball(self.space, x, y, striped_colors[5], 14, is_striped=True))  # Green stripe
            elif i == 1:
                rack.append(Ball(self.space, x, y, solid_colors[6], 7))  # Brown

        # Column 5 (leftmost) - 1 ball
        x = start_x - 2 * spacing
        y = start_y
        rack.append(Ball(self.space, x, y, striped_colors[6], 15, is_striped=True))  # Brown stripe
        
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

    def are_balls_moving(self):
        # Threshold for considering a ball "stopped" (in pixels per second)
        VELOCITY_THRESHOLD = 10
        any_ball_moving = False
        
        for ball in self.balls:
            velocity = ball.body.velocity
            speed = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y)
            if speed > VELOCITY_THRESHOLD:
                any_ball_moving = True
            else:
                # Set velocity to 0 if below threshold
                ball.body.velocity = (0, 0)
        
        return any_ball_moving

    def draw(self):
        self.screen.fill(GREEN)
        
        # Draw table border
        pygame.draw.rect(self.screen, BROWN, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 20)
        
        # Draw balls
        for ball in self.balls:
            ball.draw(self.screen)
            
        # Only show targeting line and cue stick if balls are not moving
        if not self.are_balls_moving():
            # Draw targeting line (dashed)
            mouse_pos = pygame.mouse.get_pos()
            cue_pos = self.cue_ball.body.position
            
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
        else:
            font = pygame.font.Font(None, 36)
            text = font.render("Wait for balls to stop...", True, WHITE)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100))
            self.screen.blit(text, text_rect)
            
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