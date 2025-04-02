import pygame
import math
import pymunk
import random
import ctypes
import json
import os

# Initialize Pygame
pygame.init()

# Get the user's screen dimensions
user32 = ctypes.windll.user32
SCREEN_WIDTH = user32.GetSystemMetrics(0)  # Width
SCREEN_HEIGHT = user32.GetSystemMetrics(1)  # Height

# Calculate the maximum game dimensions while accounting for window decorations and taskbar
# Assuming taskbar is 40px and window decorations are 30px
TASKBAR_HEIGHT = 40
WINDOW_DECORATIONS = 30
MAX_GAME_HEIGHT = SCREEN_HEIGHT - TASKBAR_HEIGHT - WINDOW_DECORATIONS
MAX_GAME_WIDTH = SCREEN_WIDTH - WINDOW_DECORATIONS

# Base game dimensions (original size)
BASE_WIDTH = 800
BASE_HEIGHT = 600

# Calculate scaling factor while maintaining aspect ratio
width_scale = MAX_GAME_WIDTH / BASE_WIDTH
height_scale = MAX_GAME_HEIGHT / BASE_HEIGHT
SCALE_FACTOR = min(width_scale, height_scale) * 0.85  # Add 15% margin

# Scaled game dimensions
WINDOW_WIDTH = int(BASE_WIDTH * SCALE_FACTOR)

# Menu constants
MENU_HEIGHT = int(30 * SCALE_FACTOR)
SAVE_FILE = "pool_game_save.json"

# Scaled game dimensions (continued)
WINDOW_HEIGHT = int(BASE_HEIGHT * SCALE_FACTOR) + MENU_HEIGHT  # Add menu height to total window height

# Scale all game constants
BALL_RADIUS = int(15 * SCALE_FACTOR)
CUE_LENGTH = int(200 * SCALE_FACTOR)
CUE_WIDTH = int(8 * SCALE_FACTOR)
POWER_METER_WIDTH = int(200 * SCALE_FACTOR)
POWER_METER_HEIGHT = int(20 * SCALE_FACTOR)
WALL_THICKNESS = int(20 * SCALE_FACTOR)

FPS = 60
SPEED_UP_FACTOR = 8

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
GRAY = (128, 128, 128)

# Ball properties
BALL_MASS = 1.0
FRICTION = 0.5
ELASTICITY = 0.8

# Cue stick properties
CUE_COLOR = (139, 69, 19)  # Brown
CUE_TIP_COLOR = (255, 255, 255)  # White
POWER_METER_COLOR = (255, 0, 0)  # Red
POWER_METER_BG = (200, 200, 200)  # Gray

# Shot properties
MAX_SHOT_POWER = 2500
MIN_SHOT_POWER = 100

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
        font = pygame.font.Font(None, int(24 * SCALE_FACTOR))
        text = font.render(str(self.number), True, WHITE if self.color == BLACK else BLACK)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)
        
        # Draw stripe for striped balls
        if self.is_striped:
            stripe_width = int(8 * SCALE_FACTOR)
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
        # Create resizable window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
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
        
        # Speed control
        self.speed_multiplier = 1
        self.SPEED_UP_FACTOR = SPEED_UP_FACTOR
        
        # Add menu
        self.menu = Menu()

    def handle_resize(self, width, height):
        """Handle window resize events"""
        # Update window dimensions
        global WINDOW_WIDTH, WINDOW_HEIGHT
        WINDOW_WIDTH = width
        WINDOW_HEIGHT = height
        
        # Recreate table boundaries
        self.space = pymunk.Space()
        self.space.damping = 0.85
        self.create_table_boundaries()
        
        # Recreate balls
        self.balls = []
        self.setup_balls()

    def create_table_boundaries(self):
        wall_thickness = WALL_THICKNESS
        walls = [
            # Left wall
            [(wall_thickness/2, (WINDOW_HEIGHT + MENU_HEIGHT)/2), (wall_thickness, WINDOW_HEIGHT - MENU_HEIGHT)],
            # Right wall
            [(WINDOW_WIDTH - wall_thickness/2, (WINDOW_HEIGHT + MENU_HEIGHT)/2), (wall_thickness, WINDOW_HEIGHT - MENU_HEIGHT)],
            # Top wall
            [(WINDOW_WIDTH/2, wall_thickness/2 + MENU_HEIGHT), (WINDOW_WIDTH, wall_thickness)],
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
        
        # Calculate spacing - balls should be touching each other
        # The distance between ball centers should be exactly 2 * BALL_RADIUS
        spacing = BALL_RADIUS * 2

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

    def calculate_shot_direction(self, mouse_pos, cue_pos):
        """Calculate the normalized direction vector for a shot from the ball to the mouse."""
        # Calculate vector from ball to mouse
        dx = mouse_pos[0] - cue_pos.x
        dy = mouse_pos[1] - cue_pos.y
        
        # Calculate distance
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 0:
            # Normalize the direction vector
            dir_x = dx / distance
            dir_y = dy / distance
            return (dir_x, dir_y)
        return (0, 0)

    def draw_cue_stick(self, screen, mouse_pos):
        cue_pos = self.cue_ball.body.position
        dir_x, dir_y = self.calculate_shot_direction(mouse_pos, cue_pos)
        
        # Calculate the cue stick position (from behind the ball)
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

    def save_game(self):
        """Save the current game state to a file."""
        game_state = {
            'balls': [{
                'number': ball.number,
                'color': ball.color,
                'is_striped': ball.is_striped,
                'position': (ball.body.position.x, ball.body.position.y),
                'velocity': (ball.body.velocity.x, ball.body.velocity.y)
            } for ball in self.balls if ball != self.cue_ball],  # Exclude cue ball from regular balls
            'cue_ball': {
                'position': (self.cue_ball.body.position.x, self.cue_ball.body.position.y),
                'velocity': (self.cue_ball.body.velocity.x, self.cue_ball.body.velocity.y)
            }
        }
        
        try:
            with open(SAVE_FILE, 'w') as f:
                json.dump(game_state, f)
            print("Game saved successfully")
        except Exception as e:
            print(f"Error saving game: {e}")

    def load_game(self):
        """Load a saved game state from a file."""
        try:
            if not os.path.exists(SAVE_FILE):
                print("No save file found")
                return False
                
            with open(SAVE_FILE, 'r') as f:
                game_state = json.load(f)
                
            # Clear all balls from the space
            for ball in self.balls:
                self.space.remove(ball.body, ball.shape)
            self.balls.clear()
            
            # Load regular balls (excluding cue ball)
            for ball_data in game_state['balls']:
                # Skip the cue ball (white ball) as we'll load it separately
                if ball_data['color'] == WHITE:
                    continue
                ball = Ball(self.space, ball_data['position'][0], ball_data['position'][1],
                          ball_data['color'], ball_data['number'], ball_data['is_striped'])
                ball.body.velocity = ball_data['velocity']
                self.balls.append(ball)
                
            # Load cue ball
            cue_data = game_state['cue_ball']
            self.cue_ball = Ball(self.space, cue_data['position'][0], cue_data['position'][1], WHITE)
            self.cue_ball.body.velocity = cue_data['velocity']
            self.balls.append(self.cue_ball)
            
            print("Game loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading game: {e}")
            return False

    def draw(self):
        self.screen.fill(GREEN)
        
        # Draw table border
        pygame.draw.rect(self.screen, BROWN, (0, MENU_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT - MENU_HEIGHT), WALL_THICKNESS)
        
        # Draw balls
        for ball in self.balls:
            ball.draw(self.screen)
            
        # Only show targeting line and cue stick if balls are not moving
        if not self.are_balls_moving():
            # Draw targeting line (dashed)
            mouse_pos = pygame.mouse.get_pos()
            cue_pos = self.cue_ball.body.position
            dir_x, dir_y = self.calculate_shot_direction(mouse_pos, cue_pos)
            
            # Draw dashed line from ball to mouse
            dash_length = 10
            gap_length = 5
            current_x = cue_pos.x
            current_y = cue_pos.y
            line_length = 1000  # Long enough to reach any point on screen
            
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
        
        # Draw menu
        self.menu.draw(self.screen)

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

    def stop_all_balls(self):
        """Stop all balls by setting their velocities to zero."""
        print("\n" + "="*50)
        print("STOPPING ALL BALLS")
        print("="*50)
        for ball in self.balls:
            print(f"Ball {ball.number} velocity before stop: {ball.body.velocity}")
            ball.body.velocity = (0, 0)
            print(f"Ball {ball.number} velocity after stop: {ball.body.velocity}")
        print("="*50 + "\n")

    def reset_game(self):
        """Reset the game to its initial state"""
        # Clear all balls from the space
        for ball in self.balls:
            self.space.remove(ball.body, ball.shape)
        self.balls.clear()
        
        # Recreate the balls in their initial positions
        self.setup_balls()
        
        # Reset shooting mechanics
        self.aiming = False
        self.power = 0
        self.power_increasing = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    width, height = event.size
                    # Maintain aspect ratio
                    new_width = min(width, MAX_GAME_WIDTH)
                    new_height = min(height, MAX_GAME_HEIGHT)
                    self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                    self.handle_resize(new_width, new_height)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # First check for menu clicks
                    menu_result = self.menu.handle_click(event.pos)
                    if menu_result:
                        menu_name, item = menu_result
                        if menu_name == 'File':
                            if item == 'New Game':
                                self.reset_game()
                            elif item == 'Save Game':
                                self.save_game()
                            elif item == 'Load Game':
                                self.load_game()
                            elif item == 'Quit':
                                self.running = False
                        # Close the menu after selecting an option
                        self.menu.active_menu = None
                        continue
                    
                    # Check if clicking on menu area (including active menu)
                    if event.pos[1] < MENU_HEIGHT or (self.menu.active_menu and event.pos[1] < MENU_HEIGHT + len(self.menu.menu_items[self.menu.active_menu]) * (int(25 * SCALE_FACTOR) + 2 * self.menu.padding)):
                        continue
                        
                    # Start aiming only if not clicking in menu area
                    if event.button == 1:  # Left click
                        self.aiming = True
                        self.power = 0
                        self.power_increasing = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    # Check if clicking on menu area (including active menu)
                    if event.pos[1] < MENU_HEIGHT or (self.menu.active_menu and event.pos[1] < MENU_HEIGHT + len(self.menu.menu_items[self.menu.active_menu]) * (int(25 * SCALE_FACTOR) + 2 * self.menu.padding)):
                        continue
                        
                    # Shoot the ball only if not clicking in menu area
                    if event.button == 1:  # Left click
                        if self.aiming:
                            print("\n" + "="*50)
                            print("TAKING SHOT")
                            print("="*50)
                            
                            mouse_pos = pygame.mouse.get_pos()
                            cue_pos = self.cue_ball.body.position
                            dir_x, dir_y = self.calculate_shot_direction(mouse_pos, cue_pos)
                            
                            # Calculate force based on power (between MIN and MAX)
                            power_factor = self.power / self.max_power
                            force = MIN_SHOT_POWER + (MAX_SHOT_POWER - MIN_SHOT_POWER) * power_factor
                            
                            # Apply force in the direction from ball to mouse
                            impulse = (dir_x * force, dir_y * force)
                            print(f"\nShot Execution Debug:")
                            print(f"Power: {self.power:.1f}%")
                            print(f"Force: {force:.1f}")
                            print(f"Final impulse: ({impulse[0]:.1f}, {impulse[1]:.1f})")
                            print(f"Cue ball velocity before shot: {self.cue_ball.body.velocity}")
                            
                            # Apply impulse directly to the ball's center
                            self.cue_ball.body.apply_impulse_at_local_point(impulse, (0, 0))
                            
                            # Ensure the velocity follows the intended direction
                            current_velocity = self.cue_ball.body.velocity
                            speed = math.sqrt(current_velocity.x * current_velocity.x + current_velocity.y * current_velocity.y)
                            self.cue_ball.body.velocity = (dir_x * speed, dir_y * speed)
                            
                            # Add some damping to the velocity immediately after the shot
                            self.cue_ball.body.velocity *= 0.95
                            
                            print(f"Cue ball velocity after shot: {self.cue_ball.body.velocity}")
                            print("="*50 + "\n")
                        self.aiming = False
                        self.power = 0
                elif event.type == pygame.KEYDOWN:
                    # Speed up key
                    if event.key == pygame.K_s:
                        self.speed_multiplier = self.SPEED_UP_FACTOR
                elif event.type == pygame.KEYUP:
                    # Return to normal speed when S is released
                    if event.key == pygame.K_s:
                        self.speed_multiplier = 1

            # Update power meter while aiming
            self.update_power()

            # Update physics multiple times if speed up is active
            for _ in range(self.speed_multiplier):
                self.space.step(1/FPS)
            
            # Draw everything
            self.draw()
            
            # Cap the framerate (adjusted for speed multiplier)
            self.clock.tick(FPS * self.speed_multiplier)

        pygame.quit()

class Menu:
    def __init__(self):
        self.font = pygame.font.Font(None, int(24 * SCALE_FACTOR))
        self.menu_items = {
            'File': ['New Game', 'Save Game', 'Load Game', 'Quit']
        }
        self.active_menu = None
        self.menu_rects = {}
        self.item_rects = {}
        self.padding = int(10 * SCALE_FACTOR)  # Padding around menu items
        
    def draw(self, screen):
        # Draw menu bar background
        pygame.draw.rect(screen, GRAY, (0, 0, WINDOW_WIDTH, MENU_HEIGHT))
        
        # Draw menu items
        x = self.padding
        for menu_name in self.menu_items:
            text = self.font.render(menu_name, True, BLACK)
            text_rect = text.get_rect(topleft=(x, int(5 * SCALE_FACTOR)))
            
            # Draw menu item background with padding
            menu_rect = pygame.Rect(
                x - self.padding,
                0,
                text_rect.width + 2 * self.padding,
                MENU_HEIGHT
            )
            pygame.draw.rect(screen, GRAY, menu_rect)
            pygame.draw.rect(screen, BLACK, menu_rect, 1)  # Add border
            
            screen.blit(text, text_rect)
            self.menu_rects[menu_name] = menu_rect
            x += text_rect.width + 2 * self.padding + int(20 * SCALE_FACTOR)  # Add spacing between menu items
            
        # Draw active menu if any
        if self.active_menu:
            menu_y = MENU_HEIGHT
            # Calculate menu width based on longest item
            max_width = 0
            for item in self.menu_items[self.active_menu]:
                text = self.font.render(item, True, BLACK)
                max_width = max(max_width, text.get_width())
            
            # Add padding to width
            menu_width = max_width + 2 * self.padding
            menu_height = len(self.menu_items[self.active_menu]) * (int(25 * SCALE_FACTOR) + 2 * self.padding)
            
            # Draw menu background
            menu_rect = pygame.Rect(0, menu_y, menu_width, menu_height)
            pygame.draw.rect(screen, GRAY, menu_rect)
            pygame.draw.rect(screen, BLACK, menu_rect, 1)  # Add border
            
            # Draw menu items with padding
            item_y = menu_y + self.padding
            for item in self.menu_items[self.active_menu]:
                text = self.font.render(item, True, BLACK)
                text_rect = text.get_rect(topleft=(self.padding, item_y))
                
                # Draw item background with padding
                item_rect = pygame.Rect(
                    self.padding,
                    item_y - self.padding,
                    menu_width - 2 * self.padding,
                    int(25 * SCALE_FACTOR) + 2 * self.padding
                )
                pygame.draw.rect(screen, GRAY, item_rect)
                
                screen.blit(text, text_rect)
                self.item_rects[item] = item_rect
                item_y += int(25 * SCALE_FACTOR) + 2 * self.padding
                
    def handle_click(self, pos):
        x, y = pos
        
        # Check if clicking on menu bar
        if y < MENU_HEIGHT:
            for menu_name, rect in self.menu_rects.items():
                if rect.collidepoint(pos):
                    self.active_menu = menu_name if self.active_menu != menu_name else None
                    return None
                    
        # Check if clicking on menu items
        if self.active_menu and y >= MENU_HEIGHT:
            # Check if click is within menu bounds
            max_width = 0
            for item in self.menu_items[self.active_menu]:
                text = self.font.render(item, True, BLACK)
                max_width = max(max_width, text.get_width())
            
            menu_width = max_width + 2 * self.padding
            menu_height = len(self.menu_items[self.active_menu]) * (int(25 * SCALE_FACTOR) + 2 * self.padding)
            
            if x < menu_width and y < MENU_HEIGHT + menu_height:
                for item, rect in self.item_rects.items():
                    if rect.collidepoint(pos):
                        return (self.active_menu, item)
            else:
                # Clicked outside menu, close it
                self.active_menu = None
                    
        return None

if __name__ == "__main__":
    game = PoolGame()
    game.run() 