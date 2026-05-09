import pygame
import pygame.sndarray
import random
import sys
import math
import time
import numpy as np

# Initialize Pygame
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Constants
WIDTH, HEIGHT = 800, 800
FPS = 60
CENTER = (WIDTH // 2, HEIGHT // 2)
RADIUS = 300

# Colors
BLACK = (20, 20, 20)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (40, 40, 40)

# Simon colors (bright)
RED = (255, 50, 50)
GREEN = (50, 220, 50)
BLUE = (50, 100, 255)
YELLOW = (255, 255, 50)

# Dimmed versions (when not lit)
RED_DIM = (100, 20, 20)
GREEN_DIM = (20, 100, 20)
BLUE_DIM = (20, 40, 120)
YELLOW_DIM = (100, 100, 20)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simon Says")
clock = pygame.time.Clock()
font_large = pygame.font.SysFont("Arial", 48, bold=True)
font_medium = pygame.font.SysFont("Arial", 32)
font_small = pygame.font.SysFont("Arial", 24)

# Sound generation using pygame's synth
def generate_tone(frequency, duration=0.3, volume=0.2):
    """Generate a sine wave tone and return a pygame Sound object."""
    sample_rate = 22050
    n_samples = int(sample_rate * duration)
    # Create stereo array (2 channels) as int16
    t = np.arange(n_samples, dtype=np.float64)
    samples = (volume * 32767 * np.sin(2 * np.pi * frequency * t / sample_rate)).astype(np.int16)
    stereo = np.column_stack((samples, samples))
    buf = pygame.sndarray.make_sound(stereo)
    return buf

def generate_chord(frequencies, duration=0.5, volume=0.15):
    """Generate a pleasant chord from a list of frequencies."""
    sample_rate = 22050
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples, dtype=np.float64)
    # Mix multiple frequencies
    wave = np.zeros(n_samples, dtype=np.float64)
    for freq in frequencies:
        wave += np.sin(2 * np.pi * freq * t / sample_rate)
    wave /= len(frequencies)  # Normalize
    samples = (volume * 32767 * wave).astype(np.int16)
    stereo = np.column_stack((samples, samples))
    return pygame.sndarray.make_sound(stereo)

def generate_buzzer(frequency=100, duration=0.3, volume=0.3):
    """Generate a harsh buzzer sound for losing."""
    sample_rate = 22050
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples, dtype=np.float64)
    # Square wave with some noise for harshness
    wave = np.sign(np.sin(2 * np.pi * frequency * t / sample_rate))
    # Add a bit of noise
    noise = np.random.uniform(-0.3, 0.3, n_samples)
    wave = wave * 0.7 + noise * 0.3
    samples = (volume * 32767 * wave).astype(np.int16)
    stereo = np.column_stack((samples, samples))
    return pygame.sndarray.make_sound(stereo)

# Create sounds for each color
SOUND_RED = generate_tone(220, 0.3)    # Low A
SOUND_GREEN = generate_tone(330, 0.3)  # E
SOUND_BLUE = generate_tone(440, 0.3)   # A
SOUND_YELLOW = generate_tone(550, 0.3) # C#

SOUNDS = {
    'red': SOUND_RED,
    'green': SOUND_GREEN,
    'blue': SOUND_BLUE,
    'yellow': SOUND_YELLOW,
}

# Win sound: ascending major chord (C-E-G-C)
SOUND_WIN = generate_chord([523, 659, 784, 1047], 0.6, 0.15)
# Lose sound: low buzzer
SOUND_LOSE = generate_buzzer(80, 0.5, 0.3)

# Game state
COLORS = ['red', 'green', 'blue', 'yellow']
COLOR_MAP = {
    'red': (RED, RED_DIM),
    'green': (GREEN, GREEN_DIM),
    'blue': (BLUE, BLUE_DIM),
    'yellow': (YELLOW, YELLOW_DIM),
}

# Quadrant angles (in radians)
# Red: top-right, Green: top-left, Blue: bottom-right, Yellow: bottom-left
QUADRANT_ANGLES = {
    'red': (0, math.pi / 2),
    'green': (math.pi / 2, math.pi),
    'blue': (3 * math.pi / 2, 2 * math.pi),
    'yellow': (math.pi, 3 * math.pi / 2),
}

def draw_quadrant(screen, color_name, lit=False):
    """Draw one quadrant of the circle."""
    bright, dim = COLOR_MAP[color_name]
    color = bright if lit else dim
    start_angle, end_angle = QUADRANT_ANGLES[color_name]
    
    # Draw the pie slice
    points = [CENTER]
    num_segments = 50
    for i in range(num_segments + 1):
        angle = start_angle + (end_angle - start_angle) * i / num_segments
        x = CENTER[0] + RADIUS * math.cos(angle)
        y = CENTER[1] - RADIUS * math.sin(angle)
        points.append((x, y))
    
    if len(points) >= 3:
        pygame.draw.polygon(screen, color, points)
    
    # Draw border
    for i in range(num_segments):
        angle1 = start_angle + (end_angle - start_angle) * i / num_segments
        angle2 = start_angle + (end_angle - start_angle) * (i + 1) / num_segments
        x1 = CENTER[0] + RADIUS * math.cos(angle1)
        y1 = CENTER[1] - RADIUS * math.sin(angle1)
        x2 = CENTER[0] + RADIUS * math.cos(angle2)
        y2 = CENTER[1] - RADIUS * math.sin(angle2)
        pygame.draw.line(screen, WHITE, (x1, y1), (x2, y2), 2)
    
    # Draw radial lines
    for angle in [start_angle, end_angle]:
        x = CENTER[0] + RADIUS * math.cos(angle)
        y = CENTER[1] - RADIUS * math.sin(angle)
        pygame.draw.line(screen, WHITE, CENTER, (x, y), 3)

def draw_simon(screen, lit_color=None):
    """Draw the full Simon board."""
    # Draw background
    screen.fill(BLACK)
    
    # Draw outer ring
    pygame.draw.circle(screen, DARK_GRAY, CENTER, RADIUS + 10)
    pygame.draw.circle(screen, WHITE, CENTER, RADIUS + 10, 3)
    
    # Draw quadrants
    for color_name in COLORS:
        draw_quadrant(screen, color_name, lit=(lit_color == 'all' or color_name == lit_color))
    
    # Draw center circle
    pygame.draw.circle(screen, DARK_GRAY, CENTER, 60)
    pygame.draw.circle(screen, WHITE, CENTER, 60, 3)
    
    # Draw center text
    text = font_large.render("SIMON", True, WHITE)
    text_rect = text.get_rect(center=CENTER)
    screen.blit(text, text_rect)

def draw_text_centered(screen, text, font, color, y_offset=0):
    """Draw text centered on screen."""
    text_surf = font.render(text, True, color)
    text_rect = text_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + y_offset))
    screen.blit(text_surf, text_rect)

def draw_text_with_border(screen, text, font, color, y_offset=0, border_color=WHITE, bg_color=(0, 0, 0, 200), padding=10):
    """Draw text centered on screen with a bordered rectangular background."""
    text_surf = font.render(text, True, color)
    text_rect = text_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + y_offset))
    
    # Create background rect with padding
    bg_rect = text_rect.inflate(padding * 2, padding * 2)
    
    # Draw semi-transparent background
    bg_surf = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
    bg_surf.fill(bg_color)
    screen.blit(bg_surf, bg_rect)
    
    # Draw border
    pygame.draw.rect(screen, border_color, bg_rect, 2, border_radius=4)
    
    # Draw text
    screen.blit(text_surf, text_rect)

def flash_color(color_name, duration=0.3):
    """Flash a color and play its sound."""
    draw_simon(screen, lit_color=color_name)
    pygame.display.flip()
    SOUNDS[color_name].play()
    time.sleep(duration)
    draw_simon(screen)
    pygame.display.flip()
    time.sleep(0.1)

def show_sequence(sequence):
    """Show the sequence to the player."""
    for color_name in sequence:
        flash_color(color_name)
        pygame.event.pump()  # Keep window responsive

def get_clicked_color(pos):
    """Determine which color was clicked based on mouse position."""
    x, y = pos
    dx = x - CENTER[0]
    dy = CENTER[1] - y  # Flip y for math coordinates
    
    # Check if inside the circle
    distance = math.sqrt(dx**2 + dy**2)
    if distance > RADIUS:
        return None
    
    # Calculate angle
    angle = math.atan2(dy, dx)
    if angle < 0:
        angle += 2 * math.pi
    
    # Determine quadrant
    for color_name, (start, end) in QUADRANT_ANGLES.items():
        if start <= angle < end:
            return color_name
    
    return None

def show_game_over(score):
    """Display game over screen."""
    draw_simon(screen)
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))
    
    texts = [
        (font_large, "GAME OVER", WHITE, -60),
        (font_medium, f"Score: {score}", WHITE, 0),
        (font_small, "Press SPACE to play again", GRAY, 50),
        (font_small, "Press ESC to quit", GRAY, 90),
    ]
    
    for font, text, color, y_offset in texts:
        draw_text_with_border(screen, text, font, color, y_offset)
    
    pygame.display.flip()

def show_start_screen():
    """Display the start screen."""
    screen.fill(BLACK)
    
    # Draw a dimmed Simon
    draw_simon(screen)
    
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 100))
    screen.blit(overlay, (0, 0))
    
    texts = [
        (font_large, "SIMON SAYS", WHITE, -80),
        (font_small, "Watch the sequence, then repeat it!", GRAY, -20),
        (font_small, "Click the colors in the correct order", GRAY, 10),
        (font_medium, "Press SPACE to start", WHITE, 60),
        (font_small, "Press ESC to quit", GRAY, 110),
    ]
    
    for font, text, color, y_offset in texts:
        draw_text_with_border(screen, text, font, color, y_offset)
    
    pygame.display.flip()

def main():
    """Main game loop."""
    sequence = []
    player_sequence = []
    score = 0
    game_state = "start"  # start, showing, input, pause, game_over
    input_index = 0
    show_timer = 0
    current_show_index = 0
    pause_until = 0
    pause_message = ""
    
    running = True
    while running:
        clock.tick(FPS)
        now = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if game_state == "start":
                        # Start new game
                        sequence = []
                        player_sequence = []
                        score = 0
                        input_index = 0
                        # Generate first color
                        sequence.append(random.choice(COLORS))
                        game_state = "showing"
                        current_show_index = 0
                        show_timer = pygame.time.get_ticks()
                    elif game_state == "game_over":
                        # Restart
                        game_state = "start"
            
            elif event.type == pygame.MOUSEBUTTONDOWN and game_state == "input":
                color = get_clicked_color(event.pos)
                if color:
                    # Flash the clicked color
                    flash_color(color, 0.15)
                    
                    # Check if correct
                    if color == sequence[input_index]:
                        input_index += 1
                        if input_index >= len(sequence):
                            # Sequence complete! Play win sound and pause
                            SOUND_WIN.play()
                            # Flash the whole board (all quadrants lit)
                            for _ in range(3):
                                draw_simon(screen, lit_color='all')
                                pygame.display.flip()
                                time.sleep(0.15)
                                draw_simon(screen)
                                pygame.display.flip()
                                time.sleep(0.1)
                            
                            score = len(sequence)
                            sequence.append(random.choice(COLORS))
                            pause_message = "Well done!"
                            game_state = "pause"
                            pause_until = pygame.time.get_ticks() + 1000
                            current_show_index = 0
                            input_index = 0
                            show_timer = 0
                    else:
                        # Wrong! Play lose sound and pause
                        SOUND_LOSE.play()
                        # Flash the whole board red (all quadrants lit red)
                        for _ in range(3):
                            draw_simon(screen, lit_color='all')
                            pygame.display.flip()
                            time.sleep(0.15)
                            draw_simon(screen)
                            pygame.display.flip()
                            time.sleep(0.1)
                        pause_message = "Game over!"
                        game_state = "pause"
                        pause_until = pygame.time.get_ticks() + 1000
        
        # Handle showing sequence
        if game_state == "showing":
            if current_show_index < len(sequence):
                if show_timer == 0:
                    show_timer = now
                
                # Show each color for 0.5s with 0.2s gap
                color_duration = 500
                gap_duration = 200
                elapsed = now - show_timer
                color_show_time = color_duration + gap_duration
                
                if elapsed < color_duration:
                    # Show lit
                    draw_simon(screen, lit_color=sequence[current_show_index])
                    # Play sound once at start of this color's display
                    if elapsed < 50:  # First 50ms
                        SOUNDS[sequence[current_show_index]].play()
                    pygame.display.flip()
                elif elapsed < color_show_time:
                    # Gap - show unlit
                    draw_simon(screen)
                    pygame.display.flip()
                else:
                    # Move to next color
                    current_show_index += 1
                    show_timer = now
            else:
                # Done showing sequence
                game_state = "input"
                input_index = 0
                draw_simon(screen)
                pygame.display.flip()
        
        elif game_state == "input":
            # Show the Simon board with a "Your turn" indicator
            draw_simon(screen)
            draw_text_with_border(screen, f"Your turn! ({input_index + 1}/{len(sequence)})", font_small, GRAY, y_offset=HEIGHT // 2 - 50, border_color=GRAY)
            pygame.display.flip()
        
        elif game_state == "pause":
            # Show the board and wait for the pause to end
            draw_simon(screen)
            # Show the pause message centered on screen
            if pause_until > now:
                # Draw a semi-transparent overlay
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 120))
                screen.blit(overlay, (0, 0))
                # Show the message with bordered background
                msg_color = GREEN if pause_message == "Well done!" else RED
                draw_text_with_border(screen, pause_message, font_large, msg_color, y_offset=0, border_color=msg_color)
            pygame.display.flip()
            if now >= pause_until:
                if pause_message == "Game over!":
                    # Reset the game back to start screen
                    sequence = []
                    score = 0
                    game_state = "start"
                else:
                    # Continue to next round
                    game_state = "showing"
                    show_timer = now
        
        elif game_state == "start":
            show_start_screen()
        
        elif game_state == "game_over":
            show_game_over(score)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()