import pygame
import random
import sys
import ctypes

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
SCALE_FACTOR = min(width_scale, height_scale)

# Scaled game dimensions
GAME_WIDTH = int(BASE_WIDTH * SCALE_FACTOR)
GAME_HEIGHT = int(BASE_HEIGHT * SCALE_FACTOR)

# Scale all game constants
PLAYER_SIZE = int(40 * SCALE_FACTOR)
PLATFORM_HEIGHT = int(20 * SCALE_FACTOR)
PLATFORM_MIN_WIDTH = int(100 * SCALE_FACTOR)
PLATFORM_MAX_WIDTH = int(200 * SCALE_FACTOR)
GRAVITY = 0.8 * SCALE_FACTOR
JUMP_FORCE = -15 * SCALE_FACTOR
BASE_SCROLL_SPEED = 1 * SCALE_FACTOR
MAX_SCROLL_SPEED_INC = 5 * SCALE_FACTOR
MAX_SCROLL_SPEED = BASE_SCROLL_SPEED + MAX_SCROLL_SPEED_INC
SCROLL_SPEED_INCREASE_RATE = 0.1 * SCALE_FACTOR
SCROLL_SPEED_DECREASE_RATE = 0.05 * SCALE_FACTOR
PLATFORM_SPACING = int(100 * SCALE_FACTOR)
MAX_HORIZONTAL_DISTANCE = int(200 * SCALE_FACTOR)
MAX_JUMP_HEIGHT = abs(JUMP_FORCE * JUMP_FORCE / (2 * GRAVITY))
PLATFORM_BUFFER = 10
COIN_SIZE = int(15 * SCALE_FACTOR)
SPEED_INCREASE_THRESHOLD = 100
SPEED_INCREASE_AMOUNT = 1 * SCALE_FACTOR

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

COIN_REWARD = 5


# Load sound effects
try:
    COIN_SOUND = pygame.mixer.Sound("assets/coin.mp3")
    COIN_SOUND.set_volume(0.1)
    JUMP_SOUND = pygame.mixer.Sound("assets/jump.mp3")
    JUMP_SOUND.set_volume(0.1)
except:
    print("Warning: Could not load sound files")

class Player:
    def __init__(self, start_x, start_y):
        self.x = start_x
        self.y = start_y
        self.velocity_y = 0
        self.velocity_x = 0
        self.is_jumping = False
        self.rect = pygame.Rect(self.x, self.y, PLAYER_SIZE, PLAYER_SIZE)
        self.speed = 5
        self.jump_cooldown = 0  # Add jump cooldown timer
        self.jump_cooldown_time = 10  # Frames to wait between jumps

    def update(self):
        # Apply gravity
        self.velocity_y += GRAVITY
        self.y += self.velocity_y

        # Handle horizontal movement (both WASD and arrow keys)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.x -= self.speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.x += self.speed

        # Screen wrapping
        if self.x < -PLAYER_SIZE:
            self.x = SCREEN_WIDTH
        elif self.x > SCREEN_WIDTH:
            self.x = -PLAYER_SIZE

        # Update rectangle position
        self.rect.x = self.x
        self.rect.y = self.y

        # Update jump cooldown
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1

    def jump(self):
        if not self.is_jumping and self.jump_cooldown == 0:
            self.velocity_y = JUMP_FORCE
            self.is_jumping = True
            self.jump_cooldown = self.jump_cooldown_time
            try:
                JUMP_SOUND.play()
            except:
                pass  # Ignore if sound couldn't be loaded

    def check_collision(self, platform):
        if self.rect.colliderect(platform.rect):
            if self.velocity_y > 0:  # Falling
                self.rect.bottom = platform.rect.top
                self.y = self.rect.y
                self.velocity_y = 0
                self.is_jumping = False
                return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, BLUE, self.rect)

class Coin:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, COIN_SIZE, COIN_SIZE)
        self.collected = False

    def update(self, scroll_speed):
        self.y += scroll_speed
        self.rect.y = self.y

    def draw(self, screen):
        if not self.collected:
            pygame.draw.circle(screen, YELLOW, (self.x + COIN_SIZE//2, self.y + COIN_SIZE//2), COIN_SIZE//2)

class Platform:
    def __init__(self, x, y, width):
        self.x = x
        self.y = y
        self.width = width
        self.rect = pygame.Rect(x, y, width, PLATFORM_HEIGHT)
        # Randomly decide if this platform has a coin
        self.coin = Coin(x + random.randint(0, width - COIN_SIZE), y - COIN_SIZE) if random.random() < 0.3 else None

    def update(self, scroll_speed):
        self.y += scroll_speed
        self.rect.y = self.y
        if self.coin:
            self.coin.update(scroll_speed)

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.rect)
        if self.coin:
            self.coin.draw(screen)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Infinite Jumper")
        self.clock = pygame.time.Clock()
        self.scroll_speed = BASE_SCROLL_SPEED
        self.current_level = 1
        self.reset_game()

    def reset_game(self):
        # Create initial platform exactly in the center of the screen
        initial_platform = Platform(
            (GAME_WIDTH - PLATFORM_MIN_WIDTH) // 2,  # Center horizontally
            GAME_HEIGHT - GAME_WIDTH*0.33,  # Start closer to bottom
            PLATFORM_MIN_WIDTH
        )
        
        # Initialize player on top of the initial platform
        player_start_x = initial_platform.x + (initial_platform.width - PLAYER_SIZE) // 2
        player_start_y = initial_platform.y - PLAYER_SIZE
        self.player = Player(player_start_x, player_start_y)
        
        self.platforms = [initial_platform]
        self.game_over = False
        self.score = 0
        self.coins_collected = 0
        self.current_level = 1
        self.font = pygame.font.Font(None, int(36 * SCALE_FACTOR))
        
        # Generate initial buffer of platforms
        for _ in range(PLATFORM_BUFFER):
            self.generate_platform()

    def generate_platform(self):
        # Get the current highest platform
        current_platform = self.platforms[-1]
        
        # Calculate the maximum horizontal distance for the next platform
        min_x = max(0, current_platform.x - MAX_HORIZONTAL_DISTANCE)
        max_x = min(GAME_WIDTH - PLATFORM_MIN_WIDTH, 
                   current_platform.x + MAX_HORIZONTAL_DISTANCE)
        
        # Generate new platform position
        x = random.randint(int(min_x), int(max_x))
        width = random.randint(PLATFORM_MIN_WIDTH, PLATFORM_MAX_WIDTH)
        
        # Ensure the platform doesn't go off screen
        if x + width > GAME_WIDTH:
            x = GAME_WIDTH - width
        
        # Create the new platform at a fixed distance above the current highest platform
        new_platform = Platform(x, current_platform.y - PLATFORM_SPACING, width)
        self.platforms.append(new_platform)

    def check_coin_collision(self):
        for platform in self.platforms:
            if platform.coin and not platform.coin.collected:
                if self.player.rect.colliderect(platform.coin.rect):
                    platform.coin.collected = True
                    self.coins_collected += 1
                    self.score += COIN_REWARD
                    try:
                        COIN_SOUND.play()
                    except:
                        pass  # Ignore if sound couldn't be loaded

    def update_scroll_speed(self):
        # Calculate how far up the screen the player is (0 to 1)
        # 0 means bottom of screen, 1 means top of screen
        screen_position = 1 - (self.player.y / SCREEN_HEIGHT)
        
        # Calculate target speed based on screen position
        # Linear interpolation between BASE_SCROLL_SPEED and MAX_SCROLL_SPEED
        target_speed = BASE_SCROLL_SPEED + (MAX_SCROLL_SPEED - BASE_SCROLL_SPEED) * screen_position
        
        # Smoothly adjust current scroll speed towards target speed
        if self.scroll_speed < target_speed:
            self.scroll_speed = min(self.scroll_speed + SCROLL_SPEED_INCREASE_RATE, target_speed)
        else:
            self.scroll_speed = max(self.scroll_speed - SCROLL_SPEED_DECREASE_RATE, target_speed)

    def update(self):
        if self.game_over:
            # Handle events for game over state
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
            return

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                width, height = event.size
                # Maintain aspect ratio
                new_width = min(width, MAX_GAME_WIDTH)
                new_height = min(height, MAX_GAME_HEIGHT)
                self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)

        # Check for held jump keys (space, W, or up arrow)
        keys = pygame.key.get_pressed()
        if (keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]) and not self.player.is_jumping:
            self.player.jump()

        # Increase base scroll speed when score reaches threshold
        if self.score >= SPEED_INCREASE_THRESHOLD*self.current_level:
            global BASE_SCROLL_SPEED
            BASE_SCROLL_SPEED += SPEED_INCREASE_AMOUNT
            MAX_SCROLL_SPEED = BASE_SCROLL_SPEED + MAX_SCROLL_SPEED_INC
            self.current_level += 1  # Increment level when speed increases

        # Update scroll speed based on player position
        self.update_scroll_speed()

        # Update player
        self.player.update()

        # Check if player fell off screen
        if self.player.y > SCREEN_HEIGHT:
            self.game_over = True

        # Update platforms
        for platform in self.platforms[:]:
            platform.update(self.scroll_speed)  # Pass current scroll speed
            
            # Check collision
            self.player.check_collision(platform)
            
            # Remove platforms that are off screen
            if platform.y > SCREEN_HEIGHT:
                self.platforms.remove(platform)
                self.score += 1

        # Check for coin collection
        self.check_coin_collision()

        # Generate new platforms when we have fewer than PLATFORM_BUFFER platforms above the screen
        visible_platforms = [p for p in self.platforms if p.y > 0]
        if len(visible_platforms) < PLATFORM_BUFFER:
            self.generate_platform()

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw platforms
        for platform in self.platforms:
            platform.draw(self.screen)
        
        # Draw player
        self.player.draw(self.screen)
        
        # Draw score, coins, and level
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        coins_text = self.font.render(f"Coins: {self.coins_collected}", True, YELLOW)
        level_text = self.font.render(f"Level: {self.current_level}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(coins_text, (10, 50))
        self.screen.blit(level_text, (10, 90))  # Add level display below coins
        
        if self.game_over:
            game_over_text = self.font.render("Game Over! Press R to restart", True, WHITE)
            text_rect = game_over_text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()

    def run(self):
        while True:
            self.update()
            self.draw()
            self.clock.tick(60)

if __name__ == "__main__":
    game = Game()
    game.run() 