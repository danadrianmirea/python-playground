import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PLAYER_SIZE = 40
PLATFORM_HEIGHT = 20
PLATFORM_MIN_WIDTH = 100
PLATFORM_MAX_WIDTH = 200
GRAVITY = 0.8
JUMP_FORCE = -15
SCROLL_SPEED = 2
PLATFORM_SPACING = 100  # Reduced from 150 to ensure platforms are reachable
MAX_HORIZONTAL_DISTANCE = 200  # Maximum horizontal distance between platforms
MAX_JUMP_HEIGHT = abs(JUMP_FORCE * JUMP_FORCE / (2 * GRAVITY))  # Maximum height player can jump
PLATFORM_BUFFER = 5  # Number of platforms to generate in advance

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

class Player:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.velocity_y = 0
        self.velocity_x = 0
        self.is_jumping = False
        self.rect = pygame.Rect(self.x, self.y, PLAYER_SIZE, PLAYER_SIZE)
        self.speed = 5

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

    def jump(self):
        if not self.is_jumping:
            self.velocity_y = JUMP_FORCE
            self.is_jumping = True

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

class Platform:
    def __init__(self, x, y, width):
        self.x = x
        self.y = y
        self.width = width
        self.rect = pygame.Rect(x, y, width, PLATFORM_HEIGHT)

    def update(self):
        self.y += SCROLL_SPEED
        self.rect.y = self.y

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.rect)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Infinite Jumper")
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self):
        self.player = Player()
        self.platforms = []
        self.game_over = False
        self.score = 0
        self.font = pygame.font.Font(None, 36)

        # Create initial platform
        initial_platform = Platform(
            SCREEN_WIDTH // 2 - PLATFORM_MIN_WIDTH // 2,
            SCREEN_HEIGHT - 100,  # Start closer to bottom
            PLATFORM_MIN_WIDTH
        )
        self.platforms.append(initial_platform)
        
        # Generate initial buffer of platforms
        for _ in range(PLATFORM_BUFFER):
            self.generate_platform()

    def generate_platform(self):
        # Get the current highest platform
        current_platform = self.platforms[-1]
        
        # Calculate the maximum horizontal distance for the next platform
        min_x = max(0, current_platform.x - MAX_HORIZONTAL_DISTANCE)
        max_x = min(SCREEN_WIDTH - PLATFORM_MIN_WIDTH, 
                   current_platform.x + MAX_HORIZONTAL_DISTANCE)
        
        # Generate new platform position
        x = random.randint(int(min_x), int(max_x))
        width = random.randint(PLATFORM_MIN_WIDTH, PLATFORM_MAX_WIDTH)
        
        # Ensure the platform doesn't go off screen
        if x + width > SCREEN_WIDTH:
            x = SCREEN_WIDTH - width
        
        # Create the new platform at a fixed distance above the current highest platform
        new_platform = Platform(x, current_platform.y - PLATFORM_SPACING, width)
        self.platforms.append(new_platform)

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
            if event.type == pygame.KEYDOWN:
                # Jump with space, W, or up arrow
                if event.key in (pygame.K_SPACE, pygame.K_w, pygame.K_UP):
                    self.player.jump()

        # Update player
        self.player.update()

        # Check if player fell off screen
        if self.player.y > SCREEN_HEIGHT:
            self.game_over = True

        # Update platforms
        for platform in self.platforms[:]:
            platform.update()
            
            # Check collision
            self.player.check_collision(platform)
            
            # Remove platforms that are off screen
            if platform.y > SCREEN_HEIGHT:
                self.platforms.remove(platform)
                self.score += 1

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
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        if self.game_over:
            game_over_text = self.font.render("Game Over! Press R to restart", True, WHITE)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
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