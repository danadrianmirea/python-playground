"""
Extended Platform Jumper with scrolling, procedural level generation, and win condition.

Features:
- Camera scrolling that follows the player
- Procedurally generated levels with reachable platforms
- Exit/goal at the end of each level
- Win detection and new level generation
- Difficulty increases with each level
"""

import pygame
import random
import math

# Global constants

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GRAY = (100, 100, 100)

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Level generation constants
PLATFORM_MIN_WIDTH = 60
PLATFORM_MAX_WIDTH = 150
PLATFORM_HEIGHT = 20
MIN_VERTICAL_GAP = 40
MAX_VERTICAL_GAP = 120
MIN_HORIZONTAL_GAP = 80
MAX_HORIZONTAL_GAP = 200
LEVEL_LENGTH_MIN = 3000  # Minimum level width in pixels
LEVEL_LENGTH_MAX = 5000  # Maximum level width in pixels
EXIT_WIDTH = 40
EXIT_HEIGHT = 60
GROUND_HEIGHT = 20


class Player(pygame.sprite.Sprite):
    """ This class represents the bar at the bottom that the player
        controls. """

    # -- Methods
    def __init__(self):
        """ Constructor function """

        # Call the parent's constructor
        super().__init__()

        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        width = 40
        height = 60
        self.image = pygame.Surface([width, height])
        self.image.fill(RED)

        # Set a referance to the image rect.
        self.rect = self.image.get_rect()

        # Set speed vector of player
        self.change_x = 0
        self.change_y = 0

        # List of sprites we can bump against
        self.level = None

    def update(self):
        """ Move the player. """
        # Gravity
        self.calc_grav()

        # Move left/right
        self.rect.x += self.change_x

        # See if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:
            # If we are moving right,
            # set our right side to the left side of the item we hit
            if self.change_x > 0:
                self.rect.right = block.rect.left
            elif self.change_x < 0:
                # Otherwise if we are moving left, do the opposite.
                self.rect.left = block.rect.right

        # Move up/down
        self.rect.y += self.change_y

        # Check and see if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:

            # Reset our position based on the top/bottom of the object.
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom

            # Stop our vertical movement
            self.change_y = 0

    def calc_grav(self):
        """ Calculate effect of gravity. """
        if self.change_y == 0:
            self.change_y = 1
        else:
            self.change_y += .35

        # See if we are on the ground.
        if self.rect.y >= SCREEN_HEIGHT - self.rect.height and self.change_y >= 0:
            self.change_y = 0
            self.rect.y = SCREEN_HEIGHT - self.rect.height

    def jump(self):
        """ Called when user hits 'jump' button. """

        # move down a bit and see if there is a platform below us.
        # Move down 2 pixels because it doesn't work well if we only move down
        # 1 when working with a platform moving down.
        self.rect.y += 2
        platform_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        self.rect.y -= 2

        # If it is ok to jump, set our speed upwards
        if len(platform_hit_list) > 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.change_y = -10

    # Player-controlled movement:
    def go_left(self):
        """ Called when the user hits the left arrow. """
        self.change_x = -6

    def go_right(self):
        """ Called when the user hits the right arrow. """
        self.change_x = 6

    def stop(self):
        """ Called when the user lets off the keyboard. """
        self.change_x = 0


class Platform(pygame.sprite.Sprite):
    """ Platform the user can jump on """

    def __init__(self, width, height):
        """ Platform constructor. """
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill(GREEN)

        self.rect = self.image.get_rect()


class Exit(pygame.sprite.Sprite):
    """ Exit/goal that the player must reach to win the level """

    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface([EXIT_WIDTH, EXIT_HEIGHT])
        self.image.fill(YELLOW)
        # Draw a door-like shape
        pygame.draw.rect(self.image, ORANGE, [5, 5, EXIT_WIDTH - 10, EXIT_HEIGHT - 10])
        pygame.draw.circle(self.image, YELLOW, (EXIT_WIDTH // 2, EXIT_HEIGHT // 2), 8)

        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Level(object):
    """ A procedurally generated level with platforms and an exit. """

    def __init__(self, player, level_number=1):
        """ Constructor. """
        self.platform_list = pygame.sprite.Group()
        self.enemy_list = pygame.sprite.Group()
        self.exit_list = pygame.sprite.Group()
        self.player = player

        # Background image
        self.background = None

        # World dimensions
        self.world_shift_x = 0
        self.level_width = random.randint(LEVEL_LENGTH_MIN, LEVEL_LENGTH_MAX)

        # Generate the level
        self.generate_level(level_number)

    def generate_level(self, level_number):
        """ Procedurally generate a level with reachable platforms and an exit. """
        # Difficulty scaling
        difficulty = min(level_number, 10)  # Cap difficulty scaling
        max_vertical_gap = MAX_VERTICAL_GAP + difficulty * 5
        min_vertical_gap = MIN_VERTICAL_GAP + difficulty * 2
        max_horizontal_gap = MAX_HORIZONTAL_GAP - difficulty * 5
        min_horizontal_gap = MIN_HORIZONTAL_GAP

        # Ensure minimum gaps are reasonable
        min_vertical_gap = min(min_vertical_gap, 60)
        max_vertical_gap = min(max_vertical_gap, 150)

        # Player jump parameters (must match Player class)
        jump_velocity = -10
        gravity = 0.35
        # Maximum jump height: v^2 / (2 * g)
        max_jump_height = (jump_velocity ** 2) / (2 * gravity)
        # Maximum horizontal distance while jumping (at max jump height)
        # Time to reach max height: v / g
        time_to_apex = abs(jump_velocity) / gravity
        # Horizontal distance during jump at full speed
        max_horizontal_jump = 6 * (2 * time_to_apex)  # 6 is player speed

        # Start with a ground platform at the bottom
        ground_width = 300
        ground = Platform(ground_width, GROUND_HEIGHT)
        ground.rect.x = 0
        ground.rect.y = SCREEN_HEIGHT - GROUND_HEIGHT
        self.platform_list.add(ground)

        # Generate platforms from left to right
        current_x = ground_width
        last_platform_y = SCREEN_HEIGHT - GROUND_HEIGHT
        last_platform_width = ground_width
        last_platform_x = 0

        # Track the highest point we've been at to ensure we can always go back down
        highest_y = SCREEN_HEIGHT - GROUND_HEIGHT

        while current_x < self.level_width:
            # Decide platform width
            plat_width = random.randint(PLATFORM_MIN_WIDTH, PLATFORM_MAX_WIDTH)

            # Calculate horizontal gap
            h_gap = random.randint(min_horizontal_gap, max_horizontal_gap)
            plat_x = current_x + h_gap

            # Calculate vertical position
            # We want the platform to be reachable from the last platform
            # The player can jump up to max_jump_height pixels up
            # and can fall any distance without dying (they just land on the next platform)

            # Determine if we go up, down, or stay level
            direction = random.choice(['up', 'down', 'level'])

            if direction == 'up':
                # Go up, but not more than max_jump_height
                v_gap = random.randint(min_vertical_gap, min(max_vertical_gap, int(max_jump_height * 0.9)))
                plat_y = last_platform_y - v_gap
            elif direction == 'down':
                # Go down
                v_gap = random.randint(min_vertical_gap, max_vertical_gap)
                plat_y = last_platform_y + v_gap
            else:
                # Stay at similar level
                plat_y = last_platform_y + random.randint(-20, 20)

            # Clamp vertical position so platforms are within screen bounds
            plat_y = max(50, min(plat_y, SCREEN_HEIGHT - GROUND_HEIGHT - 50))

            # Check if this platform is reachable from the last one
            dx = plat_x - (last_platform_x + last_platform_width)
            dy = last_platform_y - plat_y  # positive means going up

            # If going up, verify it's reachable
            if dy > 0:
                # The player needs to be able to jump this high
                if dy > max_jump_height * 0.95:
                    # Too high, adjust downward
                    plat_y = last_platform_y - int(max_jump_height * 0.85)
                    plat_y = max(50, plat_y)
                    dy = last_platform_y - plat_y

                # Check if horizontal distance is also feasible
                # Time to reach this height: solve for t in dy = v*t - 0.5*g*t^2
                # dy = 10*t - 0.175*t^2  =>  0.175*t^2 - 10*t + dy = 0
                # t = (10 - sqrt(100 - 4*0.175*dy)) / (2*0.175)
                discriminant = jump_velocity ** 2 - 2 * gravity * dy
                if discriminant >= 0:
                    time_to_height = (abs(jump_velocity) - math.sqrt(discriminant)) / gravity
                    max_reach_x = 6 * time_to_height * 2  # *2 for both ascent and descent
                    if dx > max_reach_x * 1.2:
                        # Too far, move platform closer
                        plat_x = last_platform_x + last_platform_width + int(max_reach_x * 0.8)

            # Create the platform
            platform = Platform(plat_width, PLATFORM_HEIGHT)
            platform.rect.x = plat_x
            platform.rect.y = plat_y
            self.platform_list.add(platform)

            # Update tracking variables
            current_x = plat_x + plat_width
            last_platform_y = plat_y
            last_platform_width = plat_width
            last_platform_x = plat_x
            highest_y = min(highest_y, plat_y)

        # Place the exit at the end of the level
        # The exit should be on a platform or on the ground
        exit_x = current_x - 100  # A bit before the end
        exit_y = last_platform_y - EXIT_HEIGHT

        # Make sure exit is on solid ground
        exit_platform = Platform(EXIT_WIDTH + 40, PLATFORM_HEIGHT)
        exit_platform.rect.x = exit_x - 20
        exit_platform.rect.y = exit_y + EXIT_HEIGHT - PLATFORM_HEIGHT
        self.platform_list.add(exit_platform)

        # Create the exit sprite
        exit_sprite = Exit(exit_x, exit_y)
        self.exit_list.add(exit_sprite)

    def update(self):
        """ Update everything in this level."""
        self.platform_list.update()
        self.enemy_list.update()

    def draw(self, screen):
        """ Draw everything on this level. """

        # Draw the background
        screen.fill(BLUE)

        # Draw all the sprite lists that we have
        self.platform_list.draw(screen)
        self.enemy_list.draw(screen)
        self.exit_list.draw(screen)

    def shift_world(self, shift_x):
        """ Shift the world when the player moves near the edge. """
        self.world_shift_x += shift_x

        for platform in self.platform_list:
            platform.rect.x += shift_x

        for enemy in self.enemy_list:
            enemy.rect.x += shift_x

        for exit_sprite in self.exit_list:
            exit_sprite.rect.x += shift_x


def generate_new_level(player, current_level_no):
    """ Generate a new level and set up the player. """
    new_level = Level(player, current_level_no)
    player.level = new_level

    # Place player at the start of the level
    player.rect.x = 50
    player.rect.y = SCREEN_HEIGHT - player.rect.height - 100
    player.change_x = 0
    player.change_y = 0

    return new_level


def main():
    """ Main Program """
    pygame.init()

    # Set the height and width of the screen
    size = [SCREEN_WIDTH, SCREEN_HEIGHT]
    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("Platformer Jumper - Procedural Levels")

    # Create the player
    player = Player()

    # Create the first level
    current_level_no = 1
    current_level = generate_new_level(player, current_level_no)

    active_sprite_list = pygame.sprite.Group()
    active_sprite_list.add(player)

    # Font for UI
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # Win state
    won = False
    win_timer = 0

    # -------- Main Program Loop -----------
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    player.go_left()
                if event.key == pygame.K_RIGHT:
                    player.go_right()
                if event.key == pygame.K_UP:
                    player.jump()
                if event.key == pygame.K_r and won:
                    # Generate a new level when R is pressed after winning
                    current_level_no += 1
                    current_level = generate_new_level(player, current_level_no)
                    won = False

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT and player.change_x < 0:
                    player.stop()
                if event.key == pygame.K_RIGHT and player.change_x > 0:
                    player.stop()

        # Update the player.
        active_sprite_list.update()

        # Update items in the level
        current_level.update()

        # Camera scrolling: shift the world when player gets near edges
        # Right edge scrolling
        if player.rect.right >= SCREEN_WIDTH * 0.6:
            diff = player.rect.right - (SCREEN_WIDTH * 0.6)
            player.rect.right = SCREEN_WIDTH * 0.6
            current_level.shift_world(-diff)

        # Left edge scrolling (prevent scrolling back past start)
        if player.rect.left <= SCREEN_WIDTH * 0.2:
            diff = (SCREEN_WIDTH * 0.2) - player.rect.left
            player.rect.left = SCREEN_WIDTH * 0.2
            # Don't scroll past the beginning of the level
            if current_level.world_shift_x + diff <= 0:
                current_level.shift_world(diff)
            else:
                current_level.shift_world(-current_level.world_shift_x)

        # Keep player within screen bounds vertically
        if player.rect.top < 0:
            player.rect.top = 0
            player.change_y = 0

        # Check if player fell off the screen
        if player.rect.y > SCREEN_HEIGHT + 50:
            # Reset player to the last safe position (start of level)
            player.rect.x = 50
            player.rect.y = SCREEN_HEIGHT - player.rect.height - 100
            player.change_x = 0
            player.change_y = 0

        # Check if player reached the exit
        if not won:
            exit_hit_list = pygame.sprite.spritecollide(player, current_level.exit_list, False)
            if len(exit_hit_list) > 0:
                won = True
                win_timer = pygame.time.get_ticks()

        # ALL CODE TO DRAW SHOULD GO BELOW THIS COMMENT
        current_level.draw(screen)
        active_sprite_list.draw(screen)

        # Draw UI
        level_text = font.render(f"Level: {current_level_no}", True, WHITE)
        screen.blit(level_text, (10, 10))

        # Draw progress bar
        progress = min(1.0, abs(current_level.world_shift_x) / current_level.level_width)
        bar_width = 200
        bar_height = 15
        bar_x = SCREEN_WIDTH - bar_width - 20
        bar_y = 15
        pygame.draw.rect(screen, GRAY, [bar_x, bar_y, bar_width, bar_height])
        pygame.draw.rect(screen, GREEN, [bar_x, bar_y, int(bar_width * progress), bar_height])
        pygame.draw.rect(screen, WHITE, [bar_x, bar_y, bar_width, bar_height], 2)

        progress_text = small_font.render(f"{int(progress * 100)}%", True, WHITE)
        screen.blit(progress_text, (bar_x + bar_width // 2 - 15, bar_y + bar_height + 2))

        # Draw win message
        if won:
            win_text = font.render("LEVEL COMPLETE! Press R for next level", True, YELLOW)
            text_rect = win_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            # Draw a background box for the text
            pygame.draw.rect(screen, BLACK, [text_rect.x - 10, text_rect.y - 10,
                                              text_rect.width + 20, text_rect.height + 20])
            screen.blit(win_text, text_rect)

        # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT

        # Limit to 60 frames per second
        clock.tick(60)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

    # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
    pygame.quit()


if __name__ == "__main__":
    main()