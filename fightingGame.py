"""
fightingGame.py - Mortal Kombat / Golden Axe style 2-player fighting game
Proof of concept with hack-and-slash beat 'em up gameplay.
"""

import pygame
import sys
import math
import random

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 600
FPS = 60
GROUND_Y = 480

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 30, 30)
DARK_RED = (100, 10, 10)
BLUE = (30, 30, 200)
DARK_BLUE = (10, 10, 100)
GREEN = (30, 200, 30)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (180, 30, 180)
GRAY = (100, 100, 100)
DARK_GRAY = (40, 40, 40)
SKY_BLUE = (20, 20, 60)
BLOOD_RED = (180, 0, 0)

# Game states
MENU = 0
FIGHTING = 1
ROUND_END = 2
GAME_OVER = 3

# Fighter states
IDLE = 0
WALKING = 1
ATTACKING = 2
HIT = 3
BLOCKING = 4
SPECIAL = 5
DEAD = 6

# Attack types
LIGHT_ATTACK = 0
HEAVY_ATTACK = 1
SPECIAL_ATTACK = 2


class Particle:
    """Simple particle for visual effects."""
    def __init__(self, x, y, vx, vy, color, lifetime, size=4):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.3  # gravity
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, screen):
        alpha = self.lifetime / self.max_lifetime
        size = int(self.size * alpha)
        if size > 0:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), size)


class Fighter:
    """Base fighter class with Mortal Kombat-style moves."""
    def __init__(self, x, y, facing_right, name, color_scheme):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.width = 50
        self.height = 90
        self.facing_right = facing_right
        self.name = name
        self.color_scheme = color_scheme  # (primary, dark, accent)

        # Stats
        self.max_hp = 100
        self.hp = self.max_hp
        self.speed = 5
        self.jump_power = -12
        self.gravity = 0.6

        # State
        self.state = IDLE
        self.state_timer = 0
        self.attack_cooldown = 0
        self.combo_count = 0
        self.combo_timer = 0
        self.on_ground = True
        self.blocking = False
        self.invulnerable = 0

        # Attack properties
        self.light_damage = 8
        self.heavy_damage = 18
        self.special_damage = 30
        self.light_range = 60
        self.heavy_range = 70
        self.special_range = 90

        # Current attack type being performed
        self.current_attack = None

        # Animation
        self.anim_frame = 0
        self.anim_timer = 0
        self.walk_cycle = 0

        # Visual
        self.hit_flash = 0
        self.attack_effect = None  # current attack visual effect

    def get_rect(self):
        return pygame.Rect(self.x - self.width // 2, self.y - self.height,
                           self.width, self.height)

    def get_attack_rect(self, attack_type):
        """Get the hitbox for the current attack."""
        range_val = {LIGHT_ATTACK: self.light_range,
                     HEAVY_ATTACK: self.heavy_range,
                     SPECIAL_ATTACK: self.special_range}[attack_type]
        if self.facing_right:
            return pygame.Rect(self.x + 10, self.y - self.height + 10,
                               range_val, self.height - 20)
        else:
            return pygame.Rect(self.x - 10 - range_val, self.y - self.height + 10,
                               range_val, self.height - 20)

    def take_damage(self, damage, knockback_x):
        if self.invulnerable > 0:
            return False
        if self.blocking:
            damage = damage // 4
            knockback_x //= 3
            self.hit_flash = 8
            self.hp -= damage
            self.vx = knockback_x
            return True
        self.hp -= damage
        self.hit_flash = 12
        self.state = HIT
        self.state_timer = 15
        self.vx = knockback_x
        self.vy = -5
        self.combo_count = 0
        return True

    def update(self, other_fighter):
        """Update fighter state each frame."""
        # Timers
        if self.state_timer > 0:
            self.state_timer -= 1
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if self.hit_flash > 0:
            self.hit_flash -= 1
        if self.invulnerable > 0:
            self.invulnerable -= 1
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            self.combo_count = 0

        # Animation
        self.anim_timer += 1
        if self.anim_timer > 8:
            self.anim_timer = 0
            self.anim_frame = (self.anim_frame + 1) % 4

        # Physics
        self.vy += self.gravity
        self.x += self.vx
        self.y += self.vy

        # Friction
        if self.state != ATTACKING and self.state != SPECIAL:
            self.vx *= 0.85
        else:
            self.vx *= 0.9

        # Ground collision
        if self.y >= GROUND_Y:
            self.y = GROUND_Y
            self.vy = 0
            self.on_ground = True
        else:
            self.on_ground = False

        # Screen bounds
        self.x = max(self.width // 2, min(SCREEN_WIDTH - self.width // 2, self.x))

        # State transitions
        if self.state == HIT and self.state_timer <= 0:
            self.state = IDLE
        if self.state == ATTACKING and self.state_timer <= 0:
            self.state = IDLE
        if self.state == SPECIAL and self.state_timer <= 0:
            self.state = IDLE

        # Walk cycle
        if self.state == WALKING:
            self.walk_cycle += 0.1

    def light_attack(self):
        """Quick jab attack. Max ~2 per second."""
        if self.state != IDLE and self.state != WALKING:
            return None
        if self.attack_cooldown > 0:
            return None
        self.state = ATTACKING
        self.state_timer = 20
        self.attack_cooldown = 30  # 0.5s at 60fps
        self.anim_frame = 0
        self.combo_count += 1
        self.combo_timer = 30
        self.current_attack = LIGHT_ATTACK
        return LIGHT_ATTACK

    def heavy_attack(self):
        """Slow heavy attack with more damage. Max ~1.2 per second."""
        if self.state != IDLE and self.state != WALKING:
            return None
        if self.attack_cooldown > 0:
            return None
        self.state = ATTACKING
        self.state_timer = 28
        self.attack_cooldown = 50  # ~0.83s at 60fps
        self.anim_frame = 0
        self.combo_count += 1
        self.combo_timer = 30
        self.current_attack = HEAVY_ATTACK
        return HEAVY_ATTACK

    def special_attack(self):
        """Special move - a powerful strike. Max ~0.8 per second."""
        if self.state != IDLE and self.state != WALKING:
            return None
        if self.attack_cooldown > 0:
            return None
        self.state = SPECIAL
        self.state_timer = 35
        self.attack_cooldown = 75  # ~1.25s at 60fps
        self.anim_frame = 0
        self.combo_count += 1
        self.combo_timer = 30
        self.current_attack = SPECIAL_ATTACK
        return SPECIAL_ATTACK

    def jump(self):
        if self.on_ground and self.state != HIT:
            self.vy = self.jump_power
            self.on_ground = False

    def start_block(self):
        if self.state == IDLE or self.state == WALKING:
            self.blocking = True
            self.state = BLOCKING

    def stop_block(self):
        self.blocking = False
        if self.state == BLOCKING:
            self.state = IDLE

    def draw(self, screen):
        """Draw the fighter with Mortal Kombat-style pixel art."""
        primary, dark, accent = self.color_scheme
        color = primary
        if self.hit_flash > 0 and self.hit_flash % 4 < 2:
            color = WHITE

        # Shadow
        shadow_rect = pygame.Rect(self.x - 25, GROUND_Y - 5, 50, 10)
        pygame.draw.ellipse(screen, (0, 0, 0, 128), shadow_rect)

        body_y = self.y - self.height

        # === BODY ===
        # Torso
        if self.state == ATTACKING or self.state == SPECIAL:
            # Lean forward during attack
            offset = 15 if self.facing_right else -15
            torso_rect = pygame.Rect(self.x - 18 + offset // 2, body_y + 25, 36, 35)
        else:
            torso_rect = pygame.Rect(self.x - 18, body_y + 25, 36, 35)
        pygame.draw.rect(screen, color, torso_rect)
        pygame.draw.rect(screen, dark, torso_rect, 2)

        # Head
        head_center = (self.x, body_y + 15)
        head_radius = 14
        pygame.draw.circle(screen, (255, 200, 150), head_center, head_radius)
        pygame.draw.circle(screen, dark, head_center, head_radius, 2)

        # Eyes
        eye_offset = 5 if self.facing_right else -5
        eye_y = body_y + 12
        pygame.draw.circle(screen, BLACK, (self.x + eye_offset - 2, eye_y), 2)
        pygame.draw.circle(screen, BLACK, (self.x + eye_offset + 4, eye_y), 2)

        # Mouth (determined expression)
        if self.state == ATTACKING or self.state == SPECIAL:
            pygame.draw.arc(screen, RED,
                            (self.x - 6, body_y + 16, 12, 8), 0, math.pi, 2)

        # === LEGS ===
        leg_offset = 0
        if self.state == WALKING:
            leg_offset = int(math.sin(self.walk_cycle * 2) * 8)
        elif self.state == ATTACKING:
            leg_offset = 5 if self.facing_right else -5

        # Left leg
        leg_x = self.x - 10 + leg_offset
        pygame.draw.line(screen, dark, (self.x - 8, body_y + 60),
                         (leg_x, body_y + 90), 6)
        # Right leg
        pygame.draw.line(screen, dark, (self.x + 8, body_y + 60),
                         (self.x + 10 - leg_offset, body_y + 90), 6)

        # Feet
        foot_dir = 8 if self.facing_right else -8
        pygame.draw.ellipse(screen, dark,
                            (self.x - 14 + leg_offset, body_y + 86, 16, 8))
        pygame.draw.ellipse(screen, dark,
                            (self.x + 4 - leg_offset, body_y + 86, 16, 8))

        # === ARMS ===
        arm_y = body_y + 30

        if self.state == ATTACKING or self.state == SPECIAL:
            # Attack pose - arm extended forward
            reach = 35 if self.facing_right else -35
            # Back arm
            pygame.draw.line(screen, (255, 200, 150),
                             (self.x - 15, arm_y),
                             (self.x - 25, arm_y + 15), 5)
            # Front arm (attacking)
            pygame.draw.line(screen, (255, 200, 150),
                             (self.x + 15, arm_y),
                             (self.x + 15 + reach, arm_y - 5), 6)

            # Weapon / fist effect
            if self.state == SPECIAL:
                # Energy blade / fist glow
                glow_x = self.x + 15 + reach
                glow_size = 18
                for i in range(3):
                    s = glow_size - i * 4
                    alpha = 100 - i * 30
                    glow_color = (accent[0], accent[1], accent[2], alpha)
                    pygame.draw.circle(screen, accent,
                                       (glow_x, arm_y - 5), s, 2)
                pygame.draw.circle(screen, YELLOW, (glow_x, arm_y - 5), 8)
            else:
                # Fist
                fist_x = self.x + 15 + reach
                pygame.draw.circle(screen, (255, 200, 150),
                                   (fist_x, arm_y - 5), 7)
        elif self.state == BLOCKING:
            # Block pose - arms crossed
            pygame.draw.line(screen, (255, 200, 150),
                             (self.x - 15, arm_y),
                             (self.x, arm_y - 10), 5)
            pygame.draw.line(screen, (255, 200, 150),
                             (self.x + 15, arm_y),
                             (self.x, arm_y - 10), 5)
        elif self.state == HIT:
            # Hit reaction - arms flailing
            pygame.draw.line(screen, (255, 200, 150),
                             (self.x - 15, arm_y),
                             (self.x - 30, arm_y - 10), 5)
            pygame.draw.line(screen, (255, 200, 150),
                             (self.x + 15, arm_y),
                             (self.x + 30, arm_y + 10), 5)
        else:
            # Idle pose
            pygame.draw.line(screen, (255, 200, 150),
                             (self.x - 15, arm_y),
                             (self.x - 25, arm_y + 15), 5)
            pygame.draw.line(screen, (255, 200, 150),
                             (self.x + 15, arm_y),
                             (self.x + 25, arm_y + 15), 5)

        # === HEALTH BAR ===
        bar_width = 180
        bar_height = 16
        bar_x = 20 if not self.facing_right else SCREEN_WIDTH - 20 - bar_width
        bar_y = 20

        # Name
        font = pygame.font.Font(None, 20)
        name_text = font.render(self.name, True, WHITE)
        name_rect = name_text.get_rect()
        if self.facing_right:
            name_rect.topright = (bar_x + bar_width, bar_y - 2)
        else:
            name_rect.topleft = (bar_x, bar_y - 2)
        screen.blit(name_text, name_rect)

        # HP bar background
        pygame.draw.rect(screen, DARK_RED, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(screen, GRAY, (bar_x, bar_y, bar_width, bar_height), 2)

        # HP bar fill
        hp_ratio = max(0, self.hp / self.max_hp)
        fill_width = int(bar_width * hp_ratio)
        hp_color = GREEN
        if hp_ratio < 0.3:
            hp_color = RED
        elif hp_ratio < 0.6:
            hp_color = ORANGE
        pygame.draw.rect(screen, hp_color, (bar_x, bar_y, fill_width, bar_height))

        # HP text
        hp_text = font.render(f"{max(0, int(self.hp))}/{self.max_hp}", True, WHITE)
        hp_text_rect = hp_text.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
        screen.blit(hp_text, hp_text_rect)

        # Combo counter
        if self.combo_count > 1:
            combo_font = pygame.font.Font(None, 36)
            combo_text = combo_font.render(f"{self.combo_count} HIT COMBO!", True, YELLOW)
            combo_rect = combo_text.get_rect(center=(self.x, self.y - self.height - 30))
            screen.blit(combo_text, combo_rect)

        # Block indicator
        if self.blocking:
            block_font = pygame.font.Font(None, 24)
            block_text = block_font.render("BLOCKING", True, BLUE)
            block_rect = block_text.get_rect(center=(self.x, self.y - self.height - 10))
            screen.blit(block_text, block_rect)


class FightingGame:
    """Main game class."""
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Mortal Kombat: Python Edition")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 28)

        # Game state
        self.state = MENU
        self.round = 1
        self.round_timer = 0
        self.round_fight_shown = 0
        self.winner = None
        self.particles = []
        self.p1_rounds_won = 0
        self.p2_rounds_won = 0

        # Create fighters
        self.player1 = Fighter(250, GROUND_Y, True, "SCORPION",
                                (ORANGE, (180, 100, 0), YELLOW))
        self.player2 = Fighter(774, GROUND_Y, False, "SUB-ZERO",
                                (BLUE, DARK_BLUE, (100, 150, 255)))

        # Background
        self.bg_scroll = 0
        self.bg_elements = self._create_background()

        # Screen shake
        self.screen_shake = 0

        # Sound effects (simple beeps since no audio files)
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

    def _create_background(self):
        """Create background elements for Mortal Kombat-style arena."""
        elements = []
        # Ground tiles
        for x in range(0, SCREEN_WIDTH, 64):
            elements.append(('ground', x))
        # Pillars / columns
        for x in [50, 974]:
            elements.append(('pillar', x))
        # Moon
        elements.append(('moon', 512, 80))
        # Torches
        for x in [150, 874]:
            elements.append(('torch', x))
        return elements

    def add_particles(self, x, y, color, count=15, speed=5):
        """Spawn particle effects."""
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            vx = math.cos(angle) * random.uniform(1, speed)
            vy = math.sin(angle) * random.uniform(1, speed) - 2
            lifetime = random.randint(15, 30)
            size = random.randint(2, 6)
            self.particles.append(Particle(x, y, vx, vy, color, lifetime, size))

    def handle_collision(self, attacker, defender, attack_type):
        """Check if attack hits and apply damage."""
        attack_rect = attacker.get_attack_rect(attack_type)
        defend_rect = defender.get_rect()

        if attack_rect.colliderect(defend_rect):
            # Calculate damage
            if attack_type == LIGHT_ATTACK:
                damage = attacker.light_damage
                knockback = 8 if attacker.facing_right else -8
                if attacker.combo_count >= 3:
                    damage = int(damage * 1.3)
            elif attack_type == HEAVY_ATTACK:
                damage = attacker.heavy_damage
                knockback = 14 if attacker.facing_right else -14
                if attacker.combo_count >= 3:
                    damage = int(damage * 1.3)
            else:  # SPECIAL
                damage = attacker.special_damage
                knockback = 20 if attacker.facing_right else -20

            # Apply damage
            hit_landed = defender.take_damage(damage, knockback)

            if hit_landed:
                # Visual feedback
                self.screen_shake = 6
                hit_x = (attacker.x + defender.x) // 2
                hit_y = defender.y - defender.height // 2

                if defender.blocking:
                    self.add_particles(hit_x, hit_y, BLUE, 8, 3)
                else:
                    self.add_particles(hit_x, hit_y, RED, 20, 7)
                    self.add_particles(hit_x, hit_y, BLOOD_RED, 10, 4)

                # Check death - track round wins (best of 3)
                if defender.hp <= 0:
                    defender.hp = 0
                    defender.state = DEAD
                    self.winner = attacker
                    # Increment round wins for the attacker
                    if attacker == self.player1:
                        self.p1_rounds_won += 1
                    else:
                        self.p2_rounds_won += 1
                    # If someone won 2 out of 3, go straight to game over
                    if self.p1_rounds_won >= 2 or self.p2_rounds_won >= 2:
                        self.state = GAME_OVER
                    else:
                        self.state = ROUND_END
                        self.round_timer = 150  # 2.5 seconds

                return True
        return False

    def reset_round(self):
        """Reset fighters for a new round."""
        self.player1.x = 250
        self.player1.y = GROUND_Y
        self.player1.vx = 0
        self.player1.vy = 0
        self.player1.state = IDLE
        self.player1.hp = self.player1.max_hp
        self.player1.blocking = False
        self.player1.combo_count = 0
        self.player1.combo_timer = 0

        self.player2.x = 774
        self.player2.y = GROUND_Y
        self.player2.vx = 0
        self.player2.vy = 0
        self.player2.state = IDLE
        self.player2.hp = self.player2.max_hp
        self.player2.blocking = False
        self.player2.combo_count = 0
        self.player2.combo_timer = 0

        self.particles.clear()
        self.winner = None
        self.round_fight_shown = 60  # Show "FIGHT!" for 1 second

    def draw_background(self, screen):
        """Draw the Mortal Kombat-style arena background."""
        # Sky gradient
        for y in range(0, GROUND_Y):
            r = int(10 + (y / GROUND_Y) * 20)
            g = int(10 + (y / GROUND_Y) * 15)
            b = int(40 + (y / GROUND_Y) * 30)
            pygame.draw.line(screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))

        # Ground
        pygame.draw.rect(screen, (60, 40, 20), (0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y))
        pygame.draw.rect(screen, (80, 55, 25), (0, GROUND_Y, SCREEN_WIDTH, 5))

        # Ground line pattern
        for x in range(0, SCREEN_WIDTH, 32):
            pygame.draw.line(screen, (50, 35, 15), (x, GROUND_Y), (x + 16, GROUND_Y), 2)

        # Moon
        moon_x, moon_y = 512, 80
        pygame.draw.circle(screen, (200, 200, 180), (moon_x, moon_y), 40)
        pygame.draw.circle(screen, (180, 180, 160), (moon_x, moon_y), 38)
        # Moon crater details
        for cx, cy, cr in [(moon_x - 10, moon_y - 5, 8),
                           (moon_x + 12, moon_y + 8, 6),
                           (moon_x - 5, moon_y + 15, 5)]:
            pygame.draw.circle(screen, (160, 160, 140), (cx, cy), cr)

        # Pillars
        for px in [50, 974]:
            # Pillar body
            pillar_rect = pygame.Rect(px - 15, 50, 30, GROUND_Y - 50)
            pygame.draw.rect(screen, (80, 70, 60), pillar_rect)
            pygame.draw.rect(screen, (100, 90, 80), pillar_rect, 3)
            # Pillar top
            top_rect = pygame.Rect(px - 25, 40, 50, 20)
            pygame.draw.rect(screen, (90, 80, 70), top_rect)
            pygame.draw.rect(screen, (110, 100, 90), top_rect, 2)
            # Skull on top
            skull_x, skull_y = px, 35
            pygame.draw.circle(screen, (200, 200, 200), (skull_x, skull_y), 10)
            pygame.draw.circle(screen, (180, 180, 180), (skull_x, skull_y), 8)
            # Skull eyes
            pygame.draw.circle(screen, BLACK, (skull_x - 3, skull_y - 2), 2)
            pygame.draw.circle(screen, BLACK, (skull_x + 3, skull_y - 2), 2)

        # Torches with animated fire
        for tx in [150, 874]:
            # Pole
            pygame.draw.line(screen, (80, 70, 60), (tx, GROUND_Y - 20), (tx, GROUND_Y - 120), 4)
            # Fire
            fire_y = GROUND_Y - 130
            flicker = random.randint(-3, 3)
            # Outer flame
            pygame.draw.ellipse(screen, (255, 100, 0),
                                (tx - 8 + flicker, fire_y - 15, 16, 25))
            # Inner flame
            pygame.draw.ellipse(screen, (255, 200, 50),
                                (tx - 4 + flicker, fire_y - 10, 8, 18))
            # Core
            pygame.draw.ellipse(screen, (255, 255, 200),
                                (tx - 2 + flicker, fire_y - 5, 4, 10))
            # Glow
            for i in range(3):
                glow_size = 30 + i * 15
                glow_alpha = 30 - i * 10
                glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (255, 100, 0, glow_alpha),
                                   (glow_size, glow_size), glow_size)
                screen.blit(glow_surf, (tx - glow_size, fire_y - glow_size))

    def draw_hud(self, screen):
        """Draw round indicator and other HUD elements."""
        # Round indicator
        if self.round_fight_shown > 0:
            alpha = min(255, self.round_fight_shown * 4)
            if self.round_fight_shown > 40:
                text = f"ROUND {self.round}"
                color = YELLOW
            elif self.round_fight_shown > 20:
                text = "FIGHT!"
                color = RED
            else:
                text = ""
                color = WHITE

            if text:
                text_surf = self.font_large.render(text, True, color)
                text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, 120))
                # Shadow
                shadow_surf = self.font_large.render(text, True, BLACK)
                shadow_rect = shadow_surf.get_rect(center=(SCREEN_WIDTH // 2 + 3, 123))
                screen.blit(shadow_surf, shadow_rect)
                screen.blit(text_surf, text_rect)

        # VS text
        vs_font = pygame.font.Font(None, 36)
        vs_text = vs_font.render("VS", True, YELLOW)
        vs_rect = vs_text.get_rect(center=(SCREEN_WIDTH // 2, 28))
        screen.blit(vs_text, vs_rect)

        # Controls hint
        if self.state == MENU:
            hint_font = pygame.font.Font(None, 24)
            hints = [
                "PLAYER 1 (SCORPION):  WASD - Move/Jump,  F - Light,  G - Heavy,  H - Special,  D - Block",
                "PLAYER 2 (SUB-ZERO):  Arrows - Move/Jump/Block,  INS - Light,  HOME - Heavy,  END - Special",
                "",
                "PRESS SPACE TO START"
            ]
            for i, hint in enumerate(hints):
                hint_surf = hint_font.render(hint, True, WHITE)
                hint_rect = hint_surf.get_rect(center=(SCREEN_WIDTH // 2, 500 + i * 25))
                screen.blit(hint_surf, hint_rect)

    def draw_round_end(self, screen):
        """Draw round end message between rounds."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(120)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        if self.winner:
            if self.winner == self.player1:
                player_label = "PLAYER 1"
            else:
                player_label = "PLAYER 2"
            text = f"{player_label} WINS THE ROUND!"
            color = self.winner.color_scheme[0]
            text_surf = self.font_large.render(text, True, color)
            text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, 200))
            shadow_surf = self.font_large.render(text, True, BLACK)
            shadow_rect = shadow_surf.get_rect(center=(SCREEN_WIDTH // 2 + 3, 203))
            screen.blit(shadow_surf, shadow_rect)
            screen.blit(text_surf, text_rect)

        # Show current score
        score_text = self.font_medium.render(
            f"SCORE: {self.p1_rounds_won} - {self.p2_rounds_won}", True, YELLOW)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 270))
        screen.blit(score_text, score_rect)

    def draw_game_over(self, screen):
        """Draw game over screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        if self.winner:
            # Determine player number
            if self.winner == self.player1:
                player_label = "PLAYER 1"
            else:
                player_label = "PLAYER 2"
            text = f"{player_label} WINS!"
            color = self.winner.color_scheme[0]
            text_surf = self.font_large.render(text, True, color)
            text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, 200))
            shadow_surf = self.font_large.render(text, True, BLACK)
            shadow_rect = shadow_surf.get_rect(center=(SCREEN_WIDTH // 2 + 4, 204))
            screen.blit(shadow_surf, shadow_rect)
            screen.blit(text_surf, text_rect)

        # Round score
        score_text = self.font_medium.render(
            f"{self.p1_rounds_won} - {self.p2_rounds_won}", True, YELLOW)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 280))
        screen.blit(score_text, score_rect)

        # Restart prompt
        restart_text = self.font_small.render("Press SPACE to play again", True, WHITE)
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, 350))
        screen.blit(restart_text, restart_rect)

    def run(self):
        """Main game loop."""
        running = True
        # Track which attack keys were just pressed this frame (KEYDOWN events)
        p1_attack_keys = set()
        p2_attack_keys = set()
        p1_jump_pressed = False
        p2_jump_pressed = False

        while running:
            # Reset per-frame key press tracking
            p1_attack_keys.clear()
            p2_attack_keys.clear()
            p1_jump_pressed = False
            p2_jump_pressed = False

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        if self.state == MENU:
                            self.state = FIGHTING
                            self.reset_round()
                        elif self.state == GAME_OVER:
                            self.state = MENU
                            self.round = 1
                            self.p1_rounds_won = 0
                            self.p2_rounds_won = 0
                            self.reset_round()
                    # Player 1 attack keys (KEYDOWN only - no auto-repeat)
                    if event.key == pygame.K_f:
                        p1_attack_keys.add('light')
                    elif event.key == pygame.K_g:
                        p1_attack_keys.add('heavy')
                    elif event.key == pygame.K_h:
                        p1_attack_keys.add('special')
                    elif event.key == pygame.K_w:
                        p1_jump_pressed = True
                    # Player 2 attack keys (INS/HOME/END)
                    if event.key == pygame.K_INSERT:
                        p2_attack_keys.add('light')
                    elif event.key == pygame.K_HOME:
                        p2_attack_keys.add('heavy')
                    elif event.key == pygame.K_END:
                        p2_attack_keys.add('special')
                    elif event.key == pygame.K_UP:
                        p2_jump_pressed = True

            keys = pygame.key.get_pressed()

            # Update
            if self.state == FIGHTING:
                self._update_fighting(keys, p1_attack_keys, p2_attack_keys,
                                       p1_jump_pressed, p2_jump_pressed)
            elif self.state == ROUND_END:
                self._update_round_end()
            elif self.state == GAME_OVER:
                pass  # Wait for space

            # Draw
            self._draw()

            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

    def _update_fighting(self, keys, p1_attack_keys, p2_attack_keys,
                         p1_jump_pressed, p2_jump_pressed):
        """Update game logic during fighting state."""
        # Player 1 controls (WASD + FGH)
        p1 = self.player1
        if p1.state != HIT and p1.state != DEAD:
            if keys[pygame.K_a]:
                p1.vx = -p1.speed
                p1.facing_right = False
                if p1.state == IDLE:
                    p1.state = WALKING
            elif keys[pygame.K_d]:
                p1.vx = p1.speed
                p1.facing_right = True
                if p1.state == IDLE:
                    p1.state = WALKING
            elif p1.state == WALKING:
                p1.state = IDLE

            if p1_jump_pressed and p1.on_ground:
                p1.jump()

            # Attack keys - only on KEYDOWN (no auto-repeat)
            if 'light' in p1_attack_keys:
                p1.light_attack()
            if 'heavy' in p1_attack_keys:
                p1.heavy_attack()
            if 'special' in p1_attack_keys:
                p1.special_attack()

            if keys[pygame.K_s]:
                p1.start_block()
            else:
                p1.stop_block()

        # Player 2 controls (Arrows + JKL;)
        p2 = self.player2
        if p2.state != HIT and p2.state != DEAD:
            if keys[pygame.K_LEFT]:
                p2.vx = -p2.speed
                p2.facing_right = False
                if p2.state == IDLE:
                    p2.state = WALKING
            elif keys[pygame.K_RIGHT]:
                p2.vx = p2.speed
                p2.facing_right = True
                if p2.state == IDLE:
                    p2.state = WALKING
            elif p2.state == WALKING:
                p2.state = IDLE

            if p2_jump_pressed and p2.on_ground:
                p2.jump()

            # Attack keys - only on KEYDOWN (no auto-repeat)
            if 'light' in p2_attack_keys:
                p2.light_attack()
            if 'heavy' in p2_attack_keys:
                p2.heavy_attack()
            if 'special' in p2_attack_keys:
                p2.special_attack()

            if keys[pygame.K_DOWN]:
                p2.start_block()
            else:
                p2.stop_block()

        # Update fighters
        p1.update(p2)
        p2.update(p1)

        # Keep fighters apart (push collision)
        dx = p1.x - p2.x
        if abs(dx) < 50:
            push = (50 - abs(dx)) / 2
            if dx > 0:
                p1.x += push
                p2.x -= push
            else:
                p1.x -= push
                p2.x += push

        # Handle attack collisions using current_attack tracking
        # Player 1 attacks - check during active frames
        if p1.current_attack is not None:
            if p1.state == ATTACKING:
                # Active frames: light = frames 5-14, heavy = frames 8-22
                if p1.current_attack == LIGHT_ATTACK and 5 < p1.state_timer < 15:
                    if self.handle_collision(p1, p2, LIGHT_ATTACK):
                        p1.current_attack = None  # prevent multi-hit
                elif p1.current_attack == HEAVY_ATTACK and 8 < p1.state_timer < 23:
                    if self.handle_collision(p1, p2, HEAVY_ATTACK):
                        p1.current_attack = None
            elif p1.state == SPECIAL:
                # Special active frames: frames 12-28
                if 12 < p1.state_timer < 29:
                    if self.handle_collision(p1, p2, SPECIAL_ATTACK):
                        p1.current_attack = None
            else:
                p1.current_attack = None

        # Player 2 attacks
        if p2.current_attack is not None:
            if p2.state == ATTACKING:
                if p2.current_attack == LIGHT_ATTACK and 5 < p2.state_timer < 15:
                    if self.handle_collision(p2, p1, LIGHT_ATTACK):
                        p2.current_attack = None
                elif p2.current_attack == HEAVY_ATTACK and 8 < p2.state_timer < 23:
                    if self.handle_collision(p2, p1, HEAVY_ATTACK):
                        p2.current_attack = None
            elif p2.state == SPECIAL:
                if 12 < p2.state_timer < 29:
                    if self.handle_collision(p2, p1, SPECIAL_ATTACK):
                        p2.current_attack = None
            else:
                p2.current_attack = None

    def _update_round_end(self):
        """Update during round end state. Advance to next round or game over."""
        self.round_timer -= 1
        if self.round_timer <= 0:
            # Check if someone won 2 out of 3 rounds
            if self.p1_rounds_won >= 2 or self.p2_rounds_won >= 2:
                self.state = GAME_OVER
            else:
                # Advance to next round
                self.round += 1
                self.reset_round()
                self.state = FIGHTING

    def _draw(self):
        """Draw everything."""
        screen = self.screen

        # Screen shake offset
        shake_x = 0
        shake_y = 0
        if self.screen_shake > 0:
            shake_x = random.randint(-4, 4)
            shake_y = random.randint(-4, 4)
            self.screen_shake -= 1

        # Draw background
        self.draw_background(screen)

        # Draw particles
        for p in self.particles[:]:
            if not p.update():
                self.particles.remove(p)
            else:
                p.draw(screen)

        # Draw fighters
        self.player1.draw(screen)
        self.player2.draw(screen)

        # Draw HUD
        self.draw_hud(screen)

        # Draw round end message (skip if this is the final match-winning round)
        if self.state == ROUND_END:
            # Check if this round win also ends the match
            is_match_ending = (self.p1_rounds_won >= 2 or self.p2_rounds_won >= 2)
            if not is_match_ending:
                self.draw_round_end(screen)

        # Draw game over
        if self.state == GAME_OVER:
            self.draw_game_over(screen)

        # Apply screen shake
        if shake_x != 0 or shake_y != 0:
            # We need to shift everything - easiest way is to blit the screen onto itself
            # Actually, let's just draw everything offset
            pass

        pygame.display.flip()


if __name__ == "__main__":
    game = FightingGame()
    game.run()
