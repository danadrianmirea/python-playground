import pygame
import random
import math
import sys

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
GRAVITY = 0.15
GROUND_HEIGHT = 80
GORILLA_WIDTH = 40
GORILLA_HEIGHT = 50

# Colors
SKY_COLOR = (100, 150, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
DARK_GRAY = (80, 80, 80)
BUILDING_COLORS = [
    (60, 60, 80), (70, 70, 90), (80, 80, 100),
    (90, 90, 110), (100, 100, 120), (75, 75, 95),
    (85, 85, 105), (65, 65, 85), (95, 95, 115)
]
GORILLA_BODY = (180, 130, 80)
GORILLA_DARK = (120, 80, 40)
BANANA_COLOR = (255, 220, 50)
EXPLOSION_COLORS = [(255, 100, 0), (255, 150, 0), (255, 200, 50), (255, 255, 100)]
WINDOW_COLOR = (200, 200, 50)
STAR_COLOR = (255, 255, 200)

# Aiming adjustment rate per second (frame-rate independent)
AUTOREPEAT_SPEED = 1
AUTOREPEAT_SPEED_SLOW = 0.3

class Building:
    def __init__(self, x, width, height):
        self.x = x
        self.width = width
        self.height = height
        self.color = random.choice(BUILDING_COLORS)
        self.windows = []
        # Generate windows
        win_w = 8
        win_h = 10
        win_margin = 6
        for wx in range(int(x + 10), int(x + width - 10), win_w + win_margin):
            for wy in range(int(WINDOW_HEIGHT - GROUND_HEIGHT - height + 15),
                           int(WINDOW_HEIGHT - GROUND_HEIGHT - 10), win_h + win_margin):
                if random.random() < 0.7:
                    self.windows.append(pygame.Rect(wx, wy, win_w, win_h))

    def draw(self, screen):
        rect = pygame.Rect(self.x, WINDOW_HEIGHT - GROUND_HEIGHT - self.height,
                          self.width, self.height)
        pygame.draw.rect(screen, self.color, rect)
        pygame.draw.rect(screen, DARK_GRAY, rect, 2)
        for win in self.windows:
            pygame.draw.rect(screen, WINDOW_COLOR, win)

class Gorilla:
    def __init__(self, x, y, facing_right):
        self.x = x
        self.y = y
        self.facing_right = facing_right
        self.health = 100

    def draw(self, screen):
        # Body
        body_rect = pygame.Rect(self.x - GORILLA_WIDTH // 2,
                               self.y - GORILLA_HEIGHT,
                               GORILLA_WIDTH, GORILLA_HEIGHT)
        pygame.draw.ellipse(screen, GORILLA_BODY, body_rect)

        # Head
        head_center = (self.x, self.y - GORILLA_HEIGHT - 12)
        pygame.draw.circle(screen, GORILLA_BODY, head_center, 14)

        # Eyes
        eye_offset = 5 if self.facing_right else -5
        pygame.draw.circle(screen, WHITE, (self.x + eye_offset - 3, self.y - GORILLA_HEIGHT - 14), 4)
        pygame.draw.circle(screen, WHITE, (self.x + eye_offset + 3, self.y - GORILLA_HEIGHT - 14), 4)
        pygame.draw.circle(screen, BLACK, (self.x + eye_offset - 3, self.y - GORILLA_HEIGHT - 14), 2)
        pygame.draw.circle(screen, BLACK, (self.x + eye_offset + 3, self.y - GORILLA_HEIGHT - 14), 2)

        # Arms
        arm_dir = 1 if self.facing_right else -1
        # Left arm (behind)
        pygame.draw.line(screen, GORILLA_DARK,
                        (self.x - 15, self.y - GORILLA_HEIGHT + 10),
                        (self.x - 25, self.y - GORILLA_HEIGHT // 2), 5)
        # Right arm (throwing arm)
        pygame.draw.line(screen, GORILLA_DARK,
                        (self.x + 15, self.y - GORILLA_HEIGHT + 10),
                        (self.x + 25 * arm_dir, self.y - GORILLA_HEIGHT // 2), 5)

        # Legs
        pygame.draw.line(screen, GORILLA_DARK,
                        (self.x - 10, self.y),
                        (self.x - 15, self.y + 15), 6)
        pygame.draw.line(screen, GORILLA_DARK,
                        (self.x + 10, self.y),
                        (self.x + 15, self.y + 15), 6)

class Banana:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.trail = []
        self.active = True

    def update(self, buildings, dt):
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 20:
            self.trail.pop(0)

        self.x += self.vx * dt
        self.vy += GRAVITY * dt
        self.y += self.vy * dt

        # Check terrain (ground)
        if self.y > WINDOW_HEIGHT - GROUND_HEIGHT:
            self.active = False
            return

        # Check out of bounds horizontally
        if self.x < 0 or self.x > WINDOW_WIDTH:
            self.active = False
            return

        # Check collision with buildings
        for building in buildings:
            if (building.x <= self.x <= building.x + building.width and
                WINDOW_HEIGHT - GROUND_HEIGHT - building.height <= self.y <= WINDOW_HEIGHT - GROUND_HEIGHT):
                self.active = False
                return

    def draw(self, screen):
        # Draw trail
        for i, pos in enumerate(self.trail):
            alpha = i / len(self.trail)
            size = max(1, int(4 * alpha))
            pygame.draw.circle(screen, (255, 200, 50, 100), pos, size)

        # Draw banana
        angle = math.atan2(self.vy, self.vx)
        length = 14
        end_x = self.x + math.cos(angle) * length
        end_y = self.y + math.sin(angle) * length
        pygame.draw.line(screen, BANANA_COLOR, (self.x, self.y), (end_x, end_y), 5)
        pygame.draw.circle(screen, BANANA_COLOR, (int(self.x), int(self.y)), 4)

class Explosion:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 5
        self.max_radius = 50
        self.active = True
        self.particles = []
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 3,
                'life': random.uniform(0.5, 1.0)
            })

    def update(self, dt):
        self.radius += 3 * dt
        if self.radius >= self.max_radius:
            self.active = False

        for p in self.particles:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['vy'] += 0.1 * dt
            p['life'] -= 0.02 * dt

    def draw(self, screen):
        # Explosion circle
        if self.radius < self.max_radius:
            alpha = int(255 * (1 - self.radius / self.max_radius))
            for i, color in enumerate(EXPLOSION_COLORS):
                r = self.radius - i * 8
                if r > 0:
                    pygame.draw.circle(screen, color, (int(self.x), int(self.y)), int(r), 3)

        # Particles
        for p in self.particles:
            if p['life'] > 0:
                alpha = int(255 * p['life'])
                pygame.draw.circle(screen, (255, 200, 50), (int(p['x']), int(p['y'])), max(1, int(3 * p['life'])))

class GorillasGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Gorillas")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.running = True
        self.state = "menu"
        self.buildings = []
        self.gorillas = []
        self.banana = None
        self.explosions = []
        self.current_player = 0
        self.angle = 45
        self.power = 50
        self.wind = 0
        self.message = ""
        self.message_timer = 0
        self.aiming = True
        self.turn_count = 0
        self.banana_landed = False
        self.banana_land_time = 0
        self.generate_city()

    def generate_city(self):
        self.buildings = []
        num_buildings = random.randint(8, 14)
        building_width = WINDOW_WIDTH // num_buildings
        for i in range(num_buildings):
            width = building_width - 2
            height = random.randint(60, 200)
            x = i * building_width + 1
            self.buildings.append(Building(x, width, height))

        # Place gorillas on top of random buildings
        g1_idx = random.randint(0, len(self.buildings) // 2 - 1)
        g2_idx = random.randint(len(self.buildings) // 2 + 1, len(self.buildings) - 1)

        b1 = self.buildings[g1_idx]
        b2 = self.buildings[g2_idx]

        g1_x = b1.x + b1.width // 2
        g1_y = WINDOW_HEIGHT - GROUND_HEIGHT - b1.height

        g2_x = b2.x + b2.width // 2
        g2_y = WINDOW_HEIGHT - GROUND_HEIGHT - b2.height

        self.gorillas = [
            Gorilla(g1_x, g1_y, True),
            Gorilla(g2_x, g2_y, False)
        ]

        self.wind = random.uniform(-3, 3)

    def reset_game(self):
        self.generate_city()
        self.banana = None
        self.explosions = []
        self.current_player = 0
        self.angle = 45
        self.power = 50
        self.aiming = True
        self.turn_count = 0
        self.message = ""
        self.message_timer = 0
        self.banana_landed = False
        self.banana_land_time = 0

    def handle_menu_click(self, pos):
        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 250, 200, 50)
        if btn_rect.collidepoint(pos):
            self.reset_game()
            self.state = "playing"

    def handle_game_click(self, pos):
        if not self.aiming:
            return

        # Calculate angle and power from click relative to current gorilla
        gorilla = self.gorillas[self.current_player]
        dx = pos[0] - gorilla.x
        dy = pos[1] - gorilla.y

        # Convert to angle and power
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 10:
            return

        angle_rad = math.atan2(-dy, dx)
        self.angle = math.degrees(angle_rad)
        self.power = min(100, max(10, distance / 4))

        self.throw_banana()

    def throw_banana(self):
        gorilla = self.gorillas[self.current_player]
        angle_rad = math.radians(self.angle)
        speed = self.power * 0.4

        vx = math.cos(angle_rad) * speed + self.wind * 0.3
        vy = -math.sin(angle_rad) * speed

        self.banana = Banana(gorilla.x, gorilla.y - GORILLA_HEIGHT - 10, vx, vy)
        self.aiming = False
        self.turn_count += 1
        self.banana_landed = False

    def check_hit(self):
        if not self.banana:
            return

        bx, by = self.banana.x, self.banana.y

        # Check if hit a gorilla (check against center of body+head)
        # Skip the current player's gorilla to avoid self-hit on throw
        for i, gorilla in enumerate(self.gorillas):
            if i == self.current_player:
                continue
            dist = math.sqrt((bx - gorilla.x) ** 2 + (by - (gorilla.y - GORILLA_HEIGHT // 2 - 6)) ** 2)
            if dist < 40:
                # Hit! Deactivate banana, create explosion, and mark as landed
                self.banana.active = False
                self.banana_landed = True
                self.banana_land_time = pygame.time.get_ticks()
                self.explosions.append(Explosion(bx, by))
                gorilla.health -= 50
                if gorilla.health <= 0:
                    self.state = "game_over"
                    self.message = f"Player {self.current_player + 1} wins!"
                    self.message_timer = pygame.time.get_ticks()
                else:
                    self.message = f"Direct hit on Player {i + 1}!"
                    self.message_timer = pygame.time.get_ticks()
                return

        # Only process landing effects when banana becomes inactive (hit terrain/building)
        if not self.banana.active:
            # Create explosion
            self.explosions.append(Explosion(bx, by))

            # Check if hit a building
            for building in self.buildings:
                if (building.x <= bx <= building.x + building.width and
                    WINDOW_HEIGHT - GROUND_HEIGHT - building.height <= by <= WINDOW_HEIGHT - GROUND_HEIGHT):
                    self.message = "Hit a building!"
                    self.message_timer = pygame.time.get_ticks()
                    return

            self.message = "Miss!"
            self.message_timer = pygame.time.get_ticks()

    def next_turn(self):
        self.banana = None
        self.current_player = 1 - self.current_player
        self.angle = 45
        self.power = 50
        self.aiming = True
        self.banana_landed = False

        # Random wind change
        self.wind += random.uniform(-1, 1)
        self.wind = max(-5, min(5, self.wind))

    def draw_menu(self):
        self.screen.fill(SKY_COLOR)

        # Draw all buildings for background
        for building in self.buildings:
            building.draw(self.screen)

        # Draw ground
        ground_rect = pygame.Rect(0, WINDOW_HEIGHT - GROUND_HEIGHT, WINDOW_WIDTH, GROUND_HEIGHT)
        pygame.draw.rect(self.screen, (40, 120, 40), ground_rect)
        pygame.draw.rect(self.screen, DARK_GRAY, ground_rect, 2)

        title = self.font_large.render("GORILLAS", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 50))
        self.screen.blit(title, title_rect)

        subtitle = self.font_medium.render("The Classic Artillery Game", True, WHITE)
        sub_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 100))
        self.screen.blit(subtitle, sub_rect)

        instr = [
            "Click on the screen to aim and throw",
            "Player 1: left side  |  Player 2: right side",
            "First to hit the opponent twice wins!"
        ]
        for i, line in enumerate(instr):
            t = self.font_small.render(line, True, WHITE)
            t_rect = t.get_rect(center=(WINDOW_WIDTH // 2, 150 + i * 30))
            self.screen.blit(t, t_rect)
        
        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 250, 200, 50)
        mouse_pos = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mouse_pos)
        color = (100, 200, 100) if hover else (60, 150, 60)
        pygame.draw.rect(self.screen, color, btn_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, btn_rect, 2, border_radius=10)
        text = self.font_medium.render("Play Game", True, WHITE)
        text_rect = text.get_rect(center=btn_rect.center)
        self.screen.blit(text, text_rect)


    def draw_game(self):
        self.screen.fill(SKY_COLOR)

        # Draw stars (for fun)
        for _ in range(50):
            sx = random.randint(0, WINDOW_WIDTH)
            sy = random.randint(0, WINDOW_HEIGHT // 3)
            self.screen.set_at((sx, sy), STAR_COLOR)

        # Draw buildings
        for building in self.buildings:
            building.draw(self.screen)

        # Draw ground
        ground_rect = pygame.Rect(0, WINDOW_HEIGHT - GROUND_HEIGHT, WINDOW_WIDTH, GROUND_HEIGHT)
        pygame.draw.rect(self.screen, (40, 120, 40), ground_rect)
        pygame.draw.rect(self.screen, DARK_GRAY, ground_rect, 2)

        # Draw gorillas
        for gorilla in self.gorillas:
            gorilla.draw(self.screen)

        # Draw banana
        if self.banana and self.banana.active:
            self.banana.draw(self.screen)

        # Draw explosions
        for explosion in self.explosions:
            explosion.draw(self.screen)

        # Draw UI
        gorilla = self.gorillas[self.current_player]
        player_name = f"Player {self.current_player + 1}"
        player_color = (100, 200, 255) if self.current_player == 0 else (255, 150, 150)

        # Top bar
        pygame.draw.rect(self.screen, (0, 0, 0, 150), (0, 0, WINDOW_WIDTH, 60))
        pygame.draw.rect(self.screen, WHITE, (0, 0, WINDOW_WIDTH, 60), 2)

        turn_text = self.font_medium.render(f"{player_name}'s Turn", True, player_color)
        turn_rect = turn_text.get_rect(center=(WINDOW_WIDTH // 2, 20))
        self.screen.blit(turn_text, turn_rect)

        if self.aiming:
            info = f"Angle: {self.angle:.0f}°  Power: {self.power:.0f}%"
        else:
            info = "Banana in flight..."
        info_text = self.font_small.render(info, True, WHITE)
        info_rect = info_text.get_rect(center=(WINDOW_WIDTH // 2, 45))
        self.screen.blit(info_text, info_rect)

        # Wind indicator
        wind_text = f"Wind: {self.wind:+.1f}"
        wind_color = (100, 200, 255) if self.wind > 0 else (255, 150, 150)
        wind_surf = self.font_small.render(wind_text, True, wind_color)
        wind_rect = wind_surf.get_rect(topright=(WINDOW_WIDTH - 20, 10))
        self.screen.blit(wind_surf, wind_rect)

        # Health bars
        for i, g in enumerate(self.gorillas):
            bar_x = 20 if i == 0 else WINDOW_WIDTH - 220
            bar_y = 10
            bar_width = 200
            bar_height = 15

            pygame.draw.rect(self.screen, DARK_GRAY, (bar_x, bar_y, bar_width, bar_height))
            health_width = int(bar_width * (g.health / 100))
            health_color = (100, 200, 100) if g.health > 50 else (200, 100, 100)
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, health_width, bar_height))
            pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 1)

            hp_text = self.font_small.render(f"P{i+1}: {g.health}", True, WHITE)
            hp_rect = hp_text.get_rect(topleft=(bar_x, bar_y + bar_height + 2))
            self.screen.blit(hp_text, hp_rect)

        # Aiming line
        if self.aiming:
            angle_rad = math.radians(self.angle)
            speed = self.power * 0.4
            end_x = gorilla.x + math.cos(angle_rad) * 80
            end_y = gorilla.y - GORILLA_HEIGHT - 10 - math.sin(angle_rad) * 80
            pygame.draw.line(self.screen, (255, 255, 255, 100),
                           (gorilla.x, gorilla.y - GORILLA_HEIGHT - 10),
                           (end_x, end_y), 2)

            # Draw trajectory preview
            preview_x = gorilla.x
            preview_y = gorilla.y - GORILLA_HEIGHT - 10
            pvx = math.cos(angle_rad) * speed + self.wind * 0.3
            pvy = -math.sin(angle_rad) * speed
            for _ in range(30):
                preview_x += pvx
                pvy += GRAVITY
                preview_y += pvy
                if preview_y > WINDOW_HEIGHT - GROUND_HEIGHT:
                    break
                pygame.draw.circle(self.screen, (255, 255, 255, 50), (int(preview_x), int(preview_y)), 2)

        # Message
        if self.message and pygame.time.get_ticks() - self.message_timer < 1500:
            msg = self.font_medium.render(self.message, True, WHITE)
            msg_rect = msg.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 40))
            bg_rect = msg_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect, border_radius=5)
            self.screen.blit(msg, msg_rect)

    def draw_game_over(self):
        self.draw_game()

        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        winner = self.current_player  # The thrower wins when the opponent is hit
        congrats = self.font_large.render(f"Player {winner + 1} Wins!", True, (255, 215, 0))
        congrats_rect = congrats.get_rect(center=(WINDOW_WIDTH // 2, 200))
        self.screen.blit(congrats, congrats_rect)

        stats = self.font_medium.render(f"Total turns: {self.turn_count}", True, WHITE)
        stats_rect = stats.get_rect(center=(WINDOW_WIDTH // 2, 260))
        self.screen.blit(stats, stats_rect)

        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 320, 200, 50)
        mouse_pos = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mouse_pos)
        color = (100, 200, 100) if hover else (60, 150, 60)
        pygame.draw.rect(self.screen, color, btn_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, btn_rect, 2, border_radius=10)
        again_text = self.font_medium.render("Play Again", True, WHITE)
        again_rect = again_text.get_rect(center=btn_rect.center)
        self.screen.blit(again_text, again_rect)

        menu_btn = pygame.Rect(WINDOW_WIDTH // 2 - 100, 390, 200, 50)
        hover2 = menu_btn.collidepoint(mouse_pos)
        color2 = (150, 150, 150) if hover2 else (100, 100, 100)
        pygame.draw.rect(self.screen, color2, menu_btn, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, menu_btn, 2, border_radius=10)
        menu_text = self.font_medium.render("Main Menu", True, WHITE)
        menu_rect = menu_text.get_rect(center=menu_btn.center)
        self.screen.blit(menu_text, menu_rect)

        return btn_rect, menu_btn

    def run(self):
        while self.running:
            # Delta time in seconds, capped at 50ms to prevent physics explosions on lag spikes
            dt = min(self.clock.tick(FPS) / 16.667, 3.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.state == "menu":
                        self.handle_menu_click(event.pos)
                    elif self.state == "playing":
                        self.handle_game_click(event.pos)
                    elif self.state == "game_over":
                        btn_rect, menu_btn = self.draw_game_over()
                        if btn_rect.collidepoint(event.pos):
                            self.reset_game()
                            self.state = "playing"
                        elif menu_btn.collidepoint(event.pos):
                            self.state = "menu"

                if event.type == pygame.KEYDOWN and self.state == "playing" and self.aiming:
                    if event.key == pygame.K_SPACE:
                        self.throw_banana()

            if self.state == "playing":
                # Handle smooth continuous key input for aiming (frame-rate independent)
                if self.aiming:
                    keys = pygame.key.get_pressed()
                    autoRepeatSpeed = AUTOREPEAT_SPEED
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                        autoRepeatSpeed = AUTOREPEAT_SPEED_SLOW
                    if keys[pygame.K_LEFT]:
                        self.angle = min(180, self.angle + autoRepeatSpeed * dt)
                    if keys[pygame.K_RIGHT]:
                        self.angle = max(0, self.angle - autoRepeatSpeed * dt)
                    if keys[pygame.K_UP]:
                        self.power = min(100, self.power + autoRepeatSpeed * dt)
                    if keys[pygame.K_DOWN]:
                        self.power = max(10, self.power - autoRepeatSpeed * dt)

                # Update banana
                if self.banana:
                    if self.banana.active:
                        self.banana.update(self.buildings, dt)
                        # Check for gorilla hit every frame while banana is in flight
                        self.check_hit()
                    elif not self.banana_landed:
                        # First frame the banana is inactive - process hit and record time
                        self.check_hit()
                        self.banana_landed = True
                        self.banana_land_time = pygame.time.get_ticks()
                    else:
                        # Wait a moment before next turn
                        if pygame.time.get_ticks() - self.banana_land_time > 800:
                            self.next_turn()

                # Update explosions
                self.explosions = [e for e in self.explosions if e.active]
                for explosion in self.explosions:
                    explosion.update(dt)

            if self.state == "menu":
                self.draw_menu()
            elif self.state == "playing":
                self.draw_game()
            elif self.state == "game_over":
                self.draw_game_over()

            pygame.display.flip()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = GorillasGame()
    game.run()
