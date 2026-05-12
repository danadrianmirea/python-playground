# racing_game.py - Top-down racing game

import pygame
import math
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Racing Game")

# Game constants
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 100, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
BROWN = (139, 69, 19)

# Fonts
font = pygame.font.Font(None, 36)
score_font = pygame.font.Font(None, 48)
game_over_font = pygame.font.Font(None, 60)
small_font = pygame.font.Font(None, 24)

# Clock
clock = pygame.time.Clock()

# Track definition (waypoints forming a loop)
TRACK_WAYPOINTS = [
    (100, 100),   # Top-left
    (400, 80),    # Top
    (700, 100),   # Top-right
    (800, 300),   # Right
    (750, 500),   # Bottom-right
    (500, 600),   # Bottom
    (200, 580),   # Bottom-left
    (80, 400),    # Left
]

TRACK_WIDTH = 120  # Half-width of the track from center line


def get_track_segment(point_a, point_b, t):
    """Get a point along a track segment."""
    x = point_a[0] + (point_b[0] - point_a[0]) * t
    y = point_a[1] + (point_b[1] - point_a[1]) * t
    return (x, y)


def get_closest_point_on_segment(point, seg_a, seg_b):
    """Get the closest point on a line segment to a given point."""
    ax, ay = seg_a
    bx, by = seg_b
    px, py = point

    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy

    if length_sq == 0:
        return seg_a, 0

    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0, min(1, t))

    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return (closest_x, closest_y), t


def point_to_segment_distance(point, seg_a, seg_b):
    """Get distance from a point to a line segment."""
    closest, _ = get_closest_point_on_segment(point, seg_a, seg_b)
    dx = point[0] - closest[0]
    dy = point[1] - closest[1]
    return math.sqrt(dx * dx + dy * dy)


class Car:
    """A race car."""

    def __init__(self, x, y, color, name="Player", is_player=False):
        self.x = x
        self.y = y
        self.color = color
        self.name = name
        self.is_player = is_player

        # Car physics
        self.angle = 0  # radians, 0 = right
        self.speed = 0
        self.max_speed = 6 if is_player else 5
        self.acceleration = 0.15
        self.braking = 0.2
        self.friction = 0.05
        self.turn_speed = 0.04  # radians per frame

        # Car dimensions (height is the length of the car, width is the width)
        self.width = 35
        self.height = 20

        # Racing state
        self.current_waypoint = 0
        self.lap = 0
        self.laps_completed = 0
        self.total_laps = 3
        self.finished = False
        self.finish_time = 0
        self.race_time = 0

        # AI state
        self.ai_target_speed = 0
        self.ai_steering = 0

    def get_rect(self):
        """Get the car's collision rect."""
        return pygame.Rect(self.x - self.width // 2, self.y - self.height // 2,
                           self.width, self.height)

    def get_corners(self):
        """Get the 4 corners of the car (for collision)."""
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        hw = self.width / 2
        hh = self.height / 2

        corners = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            corners.append((self.x + rx, self.y + ry))
        return corners

    def accelerate(self):
        self.speed = min(self.speed + self.acceleration, self.max_speed)

    def brake(self):
        self.speed = max(self.speed - self.braking, -self.max_speed / 2)

    def turn_left(self):
        if abs(self.speed) > 0.5:
            self.angle -= self.turn_speed * (self.speed / self.max_speed)

    def turn_right(self):
        if abs(self.speed) > 0.5:
            self.angle += self.turn_speed * (self.speed / self.max_speed)

    def update(self, track_waypoints):
        """Update car position and state."""
        if self.finished:
            return

        self.race_time += 1

        # Apply friction
        if abs(self.speed) < self.friction:
            self.speed = 0
        elif self.speed > 0:
            self.speed -= self.friction
        elif self.speed < 0:
            self.speed += self.friction

        # Move car
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # Check waypoint progression
        self._update_waypoints(track_waypoints)

    def _update_waypoints(self, waypoints):
        """Check if car has reached the next waypoint."""
        target = waypoints[self.current_waypoint]
        dx = self.x - target[0]
        dy = self.y - target[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 40:
            self.current_waypoint = (self.current_waypoint + 1) % len(waypoints)

            # Check if we completed a lap
            if self.current_waypoint == 0:
                self.laps_completed += 1
                if self.laps_completed >= self.total_laps:
                    self.finished = True
                    self.finish_time = self.race_time

    def is_on_track(self, waypoints):
        """Check if car is on the track (within TRACK_WIDTH of center line)."""
        # Find closest segment
        min_dist = float('inf')
        for i in range(len(waypoints)):
            a = waypoints[i]
            b = waypoints[(i + 1) % len(waypoints)]
            dist = point_to_segment_distance((self.x, self.y), a, b)
            min_dist = min(min_dist, dist)

        return min_dist < TRACK_WIDTH

    def draw(self, surface, camera_offset=(0, 0)):
        """Draw the car."""
        cx = self.x - camera_offset[0]
        cy = self.y - camera_offset[1]

        # Don't draw if off screen
        if cx < -50 or cx > SCREEN_WIDTH + 50 or cy < -50 or cy > SCREEN_HEIGHT + 50:
            return

        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        hw = self.width / 2
        hh = self.height / 2

        # Car body corners
        corners = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            corners.append((cx + rx, cy + ry))

        # Draw car body
        pygame.draw.polygon(surface, self.color, corners)

        # Draw windshield
        windshield_corners = []
        for dx, dy in [(-hw * 0.6, -hh * 0.3), (hw * 0.6, -hh * 0.3),
                       (hw * 0.5, hh * 0.3), (-hw * 0.5, hh * 0.3)]:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            windshield_corners.append((cx + rx, cy + ry))
        pygame.draw.polygon(surface, (min(self.color[0] + 80, 255),
                                       min(self.color[1] + 80, 255),
                                       min(self.color[2] + 80, 255)),
                            windshield_corners)

        # Draw number on roof
        if self.is_player:
            text = small_font.render("P1", True, WHITE)
        else:
            text = small_font.render(self.name[0].upper(), True, WHITE)
        text_rect = text.get_rect(center=(cx, cy))
        surface.blit(text, text_rect)


class AICar(Car):
    """AI-controlled car."""

    def __init__(self, x, y, color, name):
        super().__init__(x, y, color, name, is_player=False)
        self.max_speed = 4 + random.uniform(0.5, 1.5)
        self.look_ahead = 2  # waypoints to look ahead
        self.steering_noise = random.uniform(-0.01, 0.01)

    def update_ai(self, waypoints):
        """AI steering logic."""
        if self.finished:
            return

        # Look ahead a few waypoints
        target_idx = (self.current_waypoint + self.look_ahead) % len(waypoints)
        target = waypoints[target_idx]

        # Calculate angle to target
        dx = target[0] - self.x
        dy = target[1] - self.y
        target_angle = math.atan2(dy, dx)

        # Calculate angle difference
        angle_diff = target_angle - self.angle
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Steer towards target
        steer_strength = 0.03
        if angle_diff > 0.1:
            self.angle += steer_strength + self.steering_noise
        elif angle_diff < -0.1:
            self.angle -= steer_strength + self.steering_noise

        # Speed control
        dist_to_next = math.sqrt((waypoints[self.current_waypoint][0] - self.x) ** 2 +
                                  (waypoints[self.current_waypoint][1] - self.y) ** 2)

        # Slow down for sharp turns, speed up on straights
        if abs(angle_diff) > 0.5:
            self.ai_target_speed = self.max_speed * 0.5
        elif abs(angle_diff) > 0.3:
            self.ai_target_speed = self.max_speed * 0.7
        else:
            self.ai_target_speed = self.max_speed

        # Accelerate or brake to reach target speed
        if self.speed < self.ai_target_speed:
            self.accelerate()
        elif self.speed > self.ai_target_speed + 0.5:
            self.brake()

        # Check if stuck (very slow and near waypoint)
        if self.speed < 1 and dist_to_next < 60:
            self.speed = 2
            self.angle += 0.1  # nudge to get unstuck


class Game:
    """Main game class."""

    def __init__(self):
        self.waypoints = TRACK_WAYPOINTS
        self.cars = []
        self.player = None
        self.camera_x = 0
        self.camera_y = 0
        self.game_over = False
        self.won = False
        self.race_started = False
        self.countdown = 120  # 2 seconds at 60fps
        self.finished_positions = []
        self.grass_patches = self._generate_grass()

        # Create player car
        start_pos = self._get_start_position()
        self.player = Car(start_pos[0], start_pos[1], RED, "Player", is_player=True)
        self.cars.append(self.player)

        # Create AI cars
        ai_configs = [
            ("Blaze", ORANGE),
            ("Storm", CYAN),
            ("Shadow", PURPLE),
        ]
        for i, (name, color) in enumerate(ai_configs):
            offset = (i + 1) * 30
            ai_start = (start_pos[0] - offset, start_pos[1])
            ai_car = AICar(ai_start[0], ai_start[1], color, name)
            self.cars.append(ai_car)

    def _generate_grass(self):
        """Generate decorative grass patches."""
        patches = []
        for _ in range(200):
            x = random.randint(0, SCREEN_WIDTH * 3)
            y = random.randint(0, SCREEN_HEIGHT * 3)
            # Only place grass outside the track
            on_track = False
            for i in range(len(self.waypoints)):
                a = self.waypoints[i]
                b = self.waypoints[(i + 1) % len(self.waypoints)]
                dist = point_to_segment_distance((x, y), a, b)
                if dist < TRACK_WIDTH + 20:
                    on_track = True
                    break
            if not on_track:
                patches.append((x, y, random.randint(3, 6)))
        return patches

    def _get_start_position(self):
        """Get the starting position on the track."""
        # Start between first and last waypoint
        a = self.waypoints[0]
        b = self.waypoints[-1]
        mx = (a[0] + b[0]) / 2
        my = (a[1] + b[1]) / 2
        return (mx, my - TRACK_WIDTH // 2)

    def handle_event(self, event):
        """Handle keyboard input."""
        if event.type == pygame.KEYDOWN:
            if self.game_over or self.won:
                if event.key == pygame.K_SPACE:
                    self.__init__()
                return

    def update(self):
        """Update game state."""
        if self.game_over or self.won:
            return

        if self.countdown > 0:
            self.countdown -= 1
            return

        self.race_started = True

        # Player input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.player.accelerate()
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.player.brake()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.player.turn_left()
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.player.turn_right()

        # Update player
        self.player.update(self.waypoints)

        # Update AI cars
        for car in self.cars:
            if car.is_player:
                continue
            car.update_ai(self.waypoints)
            car.update(self.waypoints)

        # Check car-to-car collisions
        self._check_car_collisions()

        # Check if player is off track
        if not self.player.is_on_track(self.waypoints):
            self.player.speed *= 0.9  # Slow down on grass

        # Update camera to follow player
        self.camera_x = self.player.x - SCREEN_WIDTH // 2
        self.camera_y = self.player.y - SCREEN_HEIGHT // 2

        # Check finish
        finished_cars = [c for c in self.cars if c.finished and c not in self.finished_positions]
        for car in finished_cars:
            self.finished_positions.append(car)

        if self.player.finished:
            player_pos = self.finished_positions.index(self.player) + 1
            if player_pos == 1:
                self.won = True
            else:
                self.game_over = True

        # Check if all AI finished (player lost)
        ai_finished = sum(1 for c in self.cars if not c.is_player and c.finished)
        if ai_finished >= 3 and not self.player.finished:
            self.game_over = True

    def _check_car_collisions(self):
        """Check and resolve collisions between cars."""
        for i in range(len(self.cars)):
            for j in range(i + 1, len(self.cars)):
                car1 = self.cars[i]
                car2 = self.cars[j]

                # Simple distance-based collision
                dx = car1.x - car2.x
                dy = car1.y - car2.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < 25:  # Collision threshold
                    # Push cars apart
                    if dist > 0:
                        push = (25 - dist) / 2
                        nx = dx / dist
                        ny = dy / dist
                        car1.x += nx * push
                        car1.y += ny * push
                        car2.x -= nx * push
                        car2.y -= ny * push

                    # Reduce speeds
                    avg_speed = (abs(car1.speed) + abs(car2.speed)) / 2
                    car1.speed *= 0.7
                    car2.speed *= 0.7

    def draw(self, surface):
        """Draw the entire game."""
        surface.fill(DARK_GREEN)

        # Draw grass patches
        for gx, gy, gs in self.grass_patches:
            sx = gx - self.camera_x
            sy = gy - self.camera_y
            if -50 < sx < SCREEN_WIDTH + 50 and -50 < sy < SCREEN_HEIGHT + 50:
                pygame.draw.circle(surface, (0, random.randint(120, 160), 0),
                                   (int(sx), int(sy)), gs)

        # Draw track
        self._draw_track(surface)

        # Draw start/finish line
        self._draw_finish_line(surface)

        # Draw cars
        for car in sorted(self.cars, key=lambda c: c.y):
            car.draw(surface, (self.camera_x, self.camera_y))

        # Draw countdown
        if self.countdown > 0:
            countdown_sec = self.countdown // 60 + 1
            if countdown_sec > 3:
                text = game_over_font.render("READY?", True, YELLOW)
            else:
                text = game_over_font.render(str(countdown_sec), True, YELLOW)
            surface.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2,
                                SCREEN_HEIGHT // 2 - text.get_height() // 2))

        # Draw HUD
        self._draw_hud(surface)

        # Draw game over / win
        if self.game_over or self.won:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            surface.blit(overlay, (0, 0))

            if self.won:
                text = game_over_font.render("YOU WIN!", True, YELLOW)
            else:
                text = game_over_font.render("GAME OVER", True, RED)
            surface.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2,
                                SCREEN_HEIGHT // 2 - 60))

            # Show final positions
            y_offset = SCREEN_HEIGHT // 2 + 10
            for pos, car in enumerate(self.finished_positions):
                color = GREEN if car.is_player else WHITE
                pos_text = font.render(f"{pos + 1}. {car.name}", True, color)
                surface.blit(pos_text, (SCREEN_WIDTH // 2 - 80, y_offset))
                y_offset += 35

            restart_text = font.render("Press SPACE to restart", True, WHITE)
            surface.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2,
                                        SCREEN_HEIGHT // 2 + 120))

    def _draw_track(self, surface):
        """Draw the race track."""
        # Draw track surface (gray)
        track_points = []
        inner_points = []
        outer_points = []

        for i in range(len(self.waypoints)):
            a = self.waypoints[i]
            b = self.waypoints[(i + 1) % len(self.waypoints)]

            # Calculate perpendicular direction
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                px = -dy / length * TRACK_WIDTH
                py = dx / length * TRACK_WIDTH
            else:
                px, py = 0, 0

            # Inner and outer edges
            inner_x = a[0] - px - self.camera_x
            inner_y = a[1] - py - self.camera_y
            outer_x = a[0] + px - self.camera_x
            outer_y = a[1] + py - self.camera_y

            inner_points.append((inner_x, inner_y))
            outer_points.append((outer_x, outer_y))

        # Draw track surface
        all_points = inner_points + outer_points[::-1]
        if len(all_points) >= 3:
            pygame.draw.polygon(surface, GRAY, all_points)

        # Draw track borders (white lines)
        if len(inner_points) >= 2:
            pygame.draw.lines(surface, WHITE, True, inner_points, 3)
            pygame.draw.lines(surface, WHITE, True, outer_points, 3)

        # Draw center line (dashed)
        for i in range(len(self.waypoints)):
            a = self.waypoints[i]
            b = self.waypoints[(i + 1) % len(self.waypoints)]
            segments = 20
            for j in range(0, segments, 2):
                t1 = j / segments
                t2 = (j + 1) / segments
                p1 = (a[0] + (b[0] - a[0]) * t1 - self.camera_x,
                      a[1] + (b[1] - a[1]) * t1 - self.camera_y)
                p2 = (a[0] + (b[0] - a[0]) * t2 - self.camera_x,
                      a[1] + (b[1] - a[1]) * t2 - self.camera_y)
                pygame.draw.line(surface, YELLOW, p1, p2, 2)

    def _draw_finish_line(self, surface):
        """Draw the start/finish line."""
        a = self.waypoints[0]
        b = self.waypoints[-1]

        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            px = -dy / length * TRACK_WIDTH
            py = dx / length * TRACK_WIDTH
        else:
            px, py = 0, 0

        # Draw checkered pattern
        num_checks = 8
        for i in range(num_checks):
            t1 = i / num_checks
            t2 = (i + 1) / num_checks

            p1 = (a[0] + (b[0] - a[0]) * t1 - px - self.camera_x,
                  a[1] + (b[1] - a[1]) * t1 - py - self.camera_y)
            p2 = (a[0] + (b[0] - a[0]) * t2 - px - self.camera_x,
                  a[1] + (b[1] - a[1]) * t2 - py - self.camera_y)
            p3 = (a[0] + (b[0] - a[0]) * t2 + px - self.camera_x,
                  a[1] + (b[1] - a[1]) * t2 + py - self.camera_y)
            p4 = (a[0] + (b[0] - a[0]) * t1 + px - self.camera_x,
                  a[1] + (b[1] - a[1]) * t1 + py - self.camera_y)

            color = WHITE if i % 2 == 0 else BLACK
            pygame.draw.polygon(surface, color, [p1, p2, p3, p4])

    def _draw_hud(self, surface):
        """Draw the HUD (lap counter, speed, positions)."""
        # Lap counter
        lap_text = font.render(f"Lap: {self.player.laps_completed + 1}/{self.player.total_laps}",
                               True, WHITE)
        surface.blit(lap_text, (10, 10))

        # Speed
        speed_kmh = int(abs(self.player.speed) * 30)
        speed_text = font.render(f"Speed: {speed_kmh} km/h", True, WHITE)
        surface.blit(speed_text, (10, 50))

        # Position
        positions = sorted(self.cars, key=lambda c: (c.laps_completed * 1000 +
                                                       c.current_waypoint * 10 +
                                                       (1 - c.speed / c.max_speed)),
                           reverse=True)
        player_pos = next(i for i, c in enumerate(positions) if c.is_player) + 1
        pos_text = font.render(f"Position: {player_pos}/{len(self.cars)}", True, WHITE)
        surface.blit(pos_text, (10, 90))

        # Controls hint
        if not self.race_started:
            controls = [
                "Arrow Keys / WASD - Drive",
                "UP/W - Accelerate",
                "DOWN/S - Brake/Reverse",
                "LEFT/A - Turn Left",
                "RIGHT/D - Turn Right",
            ]
            y = SCREEN_HEIGHT - 120
            for line in controls:
                ctrl_text = small_font.render(line, True, WHITE)
                surface.blit(ctrl_text, (SCREEN_WIDTH - 250, y))
                y += 22

        # AI positions on right side
        y = 130
        for i, car in enumerate(positions):
            if car.is_player:
                continue
            color = car.color
            name_text = small_font.render(f"{i + 1}. {car.name}", True, color)
            surface.blit(name_text, (10, y))
            y += 22


def main():
    """Main game loop."""
    game = Game()
    running = True

    while running:
        dt = clock.tick(FPS)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            game.handle_event(event)

        # Update
        game.update()

        # Draw
        game.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()