"""
Asteroids - A retro vector-graphics space rock blaster inspired by the 1979 arcade classic.

Fly your ship through an asteroid field, blasting rocks into smaller pieces.
Watch out for flying saucers! Survive as long as you can.

Controls:
  LEFT/RIGHT arrows - Rotate ship
  UP arrow - Thrust
  SPACE - Shoot
  R - Restart after game over
  ESC - Quit

No external dependencies required - uses only tkinter (standard library).
"""

import tkinter as tk
import random
import math


# Constants
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
FPS = 60

# Colors
BLACK = "#000000"
WHITE = "#FFFFFF"
DIM_WHITE = "#B4B4B4"
RED = "#FF5050"
GREEN = "#50FF50"
YELLOW = "#FFFF64"
ORANGE = "#FFB432"

# Ship constants
SHIP_RADIUS = 18
SHIP_ROTATION_SPEED = 5  # degrees per frame
THRUST_POWER = 0.25
FRICTION = 0.99
MAX_SPEED = 8
SHOOT_COOLDOWN = 12  # frames
BULLET_SPEED = 12
BULLET_LIFETIME = 45  # frames
INVINCIBLE_FRAMES = 90  # 1.5 seconds at 60 FPS

# Asteroid constants
ASTEROID_SPEED_MIN = 1.0
ASTEROID_SPEED_MAX = 3.5
ASTEROID_VERTICES_MIN = 8
ASTEROID_VERTICES_MAX = 12
ASTEROID_RADIUS_LARGE = 45
ASTEROID_RADIUS_MEDIUM = 25
ASTEROID_RADIUS_SMALL = 12
SCORE_LARGE = 20
SCORE_MEDIUM = 50
SCORE_SMALL = 100

# Saucer constants
SAUCER_SIZE = 24
SAUCER_SPEED = 3
SAUCER_SHOOT_COOLDOWN = 30
SAUCER_SCORE = 300
SAUCER_SPAWN_INTERVAL = 600  # frames (10 seconds)
SAUCER_BULLET_SPEED = 5

# Game states
STATE_PLAYING = 0
STATE_GAME_OVER = 1
STATE_TITLE = 2


class Ship:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = WINDOW_WIDTH // 2
        self.y = WINDOW_HEIGHT // 2
        self.angle = -90  # pointing up (0 = right, -90 = up)
        self.vx = 0
        self.vy = 0
        self.radius = SHIP_RADIUS
        self.thrust_on = False
        self.invincible = INVINCIBLE_FRAMES
        self.alive = True
        self.shoot_timer = 0

    def get_tip(self):
        """Return the position of the ship's nose (for bullet spawning)."""
        rad = math.radians(self.angle)
        tip_x = self.x + self.radius * math.cos(rad)
        tip_y = self.y + self.radius * math.sin(rad)
        return tip_x, tip_y

    def get_shape_points(self):
        """Return the points of the ship triangle + exhaust."""
        rad = math.radians(self.angle)
        tip_x = self.x + self.radius * math.cos(rad)
        tip_y = self.y + self.radius * math.sin(rad)
        left_rad = math.radians(self.angle + 150)
        left_x = self.x + self.radius * 0.7 * math.cos(left_rad)
        left_y = self.y + self.radius * 0.7 * math.sin(left_rad)
        right_rad = math.radians(self.angle - 150)
        right_x = self.x + self.radius * 0.7 * math.cos(right_rad)
        right_y = self.y + self.radius * 0.7 * math.sin(right_rad)
        exhaust_rad = math.radians(self.angle + 180)
        exhaust_x = self.x + self.radius * 0.4 * math.cos(exhaust_rad)
        exhaust_y = self.y + self.radius * 0.4 * math.sin(exhaust_rad)
        return [(tip_x, tip_y), (left_x, left_y), (right_x, right_y), (exhaust_x, exhaust_y)]

    def update(self, keys_pressed):
        if not self.alive:
            return

        if keys_pressed["left"]:
            self.angle -= SHIP_ROTATION_SPEED
        if keys_pressed["right"]:
            self.angle += SHIP_ROTATION_SPEED

        self.thrust_on = keys_pressed["up"]
        if self.thrust_on:
            rad = math.radians(self.angle)
            self.vx += THRUST_POWER * math.cos(rad)
            self.vy += THRUST_POWER * math.sin(rad)

        self.vx *= FRICTION
        self.vy *= FRICTION

        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed > MAX_SPEED:
            self.vx = (self.vx / speed) * MAX_SPEED
            self.vy = (self.vy / speed) * MAX_SPEED

        self.x += self.vx
        self.y += self.vy
        self.x %= WINDOW_WIDTH
        self.y %= WINDOW_HEIGHT

        if self.shoot_timer > 0:
            self.shoot_timer -= 1
        if self.invincible > 0:
            self.invincible -= 1

    def shoot(self):
        if self.shoot_timer > 0 or not self.alive:
            return None
        self.shoot_timer = SHOOT_COOLDOWN
        tip_x, tip_y = self.get_tip()
        rad = math.radians(self.angle)
        vx = BULLET_SPEED * math.cos(rad) + self.vx
        vy = BULLET_SPEED * math.sin(rad) + self.vy
        return Bullet(tip_x, tip_y, vx, vy)

    def draw(self, canvas):
        if not self.alive:
            return
        if self.invincible > 0 and (self.invincible // 4) % 2 == 0:
            return

        points = self.get_shape_points()
        canvas.create_polygon(points[0][0], points[0][1],
                              points[1][0], points[1][1],
                              points[2][0], points[2][1],
                              outline=WHITE, fill="", width=2)

        if self.thrust_on:
            flame_len = random.randint(8, 18)
            rad = math.radians(self.angle + 180)
            flame_tip_x = points[3][0] + flame_len * math.cos(rad)
            flame_tip_y = points[3][1] + flame_len * math.sin(rad)
            fx1 = points[1][0] + (points[3][0] - points[1][0]) * 0.3
            fy1 = points[1][1] + (points[3][1] - points[1][1]) * 0.3
            fx2 = points[2][0] + (points[3][0] - points[2][0]) * 0.3
            fy2 = points[2][1] + (points[3][1] - points[2][1]) * 0.3
            canvas.create_polygon(fx1, fy1, flame_tip_x, flame_tip_y, fx2, fy2,
                                  fill=YELLOW, outline=ORANGE, width=1)


class Bullet:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.lifetime = BULLET_LIFETIME
        self.radius = 2

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.x %= WINDOW_WIDTH
        self.y %= WINDOW_HEIGHT
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, canvas):
        canvas.create_oval(self.x - self.radius, self.y - self.radius,
                           self.x + self.radius, self.y + self.radius,
                           fill=WHITE, outline="")


class Asteroid:
    def __init__(self, x=None, y=None, radius=ASTEROID_RADIUS_LARGE):
        if x is None:
            side = random.randint(0, 3)
            if side == 0:
                self.x = random.randint(0, WINDOW_WIDTH)
                self.y = -radius
            elif side == 1:
                self.x = WINDOW_WIDTH + radius
                self.y = random.randint(0, WINDOW_HEIGHT)
            elif side == 2:
                self.x = random.randint(0, WINDOW_WIDTH)
                self.y = WINDOW_HEIGHT + radius
            else:
                self.x = -radius
                self.y = random.randint(0, WINDOW_HEIGHT)
        else:
            self.x = x
            self.y = y

        self.radius = radius
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(ASTEROID_SPEED_MIN, ASTEROID_SPEED_MAX)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        self.rotation = random.uniform(-3, 3)
        self.current_rotation = random.uniform(0, 360)

        num_vertices = random.randint(ASTEROID_VERTICES_MIN, ASTEROID_VERTICES_MAX)
        self.vertices = []
        for i in range(num_vertices):
            a = (2 * math.pi / num_vertices) * i
            r = self.radius * random.uniform(0.7, 1.3)
            self.vertices.append((r * math.cos(a), r * math.sin(a)))

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.x %= WINDOW_WIDTH
        self.y %= WINDOW_HEIGHT
        self.current_rotation += self.rotation

    def get_world_vertices(self):
        rad = math.radians(self.current_rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        world_verts = []
        for lx, ly in self.vertices:
            rx = lx * cos_r - ly * sin_r
            ry = lx * sin_r + ly * cos_r
            world_verts.append((self.x + rx, self.y + ry))
        return world_verts

    def draw(self, canvas):
        world_verts = self.get_world_vertices()
        if len(world_verts) >= 3:
            coords = []
            for vx, vy in world_verts:
                coords.extend([vx, vy])
            canvas.create_polygon(*coords, outline=WHITE, fill="", width=2)

    def split(self):
        if self.radius <= ASTEROID_RADIUS_SMALL:
            return []
        new_radius = ASTEROID_RADIUS_MEDIUM if self.radius == ASTEROID_RADIUS_LARGE else ASTEROID_RADIUS_SMALL
        new_asteroids = []
        for _ in range(2):
            a = Asteroid(self.x, self.y, new_radius)
            a.vx += random.uniform(-1.5, 1.5)
            a.vy += random.uniform(-1.5, 1.5)
            new_asteroids.append(a)
        return new_asteroids

    def get_score_value(self):
        if self.radius == ASTEROID_RADIUS_LARGE:
            return SCORE_LARGE
        elif self.radius == ASTEROID_RADIUS_MEDIUM:
            return SCORE_MEDIUM
        else:
            return SCORE_SMALL


class Saucer:
    def __init__(self):
        self.reset()

    def reset(self):
        side = random.choice([0, 1])
        if side == 0:
            self.x = -SAUCER_SIZE
            self.vx = SAUCER_SPEED
        else:
            self.x = WINDOW_WIDTH + SAUCER_SIZE
            self.vx = -SAUCER_SPEED
        self.y = random.randint(SAUCER_SIZE, WINDOW_HEIGHT - SAUCER_SIZE)
        self.vy = random.uniform(-0.5, 0.5)
        self.alive = True
        self.shoot_timer = 0

    def update(self, ship_x, ship_y):
        if not self.alive:
            return None

        self.x += self.vx
        self.y += self.vy

        if self.y < SAUCER_SIZE or self.y > WINDOW_HEIGHT - SAUCER_SIZE:
            self.vy *= -1
        self.y = max(SAUCER_SIZE, min(WINDOW_HEIGHT - SAUCER_SIZE, self.y))

        if self.x < -SAUCER_SIZE * 2 or self.x > WINDOW_WIDTH + SAUCER_SIZE * 2:
            self.alive = False
            return None

        self.shoot_timer -= 1
        if self.shoot_timer <= 0:
            self.shoot_timer = SAUCER_SHOOT_COOLDOWN
            dx = ship_x - self.x
            dy = ship_y - self.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                inaccuracy = random.uniform(-0.3, 0.3)
                angle = math.atan2(dy, dx) + inaccuracy
                vx = SAUCER_BULLET_SPEED * math.cos(angle)
                vy = SAUCER_BULLET_SPEED * math.sin(angle)
                return Bullet(self.x, self.y, vx, vy)
        return None

    def draw(self, canvas):
        if not self.alive:
            return
        half_w = SAUCER_SIZE
        half_h = SAUCER_SIZE // 2
        # Top dome
        canvas.create_oval(self.x - half_w * 0.4, self.y - half_h,
                           self.x + half_w * 0.4, self.y,
                           outline=WHITE, fill="", width=2)
        # Bottom body
        canvas.create_oval(self.x - half_w, self.y - half_h * 0.5,
                           self.x + half_w, self.y + half_h * 0.5,
                           outline=WHITE, fill="", width=2)
        # Lights
        light_x = self.x - half_w * 0.3
        for _ in range(3):
            canvas.create_oval(light_x - 2, self.y + half_h * 0.3 - 2,
                               light_x + 2, self.y + half_h * 0.3 + 2,
                               fill=GREEN, outline="")
            light_x += half_w * 0.3


class Particle:
    def __init__(self, x, y, vx, vy, color, lifetime=20, size=2):
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
        self.vx *= 0.95
        self.vy *= 0.95
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, canvas):
        alpha = self.lifetime / self.max_lifetime
        size = max(1, int(self.size * alpha))
        # Fade by using a dimmer color
        r = int(255 * alpha)
        g = int(255 * alpha)
        b = int(255 * alpha)
        color = f"#{r:02x}{g:02x}{b:02x}"
        canvas.create_oval(self.x - size, self.y - size,
                           self.x + size, self.y + size,
                           fill=color, outline="")


class AsteroidsGame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ASTEROIDS")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
                                bg=BLACK, highlightthickness=0)
        self.canvas.pack()
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)

        self.keys_pressed = {"left": False, "right": False, "up": False, "space": False}

        self.font_large = ("Courier New", 48, "bold")
        self.font_medium = ("Courier New", 28, "bold")
        self.font_small = ("Courier New", 16)

        self.stars = []
        for _ in range(100):
            self.stars.append({
                'x': random.randint(0, WINDOW_WIDTH),
                'y': random.randint(0, WINDOW_HEIGHT),
                'brightness': random.randint(50, 200),
                'size': random.choice([1, 2])
            })

        self.reset_game()

    def on_key_press(self, event):
        if event.keysym == "Left":
            self.keys_pressed["left"] = True
        elif event.keysym == "Right":
            self.keys_pressed["right"] = True
        elif event.keysym == "Up":
            self.keys_pressed["up"] = True
        elif event.keysym == "space":
            self.keys_pressed["space"] = True
        elif event.keysym == "r":
            if self.state == STATE_GAME_OVER:
                self.start_game()
        elif event.keysym == "Escape":
            self.root.quit()

    def on_key_release(self, event):
        if event.keysym == "Left":
            self.keys_pressed["left"] = False
        elif event.keysym == "Right":
            self.keys_pressed["right"] = False
        elif event.keysym == "Up":
            self.keys_pressed["up"] = False
        elif event.keysym == "space":
            self.keys_pressed["space"] = False

    def reset_game(self):
        self.state = STATE_TITLE
        self.ship = Ship()
        self.asteroids = []
        self.bullets = []
        self.saucer_bullets = []
        self.particles = []
        self.saucer = None
        self.score = 0
        self.high_score = 0
        self.level = 1
        self.saucer_timer = SAUCER_SPAWN_INTERVAL
        self.lives = 3
        self.respawn_timer = 0

    def start_game(self):
        self.state = STATE_PLAYING
        self.ship = Ship()
        self.asteroids = []
        self.bullets = []
        self.saucer_bullets = []
        self.particles = []
        self.saucer = None
        self.score = 0
        self.level = 1
        self.lives = 3
        self.saucer_timer = SAUCER_SPAWN_INTERVAL
        self.respawn_timer = 0
        self.spawn_asteroids(4)

    def spawn_asteroids(self, count):
        for _ in range(count):
            while True:
                ast = Asteroid()
                dx = ast.x - self.ship.x
                dy = ast.y - self.ship.y
                if math.sqrt(dx ** 2 + dy ** 2) > 150:
                    break
            self.asteroids.append(ast)

    def spawn_explosion(self, x, y, count=20, color=WHITE):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            lifetime = random.randint(15, 35)
            size = random.randint(1, 4)
            self.particles.append(Particle(x, y, vx, vy, color, lifetime, size))

    def handle_collisions(self):
        # Bullets vs Asteroids
        for bullet in self.bullets[:]:
            for asteroid in self.asteroids[:]:
                dx = bullet.x - asteroid.x
                dy = bullet.y - asteroid.y
                if math.sqrt(dx ** 2 + dy ** 2) < asteroid.radius:
                    self.score += asteroid.get_score_value()
                    self.spawn_explosion(asteroid.x, asteroid.y, 15, WHITE)
                    new_asteroids = asteroid.split()
                    self.asteroids.remove(asteroid)
                    self.asteroids.extend(new_asteroids)
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break

        # Ship vs Asteroids
        if self.ship.alive and self.ship.invincible <= 0:
            for asteroid in self.asteroids[:]:
                dx = self.ship.x - asteroid.x
                dy = self.ship.y - asteroid.y
                if math.sqrt(dx ** 2 + dy ** 2) < asteroid.radius + self.ship.radius:
                    self.ship.alive = False
                    self.spawn_explosion(self.ship.x, self.ship.y, 30, WHITE)
                    self.lives -= 1
                    if self.lives > 0:
                        self.respawn_timer = 60
                    else:
                        self.state = STATE_GAME_OVER
                        if self.score > self.high_score:
                            self.high_score = self.score
                    break

        # Saucer bullets vs Ship
        if self.ship.alive and self.ship.invincible <= 0:
            for bullet in self.saucer_bullets[:]:
                dx = bullet.x - self.ship.x
                dy = bullet.y - self.ship.y
                if math.sqrt(dx ** 2 + dy ** 2) < self.ship.radius + bullet.radius:
                    self.ship.alive = False
                    self.spawn_explosion(self.ship.x, self.ship.y, 30, WHITE)
                    self.lives -= 1
                    self.saucer_bullets.remove(bullet)
                    if self.lives > 0:
                        self.respawn_timer = 60
                    else:
                        self.state = STATE_GAME_OVER
                        if self.score > self.high_score:
                            self.high_score = self.score
                    break

        # Bullets vs Saucer
        if self.saucer and self.saucer.alive:
            for bullet in self.bullets[:]:
                dx = bullet.x - self.saucer.x
                dy = bullet.y - self.saucer.y
                if math.sqrt(dx ** 2 + dy ** 2) < SAUCER_SIZE:
                    self.score += SAUCER_SCORE
                    self.spawn_explosion(self.saucer.x, self.saucer.y, 25, GREEN)
                    self.saucer.alive = False
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break

        # Ship vs Saucer
        if self.ship.alive and self.ship.invincible <= 0 and self.saucer and self.saucer.alive:
            dx = self.ship.x - self.saucer.x
            dy = self.ship.y - self.saucer.y
            if math.sqrt(dx ** 2 + dy ** 2) < SAUCER_SIZE + self.ship.radius:
                self.ship.alive = False
                self.spawn_explosion(self.ship.x, self.ship.y, 30, WHITE)
                self.spawn_explosion(self.saucer.x, self.saucer.y, 25, GREEN)
                self.saucer.alive = False
                self.lives -= 1
                if self.lives > 0:
                    self.respawn_timer = 60
                else:
                    self.state = STATE_GAME_OVER
                    if self.score > self.high_score:
                        self.high_score = self.score

    def update(self):
        if self.state == STATE_TITLE:
            if self.keys_pressed["space"]:
                self.start_game()
            return

        if self.state == STATE_GAME_OVER:
            return

        # Respawn ship
        if not self.ship.alive and self.lives > 0:
            if self.respawn_timer > 0:
                self.respawn_timer -= 1
            else:
                self.ship = Ship()
                self.ship.invincible = INVINCIBLE_FRAMES

        self.ship.update(self.keys_pressed)

        if self.keys_pressed["space"]:
            bullet = self.ship.shoot()
            if bullet:
                self.bullets.append(bullet)

        self.bullets = [b for b in self.bullets if b.update()]
        self.saucer_bullets = [b for b in self.saucer_bullets if b.update()]

        for asteroid in self.asteroids:
            asteroid.update()

        self.saucer_timer -= 1
        if self.saucer_timer <= 0 and self.saucer is None:
            self.saucer = Saucer()
            self.saucer_timer = SAUCER_SPAWN_INTERVAL

        if self.saucer:
            saucer_bullet = self.saucer.update(self.ship.x, self.ship.y)
            if saucer_bullet:
                self.saucer_bullets.append(saucer_bullet)
            if not self.saucer.alive:
                self.saucer = None

        self.particles = [p for p in self.particles if p.update()]

        self.handle_collisions()

        if len(self.asteroids) == 0:
            self.level += 1
            self.spawn_asteroids(min(4 + self.level, 12))

    def draw_starfield(self):
        for star in self.stars:
            c = star['brightness']
            color = f"#{c:02x}{c:02x}{c:02x}"
            self.canvas.create_oval(star['x'] - star['size'], star['y'] - star['size'],
                                    star['x'] + star['size'], star['y'] + star['size'],
                                    fill=color, outline="")

    def draw_hud(self):
        # Score
        self.canvas.create_text(20, 15, anchor="nw", text=f"SCORE: {self.score}",
                                fill=WHITE, font=self.font_small)
        # High score
        self.canvas.create_text(WINDOW_WIDTH - 20, 15, anchor="ne",
                                text=f"HIGH: {self.high_score}",
                                fill=DIM_WHITE, font=self.font_small)
        # Level
        self.canvas.create_text(WINDOW_WIDTH // 2, 15, anchor="n",
                                text=f"LEVEL: {self.level}",
                                fill=WHITE, font=self.font_small)
        # Lives
        for i in range(self.lives):
            x = 20 + i * 30
            y = 50
            self.canvas.create_polygon(x + 12, y, x, y + 20, x + 24, y + 20,
                                       outline=WHITE, fill="", width=2)

    def draw_title(self):
        self.canvas.delete("all")
        self.draw_starfield()

        self.canvas.create_text(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100,
                                text="ASTEROIDS", fill=WHITE, font=self.font_large)
        self.canvas.create_text(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2,
                                text="ARROW KEYS to move, SPACE to shoot",
                                fill=DIM_WHITE, font=self.font_small)
        self.canvas.create_text(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 30,
                                text="Destroy all asteroids. Watch for saucers!",
                                fill=DIM_WHITE, font=self.font_small)

        # Blinking start prompt using frame counter
        if not hasattr(self, '_blink_counter'):
            self._blink_counter = 0
        self._blink_counter += 1
        if self._blink_counter // 30 % 2 == 0:
            self.canvas.create_text(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 100,
                                    text="PRESS SPACE TO START",
                                    fill=WHITE, font=self.font_medium)

        if self.high_score > 0:
            self.canvas.create_text(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 160,
                                    text=f"HIGH SCORE: {self.high_score}",
                                    fill=YELLOW, font=self.font_small)

    def draw_game_over(self):
        self.canvas.create_rectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                                     fill=BLACK, stipple="gray25")
        self.canvas.create_text(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 60,
                                text="GAME OVER", fill=RED, font=self.font_large)
        self.canvas.create_text(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2,
                                text=f"SCORE: {self.score}",
                                fill=WHITE, font=self.font_medium)

        if self.score >= self.high_score and self.score > 0:
            self.canvas.create_text(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 40,
                                    text="NEW HIGH SCORE!", fill=YELLOW,
                                    font=self.font_small)

        self.canvas.create_text(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 90,
                                text="PRESS R TO RESTART", fill=DIM_WHITE,
                                font=self.font_small)
        self.canvas.create_text(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 120,
                                text="PRESS ESC TO QUIT", fill=DIM_WHITE,
                                font=self.font_small)

    def draw(self):
        self.canvas.delete("all")

        if self.state == STATE_TITLE:
            self.draw_title()
            return

        self.draw_starfield()

        for asteroid in self.asteroids:
            asteroid.draw(self.canvas)

        self.ship.draw(self.canvas)

        for bullet in self.bullets:
            bullet.draw(self.canvas)
        for bullet in self.saucer_bullets:
            bullet.draw(self.canvas)

        if self.saucer:
            self.saucer.draw(self.canvas)

        for particle in self.particles:
            particle.draw(self.canvas)

        self.draw_hud()

        if self.state == STATE_GAME_OVER:
            self.draw_game_over()

    def game_loop(self):
        self.update()
        self.draw()
        self.root.after(1000 // FPS, self.game_loop)

    def run(self):
        self.game_loop()
        self.root.mainloop()


if __name__ == "__main__":
    game = AsteroidsGame()
    game.run()