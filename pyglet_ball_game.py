import pyglet
import math
import time

# Create a window
window = pyglet.window.Window(800, 600)

# Create custom crosshair cursor
def create_crosshair_cursor():
    size = 25
    center = size // 2
    gap = 2  # Size of the gap (in pixels)
    
    # Create raw pixel data (RGBA)
    data = []
    for y in range(size):
        for x in range(size):
            # Default to transparent black
            pixel = [0, 0, 0, 0]
            
            # Draw vertical line with gap
            if x == center and abs(y - center) > gap:
                pixel = [255, 255, 255, 255]  # White
                
            # Draw horizontal line with gap
            if y == center and abs(x - center) > gap:
                pixel = [255, 255, 255, 255]  # White
                
            # Add black pixel in the gap
            if (x == center and abs(y - center) <= gap) or (y == center and abs(x - center) <= gap):
                pixel = [0, 0, 0, 255]  # Black
                
            data.extend(pixel)
    
    image = pyglet.image.create(size, size)
    image.set_data('RGBA', size * 4, bytes(data))
    return pyglet.window.ImageMouseCursor(image, hot_x=center, hot_y=center)

cursor = create_crosshair_cursor()
window.set_mouse_cursor(cursor)

# Ball properties
ball_x = 400  # Start in center
ball_y = 300
ball_radius = 20
ball_speed = 300  # Speed in pixels per second

# Mouse position and state
mouse_x = 0
mouse_y = 0
mouse_lb_pressed = False

# Key states
key_states = {
    pyglet.window.key.W: False,
    pyglet.window.key.A: False,
    pyglet.window.key.S: False,
    pyglet.window.key.D: False
}

# Gun properties
current_angle = 0
edge_x = 0
edge_y = 0
gun_end_x = 0
gun_end_y = 0

# Bullet properties
bullets = []  # List to store active bullets
bullet_speed = 500  # Speed in pixels per second
bullet_radius = 4
fire_rate = 0.1  # Time between shots in seconds
last_shot_time = 0

class Bullet:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.shape = pyglet.shapes.Circle(
            x=x,
            y=y,
            radius=bullet_radius,
            color=(255, 255, 255)  # White color
        )
    
    def update(self, dt):
        self.x += math.cos(self.angle) * bullet_speed * dt
        self.y += math.sin(self.angle) * bullet_speed * dt
        self.shape.x = self.x
        self.shape.y = self.y
    
    def draw(self):
        self.shape.draw()
    
    def is_off_screen(self):
        return (self.x < 0 or self.x > window.width or 
                self.y < 0 or self.y > window.height)

# Create a circle shape
ball = pyglet.shapes.Circle(
    x=ball_x,
    y=ball_y,
    radius=ball_radius,
    color=(255, 0, 0)  # Red color
)

# Create gun line
gun_line = pyglet.shapes.Line(0, 0, 0, 0, color=(0, 255, 0))  # Green line for gun

@window.event
def on_mouse_motion(x, y, dx, dy):
    global mouse_x, mouse_y
    mouse_x = x
    mouse_y = y

@window.event
def on_mouse_press(x, y, button, modifiers):
    global last_shot_time, mouse_lb_pressed
    if button == pyglet.window.mouse.LEFT:
        mouse_lb_pressed = True
        last_shot_time = 0  # Reset the last shot time to fire immediately

@window.event
def on_mouse_release(x, y, button, modifiers):
    global mouse_lb_pressed
    if button == pyglet.window.mouse.LEFT:
        mouse_lb_pressed = False

@window.event
def on_key_press(symbol, modifiers):
    if symbol in key_states:
        key_states[symbol] = True

@window.event
def on_key_release(symbol, modifiers):
    if symbol in key_states:
        key_states[symbol] = False

def update(dt):
    global ball_x, ball_y, last_shot_time, current_angle, edge_x, edge_y, gun_end_x, gun_end_y
    
    # Handle movement (independent of mouse input)
    move_dx = 0
    move_dy = 0
    
    if key_states[pyglet.window.key.W]:
        move_dy += ball_speed * dt
    if key_states[pyglet.window.key.S]:
        move_dy -= ball_speed * dt
    if key_states[pyglet.window.key.A]:
        move_dx -= ball_speed * dt
    if key_states[pyglet.window.key.D]:
        move_dx += ball_speed * dt
        
    # Calculate new position
    new_x = ball_x + move_dx
    new_y = ball_y + move_dy
    
    # Check boundaries and update position
    if new_x - ball_radius >= 0 and new_x + ball_radius <= window.width:
        ball_x = new_x
    if new_y - ball_radius >= 0 and new_y + ball_radius <= window.height:
        ball_y = new_y
    
    # Update gun angle and position
    aim_dx = mouse_x - ball_x
    aim_dy = mouse_y - ball_y
    current_angle = math.atan2(aim_dy, aim_dx)
    
    # Calculate the point on the circle's edge where the gun should be
    edge_x = ball_x + ball_radius * math.cos(current_angle)
    edge_y = ball_y + ball_radius * math.sin(current_angle)
    
    # Calculate the end point of the gun line
    gun_end_x = edge_x + ball_radius * 0.6 * math.cos(current_angle)
    gun_end_y = edge_y + ball_radius * 0.6 * math.sin(current_angle)
    
    # Handle shooting (independent of movement)
    if mouse_lb_pressed:
        current_time = time.time()
        if current_time - last_shot_time >= fire_rate:
            # Create new bullet with current angle
            bullets.append(Bullet(edge_x, edge_y, current_angle))
            last_shot_time = current_time
    
    # Update bullets
    for bullet in bullets[:]:
        bullet.update(dt)
        if bullet.is_off_screen():
            bullets.remove(bullet)

@window.event
def on_draw():
    window.clear()
    
    # Update ball position
    ball.x = ball_x
    ball.y = ball_y
    ball.draw()
    
    # Draw gun line
    gun_line.x = edge_x
    gun_line.y = edge_y
    gun_line.x2 = gun_end_x
    gun_line.y2 = gun_end_y
    gun_line.draw()
    
    # Draw bullets
    for bullet in bullets:
        bullet.draw()

# Schedule the update function to be called every frame
pyglet.clock.schedule_interval(update, 1/60.0)  # 60 FPS

# Run the application
pyglet.app.run() 