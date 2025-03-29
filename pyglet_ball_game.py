import pyglet

# Create a window
window = pyglet.window.Window(800, 600)

# Ball properties
ball_x = 400  # Start in center
ball_y = 300
ball_radius = 20
ball_speed = 300  # Speed in pixels per second

# Create a circle shape
ball = pyglet.shapes.Circle(
    x=ball_x,
    y=ball_y,
    radius=ball_radius,
    color=(255, 0, 0)  # Red color
)

# Key state handler
keys = pyglet.window.key.KeyStateHandler()
window.push_handlers(keys)

def update(dt):
    global ball_x, ball_y
    
    # Calculate movement based on pressed keys
    dx = 0
    dy = 0
    
    if keys[pyglet.window.key.W]:
        dy += ball_speed * dt
    if keys[pyglet.window.key.S]:
        dy -= ball_speed * dt
    if keys[pyglet.window.key.A]:
        dx -= ball_speed * dt
    if keys[pyglet.window.key.D]:
        dx += ball_speed * dt
        
    # Calculate new position
    new_x = ball_x + dx
    new_y = ball_y + dy
    
    # Check boundaries and update position
    if new_x - ball_radius >= 0 and new_x + ball_radius <= window.width:
        ball_x = new_x
    if new_y - ball_radius >= 0 and new_y + ball_radius <= window.height:
        ball_y = new_y

@window.event
def on_draw():
    window.clear()
    ball.x = ball_x
    ball.y = ball_y
    ball.draw()

# Schedule the update function to be called every frame
pyglet.clock.schedule_interval(update, 1/60.0)  # 60 FPS

# Run the application
pyglet.app.run() 