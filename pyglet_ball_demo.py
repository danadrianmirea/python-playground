import pyglet
import math

# Create a window
window = pyglet.window.Window(800, 600)

# Ball properties
ball_x = 400  # Start in center
ball_y = 300
ball_radius = 20
ball_speed = 5

# Create a circle shape
ball = pyglet.shapes.Circle(
    x=ball_x,
    y=ball_y,
    radius=ball_radius,
    color=(255, 0, 0)  # Red color
)

# Handle keyboard input
@window.event
def on_key_press(symbol, modifiers):
    global ball_x, ball_y
    
    # Calculate new position based on key pressed
    if symbol == pyglet.window.key.W:
        new_y = ball_y + ball_speed
        if new_y + ball_radius <= window.height:
            ball_y = new_y
    elif symbol == pyglet.window.key.S:
        new_y = ball_y - ball_speed
        if new_y - ball_radius >= 0:
            ball_y = new_y
    elif symbol == pyglet.window.key.A:
        new_x = ball_x - ball_speed
        if new_x - ball_radius >= 0:
            ball_x = new_x
    elif symbol == pyglet.window.key.D:
        new_x = ball_x + ball_speed
        if new_x + ball_radius <= window.width:
            ball_x = new_x

# Update ball position
@window.event
def on_draw():
    window.clear()
    ball.x = ball_x
    ball.y = ball_y
    ball.draw()

# Run the application
pyglet.app.run() 