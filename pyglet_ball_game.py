import pyglet
import math

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

# Mouse position
mouse_x = 0
mouse_y = 0

# Debug mode
debug_mode = False  # Always off for now

# Create a circle shape
ball = pyglet.shapes.Circle(
    x=ball_x,
    y=ball_y,
    radius=ball_radius,
    color=(255, 0, 0)  # Red color
)

# Create debug lines
debug_line_to_mouse = pyglet.shapes.Line(0, 0, 0, 0, color=(255, 255, 0))  # Yellow line
debug_line_to_gun = pyglet.shapes.Line(0, 0, 0, 0, color=(0, 255, 255))    # Cyan line
gun_line = pyglet.shapes.Line(0, 0, 0, 0, color=(0, 255, 0))              # Green line for gun

# Create debug circle for edge point
debug_edge_circle = pyglet.shapes.Circle(
    x=0, y=0,  # Position will be updated in draw
    radius=3,
    color=(255, 165, 0)  # Orange color
)

# Create debug label
debug_label = pyglet.text.Label('Debug Info',
                          font_name='Times New Roman',
                          font_size=12,
                          x=10, y=window.height - 20,
                          anchor_x='left', anchor_y='top')

# Key state handler
keys = pyglet.window.key.KeyStateHandler()
window.push_handlers(keys)

@window.event
def on_mouse_motion(x, y, dx, dy):
    global mouse_x, mouse_y
    mouse_x = x
    mouse_y = y

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
    
    # Update ball position
    ball.x = ball_x
    ball.y = ball_y
    ball.draw()
    
    # Calculate angle between ball and mouse
    dx = mouse_x - ball_x
    dy = mouse_y - ball_y
    angle = math.atan2(dy, dx)
    
    # Calculate the point on the circle's edge where the gun should be
    edge_x = ball_x + ball_radius * math.cos(angle)
    edge_y = ball_y + ball_radius * math.sin(angle)
    
    # Calculate the end point of the gun line
    gun_end_x = edge_x + ball_radius * 0.6 * math.cos(angle)
    gun_end_y = edge_y + ball_radius * 0.6 * math.sin(angle)
    
    # Draw gun line
    gun_line.x = edge_x
    gun_line.y = edge_y
    gun_line.x2 = gun_end_x
    gun_line.y2 = gun_end_y
    gun_line.draw()
    
    # Only show debug elements if debug mode is enabled
    if debug_mode:
        # Update debug lines
        debug_line_to_mouse.x = ball_x
        debug_line_to_mouse.y = ball_y
        debug_line_to_mouse.x2 = mouse_x
        debug_line_to_mouse.y2 = mouse_y
        debug_line_to_mouse.draw()
        
        debug_line_to_gun.x = ball_x
        debug_line_to_gun.y = ball_y
        debug_line_to_gun.x2 = edge_x
        debug_line_to_gun.y2 = edge_y
        debug_line_to_gun.draw()
        
        # Draw debug circle at edge point
        debug_edge_circle.x = edge_x
        debug_edge_circle.y = edge_y
        debug_edge_circle.draw()
        
        # Update debug text
        debug_text = f'Angle: {math.degrees(angle):.1f}°\n'
        debug_text += f'Gun pos: ({edge_x:.1f}, {edge_y:.1f})\n'
        debug_text += f'Mouse pos: ({mouse_x:.1f}, {mouse_y:.1f})'
        debug_label.text = debug_text
        debug_label.draw()

# Schedule the update function to be called every frame
pyglet.clock.schedule_interval(update, 1/60.0)  # 60 FPS

# Run the application
pyglet.app.run() 