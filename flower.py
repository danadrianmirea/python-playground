from turtle import *
import random

# List of color names
colors = ["red", "blue", "green", "yellow", "orange", "purple", 
          "pink", "cyan", "magenta", "lime", "teal", "coral"]

speed(3)
#tracer(0)  # Speed up drawing

while True:
    # Clear everything from previous drawing
    clear()
        
    # Draw flower
    for i in range(36):
        random_color = random.choice(colors)
        color(random_color)
        circle(100)
        right(10)
        
    
    update()  # Show the drawing
    
    # Wait a moment before next one
    import time
    time.sleep(1)

done()