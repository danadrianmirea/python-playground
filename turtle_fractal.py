import turtle
import random

# Set up the turtle
screen = turtle.Screen()
screen.bgcolor("black")
t = turtle.Turtle()
t.speed(0)
t.hideturtle()

# Set a starting color for the tree (RGB values from 0.0 to 1.0)
t.color((0, 0.5, 0))  # start green

def tree(length):
    """Draw a fractal tree with a trunk of the given length."""
    if length > 5:
        # Increase randomness for natural look
        t.width(length / 20)

        # Draw the trunk
        t.forward(length)

        # Right branch
        t.right(random.uniform(20, 40))
        tree(length * random.uniform(0.6, 0.8))

        # Left branch
        t.left(random.uniform(40, 80))
        tree(length * random.uniform(0.6, 0.8))

        # Return to the original position and angle
        t.right(random.uniform(20, 40))
        t.back(length)

# Start drawing with a trunk length of 100
t.penup()
t.goto(0, -250)
t.setheading(90)  # point upwards
t.pendown()
tree(100)

screen.exitonclick()