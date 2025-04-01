import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

degreesPerSecond = 90

def draw_triangle():
    # Draw a triangle
    glBegin(GL_TRIANGLES)
    
    # Set the first vertex color and position
    glColor3f(1.0, 0.0, 0.0)  # Red
    glVertex3f(-0.5, -0.5, 0.0)

    # Set the second vertex color and position
    glColor3f(0.0, 1.0, 0.0)  # Green
    glVertex3f(0.5, -0.5, 0.0)

    # Set the third vertex color and position
    glColor3f(0.0, 0.0, 1.0)  # Blue
    glVertex3f(0.0, 0.5, 0.0)

    glEnd()

def main():
    pygame.init()

    # Set up the display using pygame
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("OpenGL Triangle with Pygame")

    # Initialize OpenGL viewport and projection
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
    glViewport(0, 0, 800, 600)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (800 / 600), 0.1, 50.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -2.0)  # Move the triangle into view

    angle = 0  # Initialize rotation angle
    clock = pygame.time.Clock()  # Initialize pygame clock for delta time calculation

    # Main loop
    running = True
    while running:
        delta_time = clock.tick(60) / 1000.0  # Calculate delta time in seconds

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Apply rotation
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -2.0)
        glRotatef(angle, 0, 1, 0)  # Rotate around the Y-axis

        # Draw the triangle
        draw_triangle()

        # Update the angle for the next frame using delta time
        angle += degreesPerSecond * delta_time  # Rotate 90 degrees per second
        if(angle > 360):
            angle -= 360

        # Swap buffers to display the rendered image
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
