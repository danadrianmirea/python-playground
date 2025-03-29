import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Define the vertices of the cube
vertices = (
    # Front face
    (-1, -1,  1),  # 0
    ( 1, -1,  1),  # 1
    ( 1,  1,  1),  # 2
    (-1,  1,  1),  # 3
    # Back face
    (-1, -1, -1),  # 4
    ( 1, -1, -1),  # 5
    ( 1,  1, -1),  # 6
    (-1,  1, -1)   # 7
)

# Define the edges of the cube
edges = (
    # Front face
    (0, 1), (1, 2), (2, 3), (3, 0),
    # Back face
    (4, 5), (5, 6), (6, 7), (7, 4),
    # Connecting edges
    (0, 4), (1, 5), (2, 6), (3, 7)
)

def draw_cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        glRotatef(1, 3, 1, 1)  # Rotate the cube
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_cube()
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
