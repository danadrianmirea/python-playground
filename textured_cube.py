import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
)

surfaces = (
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6)
)

TEXTURE="assets/texture.jpg"
texture_filenames = (TEXTURE, TEXTURE, TEXTURE, TEXTURE, TEXTURE, TEXTURE)

def LoadTexture():
    """Loads textures and returns a list of texture IDs."""
    texture_ids = glGenTextures(len(texture_filenames))

    for i, filename in enumerate(texture_filenames):
        texture_surface = pygame.image.load(filename)
        texture_data = pygame.image.tostring(texture_surface, "RGBA", 1)
        width = texture_surface.get_width()
        height = texture_surface.get_height()

        glBindTexture(GL_TEXTURE_2D, texture_ids[i])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

    return texture_ids

def Cube(texture_ids):
    """Draws the textured cube."""
    # Define texture coordinates for a proper square mapping
    texcoords = [
        (0.0, 0.0),  # Bottom-left
        (1.0, 0.0),  # Bottom-right
        (1.0, 1.0),  # Top-right
        (0.0, 1.0)   # Top-left
    ]
    
    for surface_index, surface in enumerate(surfaces):
        glBindTexture(GL_TEXTURE_2D, texture_ids[surface_index])
        glBegin(GL_QUADS)
        for vertex_index, vertex in enumerate(surface):
            glTexCoord2f(*texcoords[vertex_index])
            glVertex3fv(vertices[vertex])
        glEnd()

def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    
    # Initialize OpenGL state
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0,0.0, -5)

    texture_ids = LoadTexture() 

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube(texture_ids)
        pygame.display.flip()
        pygame.time.wait(10)

main()