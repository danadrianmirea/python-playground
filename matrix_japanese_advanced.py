import pygame
import random

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Matrix - Full Japanese")

# Comprehensive Japanese character set
KATAKANA = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
HIRAGANA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
KANJI = "日月火水木金土山川田人生気心花風雨空海"
NUMBERS = "０１２３４５６７８９"

ALL_CHARS = KATAKANA + HIRAGANA + KANJI + NUMBERS

try:
    font = pygame.font.SysFont("MS Gothic", 24)
except:
    try:
        font = pygame.font.SysFont("Hiragino Sans", 24)
    except:
        font = pygame.font.Font(None, 26)

class JapaneseDrop:
    def __init__(self, x):
        self.x = x
        self.y = random.randint(-300, -10)
        self.speed = random.uniform(0.5, 4)
        self.length = random.randint(12, 28)
        self.chars = []
        for _ in range(self.length):
            # Mix different character types
            if random.random() < 0.6:
                self.chars.append(random.choice(KATAKANA))
            elif random.random() < 0.8:
                self.chars.append(random.choice(HIRAGANA))
            else:
                self.chars.append(random.choice(KANJI))
    
    def update(self):
        self.y += self.speed
        if self.y > HEIGHT + 50:
            self.__init__(self.x)
    
    def draw(self, screen):
        char_height = 24
        for i, char in enumerate(self.chars):
            y_pos = self.y + i * char_height
            if 0 <= y_pos <= HEIGHT:
                # Fade effect
                if i == 0:
                    color = (0, 255, 0)
                elif i < 3:
                    color = (0, 255 - (i * 30), 0)
                else:
                    brightness = max(0, 255 - (i * 12))
                    color = (0, brightness, 0)
                
                # Random bright sparkle
                if random.random() < 0.003:
                    color = (0, 255, 200)
                
                text = font.render(char, True, color)
                screen.blit(text, (self.x, y_pos))

# Create drops
drops = [JapaneseDrop(x) for x in range(0, WIDTH, 24)]

# Background static text (optional)
def draw_background_code():
    for y in range(0, HEIGHT, 30):
        for x in range(0, WIDTH, 30):
            if random.random() < 0.02:
                char = random.choice(KATAKANA + NUMBERS)
                text = font.render(char, True, (0, 20, 0))
                screen.blit(text, (x, y))

clock = pygame.time.Clock()
running = True
show_background = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                drops = [JapaneseDrop(x) for x in range(0, WIDTH, 24)]
            if event.key == pygame.K_b:
                show_background = not show_background
    
    screen.fill((0, 0, 0))
    
    # Optional background code
    if show_background:
        draw_background_code()
    
    # Update and draw drops
    for drop in drops:
        drop.update()
        drop.draw(screen)
    
    # Display info
    info = font.render("SPACE: Reset  |  B: Background  |  ESC: Quit", True, (0, 50, 0))
    screen.blit(info, (10, HEIGHT - 30))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()