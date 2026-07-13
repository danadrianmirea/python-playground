import pygame, random, string
pygame.init()
s = pygame.display.set_mode((800,600))
f = pygame.font.Font(None, 20)

class Drop:
    def __init__(self, x):
        self.x = x
        self.y = random.randint(-200,-10)
        self.s = random.uniform(1,5)
        self.c = [random.choice(string.ascii_uppercase + "0123456789") for _ in range(random.randint(10,25))]
    def update(self):
        self.y += self.s
        if self.y > 600:
            self.__init__(self.x)
    def draw(self, s):
        for i, ch in enumerate(self.c):
            y = self.y + i*20
            if 0 <= y <= 600:
                b = max(0, 255 - i*20)
                s.blit(f.render(ch, True, (0,b,0)), (self.x, y))

d = [Drop(x) for x in range(0,800,20)]
r = True
while r:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): r = False
    s.fill((0,0,0))
    for drop in d: drop.update(); drop.draw(s)
    pygame.display.flip()
    pygame.time.Clock().tick(60)
pygame.quit()