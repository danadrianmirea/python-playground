import pygame
import random
import sys

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CARD_WIDTH = 70
CARD_HEIGHT = 85
CARD_MARGIN = 10
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
BLUE = (70, 130, 180)
LIGHT_BLUE = (173, 216, 230)
GREEN = (60, 179, 113)
RED = (220, 80, 80)
GOLD = (255, 215, 0)
BG_COLOR = (25, 25, 50)
CARD_BACK_COLOR = (50, 50, 120)
CARD_FRONT_COLOR = (240, 240, 250)
HIGHLIGHT_COLOR = (255, 255, 100)

class Card:
    def __init__(self, value, x, y, width, height):
        self.value = value
        self.rect = pygame.Rect(x, y, width, height)
        self.width = width
        self.height = height
        self.revealed = False
        self.matched = False
        self.flip_progress = 0.0
        self.flipping = False

    def update(self):
        if self.flipping:
            self.flip_progress += 0.15
            if self.flip_progress >= 1.0:
                self.flip_progress = 1.0
                self.flipping = False
                self.revealed = True

    def start_flip(self):
        if not self.flipping and not self.revealed:
            self.flipping = True

    def draw(self, screen, font, highlight=False):
        if self.revealed or self.matched:
            scale_x = 1.0
        elif self.flipping:
            scale_x = abs(1.0 - self.flip_progress * 2)
            if scale_x > 1.0:
                scale_x = 2.0 - scale_x
        else:
            scale_x = 1.0

        card_width = max(4, int(self.width * scale_x))
        card_x = self.rect.centerx - card_width // 2
        card_rect = pygame.Rect(card_x, self.rect.y, card_width, self.height)

        # Shadow
        shadow_rect = card_rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        pygame.draw.rect(screen, (0, 0, 0, 80), shadow_rect, border_radius=6)

        show_front = (self.revealed or self.matched or
                      (self.flipping and self.flip_progress >= 0.5))

        if show_front:
            color = GOLD if self.matched else CARD_FRONT_COLOR
            pygame.draw.rect(screen, color, card_rect, border_radius=6)
            pygame.draw.rect(screen, DARK_GRAY, card_rect, 2, border_radius=6)
            if card_width > 24:
                text = font.render(str(self.value), True, BLACK)
                text_rect = text.get_rect(center=card_rect.center)
                screen.blit(text, text_rect)
        else:
            pygame.draw.rect(screen, CARD_BACK_COLOR, card_rect, border_radius=6)
            pygame.draw.rect(screen, DARK_GRAY, card_rect, 2, border_radius=6)
            if card_width > 16:
                inner = card_rect.inflate(-12, -12)
                pygame.draw.rect(screen, (70, 70, 150), inner, border_radius=4)
                pygame.draw.rect(screen, DARK_GRAY, inner, 1, border_radius=4)

        if highlight:
            pygame.draw.rect(screen, HIGHLIGHT_COLOR, card_rect, 3, border_radius=6)

class MemoryPuzzle:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Memory Puzzle")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        self.font_card = pygame.font.Font(None, 40)
        self.running = True
        self.state = "menu"
        self.grid_size = 4  # 4x4, 6x6, etc.
        self.cards = []
        self.selected = []
        self.attempts = 0
        self.matches = 0
        self.can_flip = True
        self.message = ""
        self.message_timer = 0
        self.checking_match = False

    def create_board(self):
        self.cards = []
        self.selected = []
        self.attempts = 0
        self.matches = 0
        self.can_flip = True
        self.message = ""
        self.message_timer = 0
        self.checking_match = False

        total_cards = self.grid_size * self.grid_size
        pair_count = total_cards // 2
        values = list(range(1, pair_count + 1)) * 2
        random.shuffle(values)

        # Calculate card size to fit the grid
        available_width = WINDOW_WIDTH - 80
        available_height = WINDOW_HEIGHT - 120
        card_w = (available_width - (self.grid_size - 1) * CARD_MARGIN) // self.grid_size
        card_h = (available_height - (self.grid_size - 1) * CARD_MARGIN) // self.grid_size
        card_size = min(card_w, card_h, 80)

        total_width = self.grid_size * card_size + (self.grid_size - 1) * CARD_MARGIN
        total_height = self.grid_size * card_size + (self.grid_size - 1) * CARD_MARGIN
        start_x = (WINDOW_WIDTH - total_width) // 2
        start_y = (WINDOW_HEIGHT - total_height) // 2 + 20

        # Adjust font size based on card size
        font_px = max(18, min(40, card_size // 2))
        self.font_card = pygame.font.Font(None, font_px)

        for i, value in enumerate(values):
            col = i % self.grid_size
            row = i // self.grid_size
            x = start_x + col * (card_size + CARD_MARGIN)
            y = start_y + row * (card_size + CARD_MARGIN)
            self.cards.append(Card(value, x, y, card_size, card_size))

    def handle_menu_click(self, pos):
        grid_sizes = [(4, 100), (6, 200), (8, 300)]
        for size, y_pos in grid_sizes:
            btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, y_pos, 200, 50)
            if btn_rect.collidepoint(pos):
                self.grid_size = size
                self.create_board()
                self.state = "playing"
                return

    def handle_game_click(self, pos):
        if not self.can_flip or len(self.selected) >= 2:
            return

        for card in self.cards:
            if card.rect.collidepoint(pos) and not card.revealed and not card.matched:
                if card not in self.selected:
                    card.start_flip()
                    self.selected.append(card)
                    if len(self.selected) == 2:
                        self.can_flip = False
                    break

    def check_match(self):
        if len(self.selected) == 2:
            c1, c2 = self.selected
            if c1.value == c2.value:
                c1.matched = True
                c2.matched = True
                self.matches += 1
                self.message = "Match found!"
                self.message_timer = pygame.time.get_ticks()
                self.selected = []
                self.can_flip = True
                self.checking_match = False

                total_pairs = (self.grid_size * self.grid_size) // 2
                if self.matches == total_pairs:
                    self.state = "game_over"
            else:
                self.message = "No match!"
                self.message_timer = pygame.time.get_ticks()
                pygame.time.set_timer(pygame.USEREVENT, 800)

    def flip_back(self):
        for card in self.selected:
            card.revealed = False
            card.flip_progress = 0.0
        self.selected = []
        self.can_flip = True
        self.checking_match = False

    def draw_menu(self):
        self.screen.fill(BG_COLOR)

        title = self.font_large.render("MEMORY PUZZLE", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 50))
        self.screen.blit(title, title_rect)

        subtitle = self.font_small.render("Select grid size:", True, GRAY)
        sub_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 100))
        self.screen.blit(subtitle, sub_rect)

        grid_sizes = [(4, 180), (6, 280), (8, 380)]
        for size, y_pos in grid_sizes:
            btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, y_pos, 200, 50)
            mouse_pos = pygame.mouse.get_pos()
            hover = btn_rect.collidepoint(mouse_pos)
            color = LIGHT_BLUE if hover else BLUE
            pygame.draw.rect(self.screen, color, btn_rect, border_radius=10)
            pygame.draw.rect(self.screen, WHITE, btn_rect, 2, border_radius=10)

            pairs = (size * size) // 2
            text = self.font_medium.render(f"{size}x{size}  ({pairs} pairs)", True, WHITE)
            text_rect = text.get_rect(center=btn_rect.center)
            self.screen.blit(text, text_rect)

        instr = self.font_small.render("Click on cards to flip them. Match all pairs to win!", True, GRAY)
        instr_rect = instr.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 60))
        self.screen.blit(instr, instr_rect)

    def draw_game(self):
        self.screen.fill(BG_COLOR)

        total_pairs = (self.grid_size * self.grid_size) // 2
        header_text = f"Pairs: {self.matches}/{total_pairs}  |  Attempts: {self.attempts}"
        header = self.font_medium.render(header_text, True, WHITE)
        header_rect = header.get_rect(center=(WINDOW_WIDTH // 2, 25))
        self.screen.blit(header, header_rect)

        for card in self.cards:
            highlight = card in self.selected
            card.draw(self.screen, self.font_card, highlight)

        if self.message and pygame.time.get_ticks() - self.message_timer < 1000:
            msg_color = GREEN if "Match" in self.message else RED
            msg = self.font_medium.render(self.message, True, msg_color)
            msg_rect = msg.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
            bg_rect = msg_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect, border_radius=5)
            self.screen.blit(msg, msg_rect)

    def draw_game_over(self):
        self.screen.fill(BG_COLOR)

        for card in self.cards:
            card.draw(self.screen, self.font_card)

        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        total_pairs = (self.grid_size * self.grid_size) // 2

        congrats = self.font_large.render("CONGRATULATIONS!", True, GOLD)
        congrats_rect = congrats.get_rect(center=(WINDOW_WIDTH // 2, 120))
        self.screen.blit(congrats, congrats_rect)

        stats = self.font_medium.render(f"You matched all {total_pairs} pairs!", True, WHITE)
        stats_rect = stats.get_rect(center=(WINDOW_WIDTH // 2, 180))
        self.screen.blit(stats, stats_rect)

        attempts_text = self.font_medium.render(f"Total attempts: {self.attempts}", True, WHITE)
        attempts_rect = attempts_text.get_rect(center=(WINDOW_WIDTH // 2, 230))
        self.screen.blit(attempts_text, attempts_rect)

        optimal = total_pairs
        if self.attempts <= optimal:
            rating = "Perfect!"
        elif self.attempts <= optimal * 2:
            rating = "Excellent memory!"
        elif self.attempts <= optimal * 3:
            rating = "Good job!"
        elif self.attempts <= optimal * 4:
            rating = "Not bad."
        else:
            rating = "Keep practicing!"

        rating_text = self.font_large.render(f"Rating: {rating}", True, GOLD)
        rating_rect = rating_text.get_rect(center=(WINDOW_WIDTH // 2, 300))
        self.screen.blit(rating_text, rating_rect)

        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 370, 200, 50)
        mouse_pos = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mouse_pos)
        color = LIGHT_BLUE if hover else BLUE
        pygame.draw.rect(self.screen, color, btn_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, btn_rect, 2, border_radius=10)
        again_text = self.font_medium.render("Play Again", True, WHITE)
        again_rect = again_text.get_rect(center=btn_rect.center)
        self.screen.blit(again_text, again_rect)

        menu_btn = pygame.Rect(WINDOW_WIDTH // 2 - 100, 440, 200, 50)
        hover2 = menu_btn.collidepoint(mouse_pos)
        color2 = LIGHT_BLUE if hover2 else DARK_GRAY
        pygame.draw.rect(self.screen, color2, menu_btn, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, menu_btn, 2, border_radius=10)
        menu_text = self.font_medium.render("Main Menu", True, WHITE)
        menu_rect = menu_text.get_rect(center=menu_btn.center)
        self.screen.blit(menu_text, menu_rect)

        return btn_rect, menu_btn

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.USEREVENT:
                    self.flip_back()
                    pygame.time.set_timer(pygame.USEREVENT, 0)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.state == "menu":
                        self.handle_menu_click(event.pos)
                    elif self.state == "playing":
                        self.handle_game_click(event.pos)
                    elif self.state == "game_over":
                        btn_rect, menu_btn = self.draw_game_over()
                        if btn_rect.collidepoint(event.pos):
                            self.create_board()
                            self.state = "playing"
                        elif menu_btn.collidepoint(event.pos):
                            self.state = "menu"

            if self.state == "playing":
                for card in self.cards:
                    card.update()

                if not self.can_flip and len(self.selected) == 2 and not self.checking_match:
                    all_flipped = all(c.flip_progress >= 1.0 for c in self.selected)
                    if all_flipped:
                        self.checking_match = True
                        self.attempts += 1
                        self.check_match()

            if self.state == "menu":
                self.draw_menu()
            elif self.state == "playing":
                self.draw_game()
            elif self.state == "game_over":
                self.draw_game_over()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = MemoryPuzzle()
    game.run()