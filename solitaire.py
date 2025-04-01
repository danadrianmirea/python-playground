import pygame
import os
import random
from typing import List, Tuple, Optional

# Initialize Pygame
pygame.init()

# Constants
CARD_WIDTH = 71
CARD_HEIGHT = 96
CARD_SPACING = 20
TABLEAU_SPACING = 100
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MENU_HEIGHT = 30
DEBUG = False  # Set to True to enable debug messages

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)

class Card:
    def __init__(self, suit: str, value: str, image_path: str):
        self.suit = suit
        self.value = value
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (CARD_WIDTH, CARD_HEIGHT))
        self.face_up = False
        self.rect = self.image.get_rect()
        
    def flip(self):
        self.face_up = not self.face_up
        
    def get_value(self) -> int:
        if self.value == 'ace':
            return 1
        elif self.value == 'jack':
            return 11
        elif self.value == 'queen':
            return 12
        elif self.value == 'king':
            return 13
        return int(self.value)
        
    def is_red(self) -> bool:
        return self.suit in ['hearts', 'diamonds']

class Menu:
    def __init__(self):
        self.font = pygame.font.Font(None, 24)
        self.menu_items = {
            'File': ['New Game', 'Quit']
        }
        self.active_menu = None
        self.menu_rects = {}
        self.item_rects = {}
        
    def draw(self, screen):
        # Draw menu bar background
        pygame.draw.rect(screen, GRAY, (0, 0, WINDOW_WIDTH, MENU_HEIGHT))
        
        # Draw menu items
        x = 10
        for menu_name in self.menu_items:
            text = self.font.render(menu_name, True, BLACK)
            rect = text.get_rect(topleft=(x, 5))
            screen.blit(text, rect)
            self.menu_rects[menu_name] = rect
            x += 100
            
        # Draw active menu if any
        if self.active_menu:
            menu_y = MENU_HEIGHT
            # Draw background for menu items
            menu_width = 150  # Width of the menu
            menu_height = len(self.menu_items[self.active_menu]) * 25
            pygame.draw.rect(screen, GRAY, (0, menu_y, menu_width, menu_height))
            
            # Draw menu items
            for item in self.menu_items[self.active_menu]:
                text = self.font.render(item, True, BLACK)
                rect = text.get_rect(topleft=(10, menu_y))
                screen.blit(text, rect)
                self.item_rects[item] = rect
                menu_y += 25
                
    def handle_click(self, pos):
        x, y = pos
        
        # Check if clicking on menu bar
        if y < MENU_HEIGHT:
            for menu_name, rect in self.menu_rects.items():
                if rect.collidepoint(pos):
                    self.active_menu = menu_name if self.active_menu != menu_name else None
                    return None
                    
        # Check if clicking on menu items
        if self.active_menu and y >= MENU_HEIGHT:
            # Check if click is within menu bounds
            menu_width = 150
            menu_height = len(self.menu_items[self.active_menu]) * 25
            if x < menu_width and y < MENU_HEIGHT + menu_height:
                for item, rect in self.item_rects.items():
                    if rect.collidepoint(pos):
                        return (self.active_menu, item)
            else:
                # Clicked outside menu, close it
                self.active_menu = None
                    
        return None

class Solitaire:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Solitaire")
        self.clock = pygame.time.Clock()
        self.menu = Menu()
        self.last_drawn_card = None  # Track the last drawn card
        self.reset_game()
        
    def reset_game(self):
        self.deck = []
        self.tableau_piles = [[] for _ in range(7)]
        self.foundation_piles = [[] for _ in range(4)]
        self.stock_pile = []
        self.waste_pile = []
        self.selected_card = None
        self.selected_pile = None
        self.selected_cards = []
        self.dragging = False
        self.drag_offset = (0, 0)
        self.original_positions = []
        self.load_cards()
        self.deal_cards()
        
    def load_cards(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        values = ['ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king']
        
        for suit in suits:
            for value in values:
                # Try regular filename first
                image_path = f"assets/cards/{value}_of_{suit}.png"
                # If not found, try with "2" suffix
                if not os.path.exists(image_path):
                    image_path = f"assets/cards/{value}_of_{suit}2.png"
                if os.path.exists(image_path):
                    self.deck.append(Card(suit, value, image_path))
        
        random.shuffle(self.deck)
        
    def deal_cards(self):
        # Deal cards to tableau piles
        for i in range(7):
            for j in range(i, 7):
                card = self.deck.pop()
                if i == j:
                    card.face_up = True
                self.tableau_piles[j].append(card)
        
        # Remaining cards go to stock pile
        self.stock_pile = self.deck
        
    def get_pile_at_pos(self, pos: Tuple[int, int]) -> Optional[Tuple[List[Card], int]]:
        x, y = pos
        
        # Adjust y position for menu bar
        y -= MENU_HEIGHT
        
        # Check tableau piles
        for i, pile in enumerate(self.tableau_piles):
            pile_x = 50 + i * TABLEAU_SPACING
            if pile_x <= x <= pile_x + CARD_WIDTH:
                # Check if there are any cards in the pile
                if pile:
                    pile_y = 150
                    for j, card in enumerate(pile):
                        if card.face_up:
                            card.rect.topleft = (pile_x, pile_y)
                            if card.rect.collidepoint(pos):
                                return pile, j
                        pile_y += CARD_SPACING
                else:
                    # Empty tableau pile - check if click is in the empty space
                    empty_rect = pygame.Rect(pile_x, 150, CARD_WIDTH, CARD_HEIGHT)
                    if empty_rect.collidepoint(pos):
                        return pile, -1
                    
        # Check foundation piles
        for i, pile in enumerate(self.foundation_piles):
            pile_x = 50 + i * TABLEAU_SPACING
            if pile_x <= x <= pile_x + CARD_WIDTH and 50 <= y <= 50 + CARD_HEIGHT:
                if pile:
                    pile[-1].rect.topleft = (pile_x, 50)
                    if pile[-1].rect.collidepoint(pos):
                        return pile, -1
                else:
                    # Empty foundation pile
                    empty_rect = pygame.Rect(pile_x, 50, CARD_WIDTH, CARD_HEIGHT)
                    if empty_rect.collidepoint(pos):
                        return pile, -1
                
        # Check stock pile
        if 50 <= x <= 50 + CARD_WIDTH and 400 <= y <= 400 + CARD_HEIGHT:
            # Create a rectangle for the stock pile
            stock_rect = pygame.Rect(50, 400, CARD_WIDTH, CARD_HEIGHT)
            if stock_rect.collidepoint(pos):
                return self.stock_pile, -1
            
        # Check waste pile
        if 150 <= x <= 150 + CARD_WIDTH and 400 <= y <= 400 + CARD_HEIGHT:
            if self.waste_pile:
                self.waste_pile[-1].rect.topleft = (150, 400)
                if self.waste_pile[-1].rect.collidepoint(pos):
                    return self.waste_pile, -1
            else:
                # Empty waste pile
                empty_rect = pygame.Rect(150, 400, CARD_WIDTH, CARD_HEIGHT)
                if empty_rect.collidepoint(pos):
                    return self.waste_pile, -1
            
        return None
        
    def can_move_to_tableau(self, card: Card, target_pile: List[Card]) -> bool:
        if DEBUG:
            print(f"Tableau move check - card: {card.value} ({card.get_value()}), target pile: {'empty' if not target_pile else target_pile[-1].value}")
        if not target_pile:
            return card.value == 'king'
        target_card = target_pile[-1]
        return (card.get_value() == target_card.get_value() - 1 and 
                card.is_red() != target_card.is_red())
                
    def can_move_to_foundation(self, card: Card, target_pile: List[Card]) -> bool:
        if not target_pile:
            return card.value == 'ace'
        target_card = target_pile[-1]
        return (card.suit == target_card.suit and 
                card.get_value() == target_card.get_value() + 1)
                
    def move_cards(self, source_pile: List[Card], target_pile: List[Card], 
                  start_index: int, end_index: Optional[int] = None) -> bool:
        if DEBUG:
            print(f"\nMove cards attempt:")
            print(f"Source pile: {source_pile[-1].value if source_pile else 'empty'}")
            print(f"Target pile: {target_pile[-1].value if target_pile else 'empty'}")
            print(f"Start index: {start_index}")
        
        if end_index is None:
            end_index = len(source_pile)
            
        cards_to_move = source_pile[start_index:end_index]
        if not cards_to_move:
            if DEBUG:
                print("No cards to move")
            return False
            
        if DEBUG:
            print(f"Moving card: {cards_to_move[0].value}")
            
        # Check if the move is valid
        if target_pile in self.tableau_piles:
            if DEBUG:
                print("Checking tableau move")
            if not self.can_move_to_tableau(cards_to_move[0], target_pile):
                if DEBUG:
                    print("Invalid tableau move")
                return False
        elif target_pile in self.foundation_piles:
            if DEBUG:
                print("Checking foundation move")
            if not self.can_move_to_foundation(cards_to_move[0], target_pile):
                if DEBUG:
                    print("Invalid foundation move")
                return False
                
        # Move the cards
        target_pile.extend(cards_to_move)
        source_pile[start_index:end_index] = []
        
        # Flip the top card of the source pile if it's face down
        if source_pile and not source_pile[-1].face_up:
            source_pile[-1].flip()
            
        if DEBUG:
            print("Move successful")
        return True
        
    def draw(self):
        self.screen.fill(GREEN)
        
        # Draw tableau piles
        for i, pile in enumerate(self.tableau_piles):
            x = 50 + i * TABLEAU_SPACING
            y = 150 + MENU_HEIGHT
            for j, card in enumerate(pile):
                # Skip drawing cards that are being dragged
                if self.dragging and self.selected_pile == pile and j >= self.selected_pile.index(self.selected_card):
                    continue
                    
                if card.face_up:
                    card.rect.topleft = (x, y)
                    self.screen.blit(card.image, card.rect)
                else:
                    # Draw card back
                    pygame.draw.rect(self.screen, WHITE, (x, y, CARD_WIDTH, CARD_HEIGHT))
                y += CARD_SPACING
                
        # Draw foundation piles
        for i in range(4):
            x = 50 + i * TABLEAU_SPACING
            y = 50 + MENU_HEIGHT
            if self.foundation_piles[i]:
                # Skip drawing the top card if it's being dragged
                if self.dragging and self.selected_pile == self.foundation_piles[i]:
                    if len(self.foundation_piles[i]) > 1:
                        card = self.foundation_piles[i][-2]
                        card.rect.topleft = (x, y)
                        self.screen.blit(card.image, card.rect)
                else:
                    card = self.foundation_piles[i][-1]
                    card.rect.topleft = (x, y)
                    self.screen.blit(card.image, card.rect)
            else:
                pygame.draw.rect(self.screen, WHITE, (x, y, CARD_WIDTH, CARD_HEIGHT))
                
        # Draw stock pile
        x = 50
        y = 400 + MENU_HEIGHT
        if self.stock_pile:
            pygame.draw.rect(self.screen, WHITE, (x, y, CARD_WIDTH, CARD_HEIGHT))
            
        # Draw waste pile
        x = 150
        y = 400 + MENU_HEIGHT
        if self.waste_pile:
            # Skip drawing the top card if it's being dragged
            if self.dragging and self.selected_pile == self.waste_pile:
                if len(self.waste_pile) > 1:
                    card = self.waste_pile[-2]
                    card.rect.topleft = (x, y)
                    self.screen.blit(card.image, card.rect)
            else:
                card = self.waste_pile[-1]
                card.rect.topleft = (x, y)
                self.screen.blit(card.image, card.rect)
            
        # Draw dragged cards
        if self.dragging and self.selected_card:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            x = mouse_x - self.drag_offset[0]
            y = mouse_y - self.drag_offset[1]
            
            # Draw all selected cards
            for i, card in enumerate(self.selected_cards):
                card.rect.topleft = (x, y + i * CARD_SPACING)
                self.screen.blit(card.image, card.rect)
        
        # Draw menu
        self.menu.draw(self.screen)
            
        pygame.display.flip()
        
    def handle_mouse_down(self, pos: Tuple[int, int]):
        # Check menu first
        menu_result = self.menu.handle_click(pos)
        if menu_result:
            menu_name, item = menu_result
            if menu_name == 'File':
                if item == 'New Game':
                    self.reset_game()
                elif item == 'Quit':
                    return False
            # Close the menu after selecting an option
            self.menu.active_menu = None
            return True
            
        # Adjust pos for menu bar
        adjusted_pos = (pos[0], pos[1] - MENU_HEIGHT)
        
        result = self.get_pile_at_pos(adjusted_pos)
        if not result:
            return True
            
        pile, index = result
        
        # Handle stock pile click
        if pile == self.stock_pile:
            if not self.stock_pile:
                # Move all cards from waste pile back to stock pile
                self.stock_pile = self.waste_pile[::-1]
                self.waste_pile = []
                for card in self.stock_pile:
                    card.face_up = False
                self.last_drawn_card = None
            else:
                # Draw a card from stock to waste
                self.last_drawn_card = self.stock_pile.pop()
                self.last_drawn_card.face_up = True
                self.waste_pile.append(self.last_drawn_card)
            return True
            
        # Handle card selection
        if (index >= 0 and pile[index].face_up) or (pile == self.waste_pile and pile):
            self.selected_card = pile[-1] if pile == self.waste_pile else pile[index]
            self.selected_pile = pile
            
            # For tableau piles, allow dragging multiple cards
            if pile in self.tableau_piles:
                self.selected_cards = pile[index:]
            else:
                # For waste and foundation piles, only allow dragging one card
                self.selected_cards = [self.selected_card]
                
            self.dragging = True
            
            # Calculate offset from mouse position to card position
            if pile in self.tableau_piles:
                pile_index = self.tableau_piles.index(pile)
                card_x = 50 + pile_index * TABLEAU_SPACING
                card_y = 150 + index * CARD_SPACING
            elif pile in self.foundation_piles:
                pile_index = self.foundation_piles.index(pile)
                card_x = 50 + pile_index * TABLEAU_SPACING
                card_y = 50
            else:  # waste pile
                card_x = 150
                card_y = 400
                
            self.drag_offset = (adjusted_pos[0] - card_x, adjusted_pos[1] - card_y)
            
        return True
            
    def handle_mouse_up(self, pos: Tuple[int, int]):
        if DEBUG:
            print("\nMouse up event:")
        if not self.dragging:
            if DEBUG:
                print("Not dragging, ignoring")
            return True
            
        # Adjust pos for menu bar
        adjusted_pos = (pos[0], pos[1] - MENU_HEIGHT)
        if DEBUG:
            print(f"Adjusted position: {adjusted_pos}")
        
        result = self.get_pile_at_pos(adjusted_pos)
        if result:
            pile, _ = result
            if DEBUG:
                print(f"Found pile at position: {pile}")
            if pile != self.selected_pile:
                if DEBUG:
                    print(f"Attempting to move from {self.selected_pile} to {pile}")
                if self.move_cards(self.selected_pile, pile, 
                                 self.selected_pile.index(self.selected_card)):
                    if DEBUG:
                        print("Move successful")
                    self.selected_card = None
                    self.selected_pile = None
                    self.selected_cards = []
                else:
                    if DEBUG:
                        print("Move failed, returning card to original position")
                    # Return cards to original position
                    if self.selected_pile in self.tableau_piles:
                        pile_index = self.tableau_piles.index(self.selected_pile)
                        card_index = self.selected_pile.index(self.selected_card)
                        x = 50 + pile_index * TABLEAU_SPACING
                        y = 150 + card_index * CARD_SPACING
                    elif self.selected_pile in self.foundation_piles:
                        pile_index = self.foundation_piles.index(self.selected_pile)
                        x = 50 + pile_index * TABLEAU_SPACING
                        y = 50
                    else:  # waste pile
                        x = 150
                        y = 400
                    self.selected_card.rect.topleft = (x, y)
        else:
            if DEBUG:
                print("No pile found at position")
        
        self.dragging = False
        self.selected_card = None
        self.selected_pile = None
        self.selected_cards = []
        return True
                
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        if not self.handle_mouse_down(event.pos):
                            running = False
                    elif event.button == 3:  # Right click
                        # Handle right click on stock pile
                        adjusted_pos = (event.pos[0], event.pos[1] - MENU_HEIGHT)
                        result = self.get_pile_at_pos(adjusted_pos)
                        if result:
                            pile, _ = result
                            if pile == self.stock_pile and self.last_drawn_card:
                                # Move the last drawn card back to stock pile
                                self.stock_pile.append(self.last_drawn_card)
                                self.waste_pile.pop()
                                self.last_drawn_card.face_up = False
                                self.last_drawn_card = None
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left click
                        if not self.handle_mouse_up(event.pos):
                            running = False
                    
            self.draw()
            self.clock.tick(60)
            
        pygame.quit()

if __name__ == "__main__":
    game = Solitaire()
    game.run()
