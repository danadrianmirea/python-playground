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

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)

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
        if self.value == 'A':
            return 1
        elif self.value == 'J':
            return 11
        elif self.value == 'Q':
            return 12
        elif self.value == 'K':
            return 13
        return int(self.value)
        
    def is_red(self) -> bool:
        return self.suit in ['hearts', 'diamonds']

class Solitaire:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Solitaire")
        self.clock = pygame.time.Clock()
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
        values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        for suit in suits:
            for value in values:
                image_path = f"assets/cards/{value}_of_{suit}.png"
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
        
        # Check tableau piles
        for i, pile in enumerate(self.tableau_piles):
            pile_x = 50 + i * TABLEAU_SPACING
            if pile_x <= x <= pile_x + CARD_WIDTH:
                pile_y = 150
                for j, card in enumerate(pile):
                    if card.face_up:
                        card.rect.topleft = (pile_x, pile_y)
                        if card.rect.collidepoint(pos):
                            return pile, j
                    pile_y += CARD_SPACING
                    
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
        if not target_pile:
            return card.value == 'K'
        target_card = target_pile[-1]
        return (card.get_value() == target_card.get_value() - 1 and 
                card.is_red() != target_card.is_red())
                
    def can_move_to_foundation(self, card: Card, target_pile: List[Card]) -> bool:
        if not target_pile:
            return card.value == 'A'
        target_card = target_pile[-1]
        return (card.suit == target_card.suit and 
                card.get_value() == target_card.get_value() + 1)
                
    def move_cards(self, source_pile: List[Card], target_pile: List[Card], 
                  start_index: int, end_index: Optional[int] = None) -> bool:
        if end_index is None:
            end_index = len(source_pile)
            
        cards_to_move = source_pile[start_index:end_index]
        if not cards_to_move:
            return False
            
        # Check if the move is valid
        if target_pile in self.foundation_piles:
            if not self.can_move_to_foundation(cards_to_move[0], target_pile):
                return False
        elif target_pile in self.tableau_piles:
            if not self.can_move_to_tableau(cards_to_move[0], target_pile):
                return False
                
        # Move the cards
        target_pile.extend(cards_to_move)
        source_pile[start_index:end_index] = []
        
        # Flip the top card of the source pile if it's face down
        if source_pile and not source_pile[-1].face_up:
            source_pile[-1].flip()
            
        return True
        
    def draw(self):
        self.screen.fill(GREEN)
        
        # Draw tableau piles
        for i, pile in enumerate(self.tableau_piles):
            x = 50 + i * TABLEAU_SPACING
            y = 150
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
            y = 50
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
        y = 400
        if self.stock_pile:
            pygame.draw.rect(self.screen, WHITE, (x, y, CARD_WIDTH, CARD_HEIGHT))
            
        # Draw waste pile
        x = 150
        y = 400
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
            
        pygame.display.flip()
        
    def handle_mouse_down(self, pos: Tuple[int, int]):
        result = self.get_pile_at_pos(pos)
        if not result:
            return
            
        pile, index = result
        
        # Handle stock pile click
        if pile == self.stock_pile:
            if not self.stock_pile:
                # Move all cards from waste pile back to stock pile
                self.stock_pile = self.waste_pile[::-1]
                self.waste_pile = []
                for card in self.stock_pile:
                    card.face_up = False
            else:
                # Draw a card from stock to waste
                card = self.stock_pile.pop()
                card.face_up = True
                self.waste_pile.append(card)
            return
            
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
                
            self.drag_offset = (pos[0] - card_x, pos[1] - card_y)
            
    def handle_mouse_up(self, pos: Tuple[int, int]):
        if not self.dragging:
            return
            
        result = self.get_pile_at_pos(pos)
        if result:
            pile, _ = result
            if pile != self.selected_pile:
                if self.move_cards(self.selected_pile, pile, 
                                 self.selected_pile.index(self.selected_card)):
                    self.selected_card = None
                    self.selected_pile = None
                    self.selected_cards = []
                else:
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
        
        self.dragging = False
        self.selected_card = None
        self.selected_pile = None
        self.selected_cards = []
                
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left click
                        self.handle_mouse_up(event.pos)
                    
            self.draw()
            self.clock.tick(60)
            
        pygame.quit()

if __name__ == "__main__":
    game = Solitaire()
    game.run()
