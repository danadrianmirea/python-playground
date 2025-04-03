import pygame
import os
import random
import json
from typing import List, Tuple, Optional

# Initialize Pygame
pygame.init()

# Constants
DEBUG = True  # Set to True to enable debug messages
SAVE_FILE = "solitaire_save.json"

# Get native resolution
NATIVE_WIDTH, NATIVE_HEIGHT = pygame.display.get_desktop_sizes()[0]

# Base constants (original game size)
BASE_CARD_WIDTH = 71
BASE_CARD_HEIGHT = 96
BASE_CARD_SPACING = 20
BASE_TABLEAU_SPACING = 100
BASE_WINDOW_WIDTH = 800
BASE_WINDOW_HEIGHT = 600
BASE_MENU_HEIGHT = 30

# Calculate scaling factor based on height (to maintain aspect ratio)
# Add a small margin (0.85) to account for Windows decorations and taskbar
SCALE_FACTOR = min(NATIVE_HEIGHT / BASE_WINDOW_HEIGHT, NATIVE_WIDTH / BASE_WINDOW_WIDTH) * 0.85

# Scaled constants
CARD_WIDTH = int(BASE_CARD_WIDTH * SCALE_FACTOR)
CARD_HEIGHT = int(BASE_CARD_HEIGHT * SCALE_FACTOR)
CARD_SPACING = int(BASE_CARD_SPACING * SCALE_FACTOR)
TABLEAU_SPACING = int(BASE_TABLEAU_SPACING * SCALE_FACTOR)
WINDOW_WIDTH = int(BASE_WINDOW_WIDTH * SCALE_FACTOR)
WINDOW_HEIGHT = int(BASE_WINDOW_HEIGHT * SCALE_FACTOR)
MENU_HEIGHT = int(BASE_MENU_HEIGHT * SCALE_FACTOR)

# Center the window on the screen
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{(NATIVE_WIDTH - WINDOW_WIDTH) // 2},{(NATIVE_HEIGHT - WINDOW_HEIGHT) // 2}"

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (180, 180, 180)

# Load card back image
CARD_BACK = pygame.image.load("assets/cards/card_back_red.png")
CARD_BACK = pygame.transform.scale(CARD_BACK, (CARD_WIDTH, CARD_HEIGHT))

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
        self.font = pygame.font.Font(None, int(24 * SCALE_FACTOR))
        self.menu_items = {
            'File': ['New Game', 'Save Game', 'Load Game', 'Quit']
        }
        self.active_menu = None
        self.menu_rects = {}
        self.item_rects = {}
        self.padding = int(10 * SCALE_FACTOR)  # Padding around menu items
        
    def draw(self, screen):
        # Draw menu bar background
        pygame.draw.rect(screen, GRAY, (0, 0, WINDOW_WIDTH, MENU_HEIGHT))
        
        # Draw menu items
        x = self.padding
        for menu_name in self.menu_items:
            text = self.font.render(menu_name, True, BLACK)
            text_rect = text.get_rect(topleft=(x, int(5 * SCALE_FACTOR)))
            
            # Draw menu item background with padding
            menu_rect = pygame.Rect(
                x - self.padding,
                0,
                text_rect.width + 2 * self.padding,
                MENU_HEIGHT
            )
            pygame.draw.rect(screen, GRAY, menu_rect)
            pygame.draw.rect(screen, BLACK, menu_rect, 1)  # Add border
            
            screen.blit(text, text_rect)
            self.menu_rects[menu_name] = menu_rect
            x += text_rect.width + 2 * self.padding + int(20 * SCALE_FACTOR)  # Add spacing between menu items
            
        # Draw active menu if any
        if self.active_menu:
            menu_y = MENU_HEIGHT
            # Calculate menu width based on longest item
            max_width = 0
            for item in self.menu_items[self.active_menu]:
                text = self.font.render(item, True, BLACK)
                max_width = max(max_width, text.get_width())
            
            # Add padding to width
            menu_width = max_width + 2 * self.padding
            menu_height = len(self.menu_items[self.active_menu]) * (int(25 * SCALE_FACTOR) + 2 * self.padding)
            
            # Draw menu background
            menu_rect = pygame.Rect(0, menu_y, menu_width, menu_height)
            pygame.draw.rect(screen, GRAY, menu_rect)
            pygame.draw.rect(screen, BLACK, menu_rect, 1)  # Add border
            
            # Draw menu items with padding
            item_y = menu_y + self.padding
            for item in self.menu_items[self.active_menu]:
                text = self.font.render(item, True, BLACK)
                text_rect = text.get_rect(topleft=(self.padding, item_y))
                
                # Draw item background with padding
                item_rect = pygame.Rect(
                    self.padding,
                    item_y - self.padding,
                    menu_width - 2 * self.padding,
                    int(25 * SCALE_FACTOR) + 2 * self.padding
                )
                pygame.draw.rect(screen, GRAY, item_rect)
                
                screen.blit(text, text_rect)
                self.item_rects[item] = item_rect
                item_y += int(25 * SCALE_FACTOR) + 2 * self.padding
                
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
            max_width = 0
            for item in self.menu_items[self.active_menu]:
                text = self.font.render(item, True, BLACK)
                max_width = max(max_width, text.get_width())
            
            menu_width = max_width + 2 * self.padding
            menu_height = len(self.menu_items[self.active_menu]) * (int(25 * SCALE_FACTOR) + 2 * self.padding)
            
            if x < menu_width and y < MENU_HEIGHT + menu_height:
                for item, rect in self.item_rects.items():
                    if rect.collidepoint(pos):
                        return (self.active_menu, item)
            else:
                # Clicked outside menu, close it
                self.active_menu = None
                    
        return None

class PopupDialog:
    def __init__(self, message: str, yes_text: str = "Yes", no_text: str = "No"):
        self.font = pygame.font.Font(None, int(32 * SCALE_FACTOR))
        self.button_font = pygame.font.Font(None, int(24 * SCALE_FACTOR))
        self.message = message
        self.yes_text = yes_text
        self.no_text = no_text
        
        # Calculate popup dimensions
        self.width = int(400 * SCALE_FACTOR)
        self.height = int(200 * SCALE_FACTOR)
        self.x = (WINDOW_WIDTH - self.width) // 2
        self.y = (WINDOW_HEIGHT - self.height) // 2
        
        # Create button rectangles
        button_width = int(100 * SCALE_FACTOR)
        button_height = int(40 * SCALE_FACTOR)
        button_y = self.y + self.height - button_height - int(20 * SCALE_FACTOR)
        
        self.yes_button = pygame.Rect(
            self.x + self.width//2 - button_width - int(20 * SCALE_FACTOR),
            button_y,
            button_width,
            button_height
        )
        
        self.no_button = pygame.Rect(
            self.x + self.width//2 + int(20 * SCALE_FACTOR),
            button_y,
            button_width,
            button_height
        )
        
    def draw(self, screen):
        # Draw semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.fill(BLACK)
        overlay.set_alpha(128)
        screen.blit(overlay, (0, 0))
        
        # Draw popup background
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)
        
        # Draw message
        text = self.font.render(self.message, True, BLACK)
        text_rect = text.get_rect(center=(self.x + self.width//2, self.y + self.height//3))
        screen.blit(text, text_rect)
        
        # Draw buttons
        pygame.draw.rect(screen, LIGHT_GRAY, self.yes_button)
        pygame.draw.rect(screen, LIGHT_GRAY, self.no_button)
        pygame.draw.rect(screen, BLACK, self.yes_button, 1)
        pygame.draw.rect(screen, BLACK, self.no_button, 1)
        
        yes_text = self.button_font.render(self.yes_text, True, BLACK)
        no_text = self.button_font.render(self.no_text, True, BLACK)
        
        screen.blit(yes_text, yes_text.get_rect(center=self.yes_button.center))
        screen.blit(no_text, no_text.get_rect(center=self.no_button.center))
        
    def handle_click(self, pos):
        if self.yes_button.collidepoint(pos):
            return "yes"
        elif self.no_button.collidepoint(pos):
            return "no"
        return None

class Solitaire:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Solitaire")
        self.clock = pygame.time.Clock()
        self.menu = Menu()
        self.last_drawn_card = None  # Track the last drawn card
        self.last_click_time = 0  # Track last click time for double-click detection
        self.popup = None  # Track current popup dialog
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
            pile_x = int(50 * SCALE_FACTOR) + i * TABLEAU_SPACING
            if pile_x <= x <= pile_x + CARD_WIDTH:
                # Check if there are any cards in the pile
                if pile:
                    pile_y = int(110 * SCALE_FACTOR)  # Adjusted from 100 to 110
                    for j, card in enumerate(pile):
                        if card.face_up:
                            card.rect.topleft = (pile_x, pile_y)
                            if card.rect.collidepoint(pos):
                                return pile, j
                        pile_y += CARD_SPACING
                else:
                    # Empty tableau pile - check if click is in the empty space
                    empty_rect = pygame.Rect(pile_x, int(110 * SCALE_FACTOR), CARD_WIDTH, CARD_HEIGHT)  # Adjusted from 100 to 110
                    if empty_rect.collidepoint(pos):
                        return pile, -1
                    
        # Check foundation piles
        for i, pile in enumerate(self.foundation_piles):
            pile_x = int(50 * SCALE_FACTOR) + i * TABLEAU_SPACING
            if pile_x <= x <= pile_x + CARD_WIDTH and int(10 * SCALE_FACTOR) <= y <= int(10 * SCALE_FACTOR) + CARD_HEIGHT:  # Adjusted from 50 to 10
                if pile:
                    pile[-1].rect.topleft = (pile_x, int(10 * SCALE_FACTOR))  # Adjusted from 50 to 10
                    if pile[-1].rect.collidepoint(pos):
                        return pile, -1
                else:
                    # Empty foundation pile
                    empty_rect = pygame.Rect(pile_x, int(10 * SCALE_FACTOR), CARD_WIDTH, CARD_HEIGHT)  # Adjusted from 50 to 10
                    if empty_rect.collidepoint(pos):
                        return pile, -1
                
        # Check stock pile
        if int(50 * SCALE_FACTOR) <= x <= int(50 * SCALE_FACTOR) + CARD_WIDTH and int(450 * SCALE_FACTOR) <= y <= int(450 * SCALE_FACTOR) + CARD_HEIGHT:  # Adjusted from 400 to 450
            # Create a rectangle for the stock pile
            stock_rect = pygame.Rect(int(50 * SCALE_FACTOR), int(450 * SCALE_FACTOR), CARD_WIDTH, CARD_HEIGHT)  # Adjusted from 400 to 450
            if stock_rect.collidepoint(pos):
                return self.stock_pile, -1
            
        # Check waste pile
        if int(150 * SCALE_FACTOR) <= x <= int(150 * SCALE_FACTOR) + CARD_WIDTH and int(450 * SCALE_FACTOR) <= y <= int(450 * SCALE_FACTOR) + CARD_HEIGHT:  # Adjusted from 400 to 450
            if self.waste_pile:
                self.waste_pile[-1].rect.topleft = (int(150 * SCALE_FACTOR), int(450 * SCALE_FACTOR))  # Adjusted from 400 to 450
                if self.waste_pile[-1].rect.collidepoint(pos):
                    return self.waste_pile, -1
            else:
                # Empty waste pile
                empty_rect = pygame.Rect(int(150 * SCALE_FACTOR), int(450 * SCALE_FACTOR), CARD_WIDTH, CARD_HEIGHT)  # Adjusted from 400 to 450
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
        if DEBUG:
            print(f"Foundation move check - card: {card.value} ({card.get_value()}), target pile: {'empty' if not target_pile else target_pile[-1].value}")
        if not target_pile:
            return card.value == 'ace'  # Any Ace can be placed on an empty foundation pile
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
            
        # Check if the move is valid based on card type
        card = cards_to_move[0]
        if card.value == 'ace':
            # For Aces, check foundation piles first
            if target_pile in self.foundation_piles:
                if DEBUG:
                    print("Checking foundation move for Ace")
                if self.can_move_to_foundation(card, target_pile):
                    if DEBUG:
                        print("Valid foundation move for Ace")
                    # Move the cards
                    target_pile.extend(cards_to_move)
                    source_pile[start_index:end_index] = []
                    
                    # Flip the top card of the source pile if it's face down
                    if source_pile and not source_pile[-1].face_up:
                        source_pile[-1].flip()
                        
                    if DEBUG:
                        print("Move successful")
                    return True
            # If not a valid foundation move, check tableau
            elif target_pile in self.tableau_piles:
                if DEBUG:
                    print("Checking tableau move for Ace")
                if not self.can_move_to_tableau(card, target_pile):
                    if DEBUG:
                        print("Invalid tableau move")
                    return False
        elif card.value == 'king':
            # For Kings, check tableau piles first
            if target_pile in self.tableau_piles:
                if DEBUG:
                    print("Checking tableau move for King")
                # Kings can only be placed on empty tableau piles
                if not target_pile:
                    if DEBUG:
                        print("Valid tableau move for King (empty pile)")
                    # Move the cards
                    target_pile.extend(cards_to_move)
                    source_pile[start_index:end_index] = []
                    
                    # Flip the top card of the source pile if it's face down
                    if source_pile and not source_pile[-1].face_up:
                        source_pile[-1].flip()
                        
                    if DEBUG:
                        print("Move successful")
                    return True
                else:
                    if DEBUG:
                        print("Invalid tableau move for King (non-empty pile)")
                    return False
            # If not a valid tableau move, check foundation
            elif target_pile in self.foundation_piles:
                if DEBUG:
                    print("Checking foundation move for King")
                if not self.can_move_to_foundation(card, target_pile):
                    if DEBUG:
                        print("Invalid foundation move")
                    return False
        else:
            # For all other cards, check both in order
            if target_pile in self.foundation_piles:
                if DEBUG:
                    print("Checking foundation move")
                if not self.can_move_to_foundation(card, target_pile):
                    if DEBUG:
                        print("Invalid foundation move")
                    return False
            elif target_pile in self.tableau_piles:
                if DEBUG:
                    print("Checking tableau move")
                if not self.can_move_to_tableau(card, target_pile):
                    if DEBUG:
                        print("Invalid tableau move")
                    return False
                
        # Move the cards
        target_pile.extend(cards_to_move)
        source_pile[start_index:end_index] = []
        
        # Flip the top card of the source pile if it's face down
        if source_pile and not source_pile[-1].face_up:
            source_pile[-1].flip()
            
        if DEBUG:
            print("Move successful")
            
        # Check for win condition after moving a card to foundation
        if target_pile in self.foundation_piles and self.check_win():
            self.show_win_popup()
            
        return True
        
    def draw(self):
        self.screen.fill(GREEN)
        
        # Draw tableau piles
        for i, pile in enumerate(self.tableau_piles):
            x = int(50 * SCALE_FACTOR) + i * TABLEAU_SPACING
            y = int(110 * SCALE_FACTOR) + MENU_HEIGHT  # Adjusted from 100 to 110
            for j, card in enumerate(pile):
                # Skip drawing cards that are being dragged
                if self.dragging and self.selected_pile == pile and j >= self.selected_pile.index(self.selected_card):
                    continue
                    
                if card.face_up:
                    card.rect.topleft = (x, y)
                    self.screen.blit(card.image, card.rect)
                else:
                    # Draw card back
                    self.screen.blit(CARD_BACK, (x, y))
                y += CARD_SPACING
                
        # Draw foundation piles
        for i in range(4):
            x = int(50 * SCALE_FACTOR) + i * TABLEAU_SPACING
            y = int(10 * SCALE_FACTOR) + MENU_HEIGHT  # Adjusted from 50 to 10
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
        x = int(50 * SCALE_FACTOR)
        y = int(450 * SCALE_FACTOR) + MENU_HEIGHT  # Adjusted from 400 to 450
        if self.stock_pile:
            self.screen.blit(CARD_BACK, (x, y))
            
        # Draw waste pile
        x = int(150 * SCALE_FACTOR)
        y = int(450 * SCALE_FACTOR) + MENU_HEIGHT  # Adjusted from 400 to 450
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
        
        # Draw popup if active
        if self.popup:
            self.popup.draw(self.screen)
            
        pygame.display.flip()
        
    def save_game(self):
        """Save the current game state to a file."""
        game_state = {
            'tableau_piles': [[{'suit': card.suit, 'value': card.value, 'face_up': card.face_up} 
                              for card in pile] for pile in self.tableau_piles],
            'foundation_piles': [[{'suit': card.suit, 'value': card.value, 'face_up': card.face_up} 
                                for card in pile] for pile in self.foundation_piles],
            'stock_pile': [{'suit': card.suit, 'value': card.value, 'face_up': card.face_up} 
                          for card in self.stock_pile],
            'waste_pile': [{'suit': card.suit, 'value': card.value, 'face_up': card.face_up} 
                          for card in self.waste_pile],
            'last_drawn_card': {'suit': self.last_drawn_card.suit, 'value': self.last_drawn_card.value} 
                              if self.last_drawn_card else None
        }
        
        try:
            with open(SAVE_FILE, 'w') as f:
                json.dump(game_state, f)
            if DEBUG:
                print("Game saved successfully")
        except Exception as e:
            if DEBUG:
                print(f"Error saving game: {e}")

    def load_game(self):
        """Load a saved game state from a file."""
        try:
            if not os.path.exists(SAVE_FILE):
                if DEBUG:
                    print("No save file found")
                return False
                
            with open(SAVE_FILE, 'r') as f:
                game_state = json.load(f)
                
            # Clear all piles without resetting the game
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
            
            # Load tableau piles
            for i, pile_data in enumerate(game_state['tableau_piles']):
                for card_data in pile_data:
                    image_path = f"assets/cards/{card_data['value']}_of_{card_data['suit']}.png"
                    if not os.path.exists(image_path):
                        image_path = f"assets/cards/{card_data['value']}_of_{card_data['suit']}2.png"
                    card = Card(card_data['suit'], card_data['value'], image_path)
                    card.face_up = card_data['face_up']
                    self.tableau_piles[i].append(card)
                    
            # Load foundation piles
            for i, pile_data in enumerate(game_state['foundation_piles']):
                for card_data in pile_data:
                    image_path = f"assets/cards/{card_data['value']}_of_{card_data['suit']}.png"
                    if not os.path.exists(image_path):
                        image_path = f"assets/cards/{card_data['value']}_of_{card_data['suit']}2.png"
                    card = Card(card_data['suit'], card_data['value'], image_path)
                    card.face_up = card_data['face_up']
                    self.foundation_piles[i].append(card)
                    
            # Load stock pile
            for card_data in game_state['stock_pile']:
                image_path = f"assets/cards/{card_data['value']}_of_{card_data['suit']}.png"
                if not os.path.exists(image_path):
                    image_path = f"assets/cards/{card_data['value']}_of_{card_data['suit']}2.png"
                card = Card(card_data['suit'], card_data['value'], image_path)
                card.face_up = card_data['face_up']
                self.stock_pile.append(card)
                
            # Load waste pile
            for card_data in game_state['waste_pile']:
                image_path = f"assets/cards/{card_data['value']}_of_{card_data['suit']}.png"
                if not os.path.exists(image_path):
                    image_path = f"assets/cards/{card_data['value']}_of_{card_data['suit']}2.png"
                card = Card(card_data['suit'], card_data['value'], image_path)
                card.face_up = card_data['face_up']
                self.waste_pile.append(card)
                
            # Load last drawn card
            if game_state['last_drawn_card']:
                card_data = game_state['last_drawn_card']
                image_path = f"assets/cards/{card_data['value']}_of_{card_data['suit']}.png"
                if not os.path.exists(image_path):
                    image_path = f"assets/cards/{card_data['value']}_of_{card_data['suit']}2.png"
                self.last_drawn_card = Card(card_data['suit'], card_data['value'], image_path)
                
            if DEBUG:
                print("Game loaded successfully")
            return True
            
        except Exception as e:
            if DEBUG:
                print(f"Error loading game: {e}")
            return False

    def handle_mouse_down(self, pos: Tuple[int, int]):
        # Only proceed with card handling if not clicking on menu
        if pos[1] < MENU_HEIGHT:
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
            # For tableau piles, only select the clicked card and cards below it
            if pile in self.tableau_piles:
                # Find the actual card that was clicked by checking the y-position
                pile_x = int(50 * SCALE_FACTOR) + self.tableau_piles.index(pile) * TABLEAU_SPACING
                clicked_y = adjusted_pos[1]
                base_y = int(110 * SCALE_FACTOR)
                
                # Calculate which card was actually clicked based on y-position
                clicked_index = (clicked_y - base_y) // CARD_SPACING
                if clicked_index < 0:
                    clicked_index = 0
                if clicked_index >= len(pile):
                    clicked_index = len(pile) - 1
                    
                # Only select if the card at this position is face up
                if pile[clicked_index].face_up:
                    self.selected_card = pile[clicked_index]
                    self.selected_pile = pile
                    self.selected_cards = pile[clicked_index:]
                    self.dragging = True
                    
                    # Calculate offset from mouse position to card position
                    card_x = pile_x
                    card_y = base_y + clicked_index * CARD_SPACING
                    self.drag_offset = (adjusted_pos[0] - card_x, adjusted_pos[1] - card_y)
            else:
                # For waste and foundation piles, only allow dragging one card
                self.selected_card = pile[-1]
                self.selected_pile = pile
                self.selected_cards = [self.selected_card]
                self.dragging = True
                
                # Calculate offset from mouse position to card position
                if pile in self.foundation_piles:
                    pile_index = self.foundation_piles.index(pile)
                    card_x = int(50 * SCALE_FACTOR) + pile_index * TABLEAU_SPACING
                    card_y = int(10 * SCALE_FACTOR)
                else:  # waste pile
                    card_x = int(150 * SCALE_FACTOR)
                    card_y = int(450 * SCALE_FACTOR)
                    
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
                        x = int(50 * SCALE_FACTOR) + pile_index * TABLEAU_SPACING
                        y = int(110 * SCALE_FACTOR) + card_index * CARD_SPACING  # Adjusted from 100 to 110
                    elif self.selected_pile in self.foundation_piles:
                        pile_index = self.foundation_piles.index(self.selected_pile)
                        x = int(50 * SCALE_FACTOR) + pile_index * TABLEAU_SPACING
                        y = int(10 * SCALE_FACTOR)
                    else:  # waste pile
                        x = int(150 * SCALE_FACTOR)
                        y = int(450 * SCALE_FACTOR)
                    self.selected_card.rect.topleft = (x, y)
        else:
            if DEBUG:
                print("No pile found at position")
        
        self.dragging = False
        self.selected_card = None
        self.selected_pile = None
        self.selected_cards = []
        return True

    def find_valid_foundation_pile(self, card: Card) -> Optional[List[Card]]:
        """Find a valid foundation pile for the given card."""
        if DEBUG:
            print(f"\nLooking for valid foundation pile for {card.value} of {card.suit}")
            
        for pile in self.foundation_piles:
            if self.can_move_to_foundation(card, pile):
                if DEBUG:
                    print(f"Found valid foundation pile with {'empty' if not pile else pile[-1].value}")
                return pile
        return None

    def handle_double_click(self, pos: Tuple[int, int]) -> bool:
        """Handle double-click events to move cards to foundation piles."""
        current_time = pygame.time.get_ticks()
        
        # Check if this is a double-click (within 500ms of last click)
        if current_time - self.last_click_time < 500:
            # Adjust pos for menu bar
            adjusted_pos = (pos[0], pos[1] - MENU_HEIGHT)
            
            result = self.get_pile_at_pos(adjusted_pos)
            if result:
                pile, index = result
                # Handle waste pile differently since it's a single-card display
                if pile == self.waste_pile and pile:
                    card = pile[-1]
                    if DEBUG:
                        print(f"\nDouble-click detected on waste pile card: {card.value} of {card.suit}")
                    
                    # Try to find a valid foundation pile
                    target_pile = self.find_valid_foundation_pile(card)
                    if target_pile:
                        if DEBUG:
                            print("Moving card to foundation pile")
                        # Move the card to the foundation pile
                        if self.move_cards(pile, target_pile, -1):
                            # Reset the last click time after successful move
                            self.last_click_time = 0
                            return True
                    else:
                        if DEBUG:
                            print("No valid foundation pile found")
                # Handle other piles (tableau and foundation)
                elif index >= 0 and pile[index].face_up:
                    # For tableau piles, calculate the exact card clicked based on y-position
                    if pile in self.tableau_piles:
                        pile_x = int(50 * SCALE_FACTOR) + self.tableau_piles.index(pile) * TABLEAU_SPACING
                        clicked_y = adjusted_pos[1]
                        base_y = int(110 * SCALE_FACTOR)
                        
                        # Calculate which card was actually clicked based on y-position
                        clicked_index = (clicked_y - base_y) // CARD_SPACING
                        if clicked_index < 0:
                            clicked_index = 0
                        if clicked_index >= len(pile):
                            clicked_index = len(pile) - 1
                            
                        # Only proceed if we clicked on the last face-up card
                        last_face_up_index = -1
                        for i, card in enumerate(pile):
                            if card.face_up:
                                last_face_up_index = i
                        if clicked_index != last_face_up_index:
                            return False
                            
                        card = pile[clicked_index]
                    else:
                        card = pile[index]
                        
                    if DEBUG:
                        print(f"\nDouble-click detected on {card.value} of {card.suit}")
                    
                    # Try to find a valid foundation pile
                    target_pile = self.find_valid_foundation_pile(card)
                    if target_pile:
                        if DEBUG:
                            print("Moving card to foundation pile")
                        # Move the card to the foundation pile
                        if self.move_cards(pile, target_pile, pile.index(card)):
                            # Reset the last click time after successful move
                            self.last_click_time = 0
                            return True
                    else:
                        if DEBUG:
                            print("No valid foundation pile found")
            
            # Reset the last click time
            self.last_click_time = 0
            return False
        else:
            # Update the last click time
            self.last_click_time = current_time
            return False

    def check_win(self) -> bool:
        """Check if the game has been won (all foundation piles have a king)."""
        for pile in self.foundation_piles:
            if not pile or pile[-1].value != 'king':
                return False
        return True

    def show_win_popup(self):
        """Show the win popup dialog."""
        self.popup = PopupDialog("Congratulations! You won!", "Play Again", "Quit")

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Handle popup if active
                        if self.popup:
                            result = self.popup.handle_click(event.pos)
                            if result == "yes":
                                self.reset_game()
                                self.popup = None
                            elif result == "no":
                                running = False
                            continue
                            
                        # First check for menu clicks
                        menu_result = self.menu.handle_click(event.pos)
                        if menu_result:
                            menu_name, item = menu_result
                            if menu_name == 'File':
                                if item == 'New Game':
                                    self.reset_game()
                                elif item == 'Save Game':
                                    self.save_game()
                                elif item == 'Load Game':
                                    self.load_game()
                                elif item == 'Quit':
                                    running = False
                            # Close the menu after selecting an option
                            self.menu.active_menu = None
                            continue
                            
                        # Then check for double-click
                        if self.handle_double_click(event.pos):
                            continue
                            
                        # Handle single click
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
