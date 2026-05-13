import pygame
import os
import random
from typing import List, Tuple, Optional

# Initialize Pygame
pygame.init()

# Constants
DEBUG = True  # Set to True to enable debug messages

# Get native resolution
NATIVE_WIDTH, NATIVE_HEIGHT = pygame.display.get_desktop_sizes()[0]

# Base constants (original game size)
BASE_CARD_WIDTH = 71
BASE_CARD_HEIGHT = 96
BASE_WINDOW_WIDTH = 800
BASE_WINDOW_HEIGHT = 600
BASE_MENU_HEIGHT = 30

# Calculate scaling factor based on height (to maintain aspect ratio)
# Add a small margin (0.85) to account for Windows decorations and taskbar
SCALE_FACTOR = min(NATIVE_HEIGHT / BASE_WINDOW_HEIGHT, NATIVE_WIDTH / BASE_WINDOW_WIDTH) * 0.85

# Scaled constants
CARD_WIDTH = int(BASE_CARD_WIDTH * SCALE_FACTOR)
CARD_HEIGHT = int(BASE_CARD_HEIGHT * SCALE_FACTOR)
WINDOW_WIDTH = int(BASE_WINDOW_WIDTH * SCALE_FACTOR)
WINDOW_HEIGHT = int(BASE_WINDOW_HEIGHT * SCALE_FACTOR)
MENU_HEIGHT = int(BASE_MENU_HEIGHT * SCALE_FACTOR)

# Center the window on the screen
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{(NATIVE_WIDTH - WINDOW_WIDTH) // 2},{(NATIVE_HEIGHT - WINDOW_HEIGHT) // 2}"

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
DARK_GREEN = (0, 100, 0)
RED = (255, 0, 0)
GOLD = (255, 215, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (180, 180, 180)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Load card back image
CARD_BACK = pygame.image.load("assets/cards/card_back_red.png")
CARD_BACK = pygame.transform.scale(CARD_BACK, (CARD_WIDTH, CARD_HEIGHT))

# Fonts
FONT_LARGE = pygame.font.Font(None, int(48 * SCALE_FACTOR))
FONT_MEDIUM = pygame.font.Font(None, int(32 * SCALE_FACTOR))
FONT_SMALL = pygame.font.Font(None, int(24 * SCALE_FACTOR))


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

    def get_blackjack_value(self) -> int:
        """Get the card value for Blackjack."""
        if self.value in ['jack', 'queen', 'king']:
            return 10
        elif self.value == 'ace':
            return 11  # Aces are 11 by default, adjusted later if needed
        return int(self.value)

    def get_display_name(self) -> str:
        """Get a display-friendly name for the card."""
        return f"{self.value.capitalize()} of {self.suit.capitalize()}"


class Deck:
    def __init__(self, num_decks: int = 6):
        self.cards: List[Card] = []
        self.num_decks = num_decks
        self.load_cards()
        self.shuffle()

    def load_cards(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        values = ['ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king']

        for _ in range(self.num_decks):
            for suit in suits:
                for value in values:
                    # Try regular filename first
                    image_path = f"assets/cards/{value}_of_{suit}.png"
                    # If not found, try with "2" suffix
                    if not os.path.exists(image_path):
                        image_path = f"assets/cards/{value}_of_{suit}2.png"
                    if os.path.exists(image_path):
                        self.cards.append(Card(suit, value, image_path))

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self) -> Card:
        if not self.cards:
            if DEBUG:
                print("Deck is empty, reshuffling...")
            self.load_cards()
            self.shuffle()
        return self.cards.pop()

    def remaining(self) -> int:
        return len(self.cards)


class Hand:
    def __init__(self):
        self.cards: List[Card] = []
        self.bet: int = 0
        self.is_doubled_down: bool = False
        self.is_split: bool = False
        self.is_stand: bool = False

    def add_card(self, card: Card):
        self.cards.append(card)

    def get_value(self) -> int:
        """Calculate the hand value, handling Aces properly."""
        value = 0
        aces = 0
        for card in self.cards:
            val = card.get_blackjack_value()
            if val == 11:  # Ace
                aces += 1
            value += val

        # Adjust for Aces (count as 1 instead of 11 if busting)
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1

        return value

    def is_blackjack(self) -> bool:
        """Check if the hand is a natural blackjack (21 with exactly 2 cards)."""
        return len(self.cards) == 2 and self.get_value() == 21

    def is_bust(self) -> bool:
        return self.get_value() > 21

    def can_split(self) -> bool:
        """Check if the hand can be split (2 cards of same value)."""
        if len(self.cards) != 2:
            return False
        return self.cards[0].get_blackjack_value() == self.cards[1].get_blackjack_value()

    def can_double_down(self) -> bool:
        """Check if the hand can double down (only on first two cards)."""
        return len(self.cards) == 2 and not self.is_doubled_down

    def clear(self):
        self.cards = []
        self.bet = 0
        self.is_doubled_down = False
        self.is_split = False
        self.is_stand = False


class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 color: Tuple[int, int, int] = LIGHT_GRAY, 
                 hover_color: Tuple[int, int, int] = DARK_GRAY,
                 text_color: Tuple[int, int, int] = BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False

    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)

        text = FONT_SMALL.render(self.text, True, self.text_color)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def set_text(self, text: str):
        self.text = text


class Chip:
    def __init__(self, x: int, y: int, value: int):
        self.value = value
        self.radius = int(20 * SCALE_FACTOR)
        self.rect = pygame.Rect(x - self.radius, y - self.radius, 
                                self.radius * 2, self.radius * 2)
        self.is_hovered = False

        # Colors for different chip values
        self.chip_colors = {
            5: (255, 0, 0),       # Red
            10: (0, 0, 255),      # Blue
            25: (0, 128, 0),      # Green
            50: (128, 0, 128),    # Purple
            100: (0, 0, 0),       # Black
        }

    def draw(self, screen):
        color = self.chip_colors.get(self.value, GRAY)
        # Draw chip shadow
        pygame.draw.circle(screen, DARK_GRAY, 
                          (self.rect.centerx + 2, self.rect.centery + 2), self.radius)
        # Draw chip
        pygame.draw.circle(screen, color, self.rect.center, self.radius)
        pygame.draw.circle(screen, WHITE, self.rect.center, self.radius, 2)

        # Draw value text
        text = FONT_SMALL.render(f"${self.value}", True, WHITE)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Blackjack:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Blackjack")
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self):
        self.deck = Deck(num_decks=6)
        self.player_hands: List[Hand] = [Hand()]
        self.current_hand_index = 0
        self.dealer_hand = Hand()
        self.player_money = 1000
        self.current_bet = 0
        self.game_state = "betting"  # betting, playing, dealer_turn, game_over
        self.message = "Place your bet!"
        self.message_color = WHITE
        self.result_message = ""
        self.result_color = WHITE

        # Create buttons
        button_width = int(120 * SCALE_FACTOR)
        button_height = int(40 * SCALE_FACTOR)
        button_y = WINDOW_HEIGHT - int(60 * SCALE_FACTOR)
        center_x = WINDOW_WIDTH // 2

        self.buttons = {
            'hit': Button(center_x - button_width - int(10 * SCALE_FACTOR), button_y, 
                         button_width, button_height, "Hit", LIGHT_GRAY, DARK_GRAY),
            'stand': Button(center_x + int(10 * SCALE_FACTOR), button_y, 
                           button_width, button_height, "Stand", LIGHT_GRAY, DARK_GRAY),
            'double': Button(center_x - button_width - int(10 * SCALE_FACTOR), 
                            button_y - button_height - int(10 * SCALE_FACTOR),
                            button_width, button_height, "Double", LIGHT_GRAY, DARK_GRAY),
            'split': Button(center_x + int(10 * SCALE_FACTOR), 
                           button_y - button_height - int(10 * SCALE_FACTOR),
                           button_width, button_height, "Split", LIGHT_GRAY, DARK_GRAY),
            'deal': Button(center_x - button_width // 2, button_y,
                          button_width, button_height, "Deal", LIGHT_GRAY, DARK_GRAY),
            'new_game': Button(center_x - button_width // 2, button_y,
                              button_width, button_height, "New Game", LIGHT_GRAY, DARK_GRAY),
        }

        # Create chip buttons
        chip_y = WINDOW_HEIGHT - int(120 * SCALE_FACTOR)
        chip_start_x = center_x - int(150 * SCALE_FACTOR)
        self.chip_buttons = [
            Chip(chip_start_x, chip_y, 5),
            Chip(chip_start_x + int(60 * SCALE_FACTOR), chip_y, 10),
            Chip(chip_start_x + int(120 * SCALE_FACTOR), chip_y, 25),
            Chip(chip_start_x + int(180 * SCALE_FACTOR), chip_y, 50),
            Chip(chip_start_x + int(240 * SCALE_FACTOR), chip_y, 100),
        ]

    def get_current_hand(self) -> Hand:
        """Get the current active player hand."""
        if self.current_hand_index < len(self.player_hands):
            return self.player_hands[self.current_hand_index]
        return self.player_hands[0]

    def deal_initial_cards(self):
        """Deal the initial two cards to player and dealer."""
        if DEBUG:
            print(f"\nDealing initial cards...")
            print(f"Player bet: ${self.current_bet}")

        # Clear previous hands
        self.player_hands = [Hand()]
        self.current_hand_index = 0
        self.dealer_hand = Hand()

        # Set the bet on the current hand
        self.player_hands[0].bet = self.current_bet

        # Deal two cards to player and dealer
        for _ in range(2):
            card = self.deck.draw_card()
            card.face_up = True
            self.player_hands[0].add_card(card)

            card = self.deck.draw_card()
            if len(self.dealer_hand.cards) == 0:
                # Dealer's first card is face up
                card.face_up = True
            else:
                # Dealer's second card is face down
                card.face_up = False
            self.dealer_hand.add_card(card)

        if DEBUG:
            print(f"Player hand: {[c.get_display_name() for c in self.player_hands[0].cards]}")
            print(f"Player value: {self.player_hands[0].get_value()}")
            print(f"Dealer hand: {[c.get_display_name() for c in self.dealer_hand.cards]}")
            print(f"Dealer showing: {self.dealer_hand.cards[0].get_display_name()}")

        # Check for blackjack
        if self.player_hands[0].is_blackjack():
            if self.dealer_hand.is_blackjack():
                self.end_game("Push! Both have Blackjack!", YELLOW)
            else:
                self.end_game("Blackjack! You win 3:2!", GOLD)
            return

        if self.dealer_hand.is_blackjack():
            self.end_game("Dealer has Blackjack! You lose!", RED)
            return

        self.game_state = "playing"
        self.message = "Your turn - Hit or Stand?"
        self.message_color = WHITE

    def hit(self):
        """Player takes another card."""
        hand = self.get_current_hand()
        card = self.deck.draw_card()
        card.face_up = True
        hand.add_card(card)

        if DEBUG:
            print(f"Player hits: {card.get_display_name()}")
            print(f"Player hand value: {hand.get_value()}")

        if hand.is_bust():
            if DEBUG:
                print("Player busts!")
            self.message = f"Bust! Hand {self.current_hand_index + 1} is over {hand.get_value()}!"
            self.message_color = RED
            self.advance_to_next_hand_or_dealer()
        elif hand.get_value() == 21:
            self.message = "21! Standing automatically."
            self.message_color = GOLD
            self.stand()

    def stand(self):
        """Player stands with current hand."""
        hand = self.get_current_hand()
        hand.is_stand = True

        if DEBUG:
            print(f"Player stands with value: {hand.get_value()}")

        self.advance_to_next_hand_or_dealer()

    def double_down(self):
        """Player doubles down (double bet, one more card, then stand)."""
        hand = self.get_current_hand()

        if self.player_money < hand.bet:
            self.message = "Not enough money to double down!"
            self.message_color = RED
            return

        # Double the bet
        self.player_money -= hand.bet
        hand.bet *= 2
        hand.is_doubled_down = True

        # Take one card
        card = self.deck.draw_card()
        card.face_up = True
        hand.add_card(card)

        if DEBUG:
            print(f"Player doubles down: {card.get_display_name()}")
            print(f"Player hand value: {hand.get_value()}")
            print(f"New bet: ${hand.bet}")

        if hand.is_bust():
            self.message = f"Bust on double down! Hand {self.current_hand_index + 1} is over!"
            self.message_color = RED
        else:
            self.message = f"Doubled down with {hand.get_value()}!"
            self.message_color = WHITE

        hand.is_stand = True
        self.advance_to_next_hand_or_dealer()

    def split(self):
        """Player splits their hand into two."""
        hand = self.get_current_hand()

        if not hand.can_split():
            self.message = "Cannot split this hand!"
            self.message_color = RED
            return

        if self.player_money < hand.bet:
            self.message = "Not enough money to split!"
            self.message_color = RED
            return

        if DEBUG:
            print("Player splits hand!")

        # Create two new hands
        hand1 = Hand()
        hand2 = Hand()

        # Split the cards
        hand1.add_card(hand.cards[0])
        hand2.add_card(hand.cards[1])

        # Set bets
        self.player_money -= hand.bet
        hand1.bet = hand.bet
        hand2.bet = hand.bet
        hand1.is_split = True
        hand2.is_split = True

        # Deal one more card to each hand
        for h in [hand1, hand2]:
            card = self.deck.draw_card()
            card.face_up = True
            h.add_card(card)

        # Replace the current hand with the two split hands
        self.player_hands.remove(hand)
        self.player_hands.insert(self.current_hand_index, hand1)
        self.player_hands.insert(self.current_hand_index + 1, hand2)

        if DEBUG:
            print(f"Hand 1: {[c.get_display_name() for c in hand1.cards]} (value: {hand1.get_value()})")
            print(f"Hand 2: {[c.get_display_name() for c in hand2.cards]} (value: {hand2.get_value()})")

        self.message = f"Hands split! Playing hand {self.current_hand_index + 1}"
        self.message_color = WHITE

    def advance_to_next_hand_or_dealer(self):
        """Move to the next hand or start dealer's turn."""
        # Check if there are more hands to play
        for i in range(self.current_hand_index + 1, len(self.player_hands)):
            hand = self.player_hands[i]
            if not hand.is_stand and not hand.is_bust() and not hand.is_blackjack():
                self.current_hand_index = i
                self.message = f"Playing hand {i + 1} - Hit or Stand?"
                self.message_color = WHITE
                return

        # All hands are done, dealer's turn
        self.dealer_turn()

    def dealer_turn(self):
        """Dealer plays according to rules (hit on 16 or below, stand on 17 or above)."""
        self.game_state = "dealer_turn"
        self.message = "Dealer's turn..."
        self.message_color = WHITE

        # Reveal dealer's hole card
        if len(self.dealer_hand.cards) > 1:
            self.dealer_hand.cards[1].face_up = True

        if DEBUG:
            print(f"\nDealer's turn:")
            print(f"Dealer hand: {[c.get_display_name() for c in self.dealer_hand.cards]}")
            print(f"Dealer value: {self.dealer_hand.get_value()}")

        # Dealer hits on soft 17 or below
        while self.dealer_hand.get_value() < 17:
            card = self.deck.draw_card()
            card.face_up = True
            self.dealer_hand.add_card(card)
            if DEBUG:
                print(f"Dealer hits: {card.get_display_name()}")
                print(f"Dealer value: {self.dealer_hand.get_value()}")

        if DEBUG:
            if self.dealer_hand.is_bust():
                print("Dealer busts!")
            else:
                print(f"Dealer stands with {self.dealer_hand.get_value()}")

        self.evaluate_results()

    def evaluate_results(self):
        """Evaluate the results of all hands."""
        self.game_state = "game_over"
        dealer_value = self.dealer_hand.get_value()
        dealer_bust = self.dealer_hand.is_bust()

        if DEBUG:
            print(f"\nEvaluating results:")
            print(f"Dealer: {dealer_value}{' (BUST)' if dealer_bust else ''}")

        total_winnings = 0
        results = []

        for i, hand in enumerate(self.player_hands):
            player_value = hand.get_value()
            player_bust = hand.is_bust()

            if DEBUG:
                print(f"Hand {i + 1}: {player_value}{' (BUST)' if player_bust else ''}")

            if player_bust:
                results.append((f"Hand {i + 1}: Bust! Lost ${hand.bet}", RED))
                # Bet was already deducted, no money returned
            elif hand.is_blackjack() and not dealer_bust and not self.dealer_hand.is_blackjack():
                # Blackjack pays 3:2 - return bet + 1.5x profit
                winnings = hand.bet + int(hand.bet * 1.5)
                total_winnings += winnings
                results.append((f"Hand {i + 1}: Blackjack! Won ${int(hand.bet * 1.5)}", GOLD))
            elif dealer_bust:
                # Return bet + equal profit
                total_winnings += hand.bet * 2
                results.append((f"Hand {i + 1}: Dealer bust! Won ${hand.bet}", GOLD))
            elif player_value > dealer_value:
                # Return bet + equal profit
                total_winnings += hand.bet * 2
                results.append((f"Hand {i + 1}: Won ${hand.bet}", GOLD))
            elif player_value == dealer_value:
                # Push - return the bet
                total_winnings += hand.bet
                results.append((f"Hand {i + 1}: Push (tie)", YELLOW))
            else:
                # Lost - bet stays with the house
                results.append((f"Hand {i + 1}: Lost ${hand.bet}", RED))

        self.player_money += total_winnings

        if DEBUG:
            print(f"Total winnings: ${total_winnings}")
            print(f"Player money: ${self.player_money}")

        # Show results
        if len(results) == 1:
            self.result_message = results[0][0]
            self.result_color = results[0][1]
        else:
            self.result_message = f"Total: {'+' if total_winnings >= 0 else ''}${total_winnings}"
            self.result_color = GOLD if total_winnings >= 0 else RED

        self.message = "Game Over!"
        self.message_color = WHITE

        if self.player_money <= 0:
            self.message = "You're out of money! Game over!"
            self.message_color = RED

    def end_game(self, message: str, color: Tuple[int, int, int]):
        """End the game with a specific message."""
        self.game_state = "game_over"
        self.message = message
        self.message_color = color
        self.result_message = ""

        # Reveal dealer's hole card
        if len(self.dealer_hand.cards) > 1:
            self.dealer_hand.cards[1].face_up = True

    def draw(self):
        self.screen.fill(DARK_GREEN)

        # Draw felt texture pattern (simple lines)
        for i in range(0, WINDOW_WIDTH, int(50 * SCALE_FACTOR)):
            pygame.draw.line(self.screen, GREEN, (i, 0), (i, WINDOW_HEIGHT), 1)
        for i in range(0, WINDOW_HEIGHT, int(50 * SCALE_FACTOR)):
            pygame.draw.line(self.screen, GREEN, (0, i), (WINDOW_WIDTH, i), 1)

        # Draw dealer's hand
        dealer_label = FONT_SMALL.render("Dealer", True, WHITE)
        self.screen.blit(dealer_label, (int(20 * SCALE_FACTOR), int(20 * SCALE_FACTOR)))

        dealer_x = int(20 * SCALE_FACTOR)
        dealer_y = int(50 * SCALE_FACTOR)
        for i, card in enumerate(self.dealer_hand.cards):
            if card.face_up:
                card.rect.topleft = (dealer_x + i * (CARD_WIDTH + int(10 * SCALE_FACTOR)), dealer_y)
                self.screen.blit(card.image, card.rect)
            else:
                self.screen.blit(CARD_BACK, (dealer_x + i * (CARD_WIDTH + int(10 * SCALE_FACTOR)), dealer_y))

        # Show dealer value (only face-up cards)
        if self.dealer_hand.cards:
            visible_value = 0
            has_ace = False
            for card in self.dealer_hand.cards:
                if card.face_up:
                    val = card.get_blackjack_value()
                    if val == 11:
                        has_ace = True
                    visible_value += val
            # Adjust for ace if needed
            if has_ace and visible_value > 21:
                visible_value -= 10

            if self.game_state in ["dealer_turn", "game_over"]:
                dealer_value_text = FONT_SMALL.render(f"Value: {self.dealer_hand.get_value()}", True, WHITE)
            else:
                dealer_value_text = FONT_SMALL.render(f"Showing: {visible_value}", True, WHITE)
            self.screen.blit(dealer_value_text, 
                           (int(20 * SCALE_FACTOR), dealer_y + CARD_HEIGHT + int(5 * SCALE_FACTOR)))

        # Draw player's hand(s)
        player_label_y = WINDOW_HEIGHT // 2 - int(50 * SCALE_FACTOR)
        player_label = FONT_SMALL.render("Player", True, WHITE)
        self.screen.blit(player_label, (int(20 * SCALE_FACTOR), player_label_y))

        player_y = player_label_y + int(30 * SCALE_FACTOR)
        for hand_idx, hand in enumerate(self.player_hands):
            hand_x = int(20 * SCALE_FACTOR) + hand_idx * (CARD_WIDTH + int(10 * SCALE_FACTOR)) * 3

            # Draw hand indicator if this is the current hand
            if hand_idx == self.current_hand_index and self.game_state == "playing":
                indicator_rect = pygame.Rect(hand_x - int(5 * SCALE_FACTOR), 
                                            player_y - int(5 * SCALE_FACTOR),
                                            len(hand.cards) * (CARD_WIDTH + int(10 * SCALE_FACTOR)) + int(10 * SCALE_FACTOR),
                                            CARD_HEIGHT + int(40 * SCALE_FACTOR))
                pygame.draw.rect(self.screen, YELLOW, indicator_rect, 2)

            for i, card in enumerate(hand.cards):
                card.rect.topleft = (hand_x + i * (CARD_WIDTH + int(10 * SCALE_FACTOR)), player_y)
                self.screen.blit(card.image, card.rect)

            # Show hand value and bet
            value_text = FONT_SMALL.render(f"Value: {hand.get_value()}", True, WHITE)
            self.screen.blit(value_text, (hand_x, player_y + CARD_HEIGHT + int(5 * SCALE_FACTOR)))

            bet_text = FONT_SMALL.render(f"Bet: ${hand.bet}", True, GOLD)
            self.screen.blit(bet_text, (hand_x, player_y + CARD_HEIGHT + int(30 * SCALE_FACTOR)))

        # Draw message
        if self.message:
            msg_text = FONT_MEDIUM.render(self.message, True, self.message_color)
            msg_rect = msg_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - int(80 * SCALE_FACTOR)))
            self.screen.blit(msg_text, msg_rect)

        # Draw result message
        if self.result_message and self.game_state == "game_over":
            result_text = FONT_LARGE.render(self.result_message, True, self.result_color)
            result_rect = result_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - int(30 * SCALE_FACTOR)))
            self.screen.blit(result_text, result_rect)

        # Draw money display
        money_text = FONT_MEDIUM.render(f"Money: ${self.player_money}", True, GOLD)
        money_rect = money_text.get_rect(topright=(WINDOW_WIDTH - int(20 * SCALE_FACTOR), int(20 * SCALE_FACTOR)))
        self.screen.blit(money_text, money_rect)

        # Draw current bet display
        if self.current_bet > 0:
            bet_display = FONT_SMALL.render(f"Current Bet: ${self.current_bet}", True, WHITE)
            bet_rect = bet_display.get_rect(topright=(WINDOW_WIDTH - int(20 * SCALE_FACTOR), int(60 * SCALE_FACTOR)))
            self.screen.blit(bet_display, bet_rect)

        # Draw deck count
        deck_text = FONT_SMALL.render(f"Cards left: {self.deck.remaining()}", True, WHITE)
        deck_rect = deck_text.get_rect(topright=(WINDOW_WIDTH - int(20 * SCALE_FACTOR), int(90 * SCALE_FACTOR)))
        self.screen.blit(deck_text, deck_rect)

        # Draw buttons based on game state
        if self.game_state == "betting":
            self.buttons['deal'].draw(self.screen)
            for chip in self.chip_buttons:
                chip.draw(self.screen)
        elif self.game_state == "playing":
            self.buttons['hit'].draw(self.screen)
            self.buttons['stand'].draw(self.screen)

            hand = self.get_current_hand()
            if hand.can_double_down() and self.player_money >= hand.bet:
                self.buttons['double'].draw(self.screen)
            if hand.can_split() and self.player_money >= hand.bet:
                self.buttons['split'].draw(self.screen)
        elif self.game_state == "game_over":
            if self.player_money > 0:
                self.buttons['deal'].set_text("Deal Again")
                self.buttons['deal'].draw(self.screen)
            else:
                self.buttons['new_game'].draw(self.screen)

        pygame.display.flip()

    def handle_betting(self, event) -> bool:
        """Handle betting phase events."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check chip clicks
            for chip in self.chip_buttons:
                if chip.handle_event(event):
                    if self.player_money >= chip.value:
                        self.current_bet += chip.value
                        self.player_money -= chip.value
                        if DEBUG:
                            print(f"Added ${chip.value} chip. Current bet: ${self.current_bet}")
                    else:
                        self.message = "Not enough money!"
                        self.message_color = RED
                    return True

            # Check deal button
            if self.buttons['deal'].handle_event(event):
                if self.current_bet > 0:
                    self.deal_initial_cards()
                else:
                    self.message = "Place a bet first!"
                    self.message_color = RED
                return True

        return False

    def handle_playing(self, event) -> bool:
        """Handle playing phase events."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.buttons['hit'].handle_event(event):
                self.hit()
                return True
            elif self.buttons['stand'].handle_event(event):
                self.stand()
                return True
            elif self.buttons['double'].handle_event(event):
                self.double_down()
                return True
            elif self.buttons['split'].handle_event(event):
                self.split()
                return True

        return False

    def handle_game_over(self, event) -> bool:
        """Handle game over phase events."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.player_money > 0:
                if self.buttons['deal'].handle_event(event):
                    # Reset for next round
                    self.current_bet = 0
                    self.player_hands = [Hand()]
                    self.current_hand_index = 0
                    self.dealer_hand = Hand()
                    self.game_state = "betting"
                    self.message = "Place your bet!"
                    self.message_color = WHITE
                    self.result_message = ""
                    return True
            else:
                if self.buttons['new_game'].handle_event(event):
                    self.reset_game()
                    return True

        return False

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Update button hover states
                if event.type == pygame.MOUSEMOTION:
                    for button in self.buttons.values():
                        button.is_hovered = button.rect.collidepoint(event.pos)
                    for chip in self.chip_buttons:
                        chip.is_hovered = chip.rect.collidepoint(event.pos)

                # Handle events based on game state
                if self.game_state == "betting":
                    self.handle_betting(event)
                elif self.game_state == "playing":
                    self.handle_playing(event)
                elif self.game_state == "dealer_turn":
                    # Dealer turn is automatic, just wait a bit for visual effect
                    pygame.time.wait(500)
                    self.dealer_turn()
                elif self.game_state == "game_over":
                    self.handle_game_over(event)

            self.draw()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    game = Blackjack()
    game.run()
