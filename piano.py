"""
Piano - A pygame script that draws a piano keyboard and can be played using the computer keyboard.

Keys mapping (two octaves):
  Lower octave (C4-B4): A S D F G H J K L ; ' (Enter)
  Upper octave (C5-B5): Z X C V B N M , . / (Shift)

  White keys: A S D F G H J K L ; ' Z X C V B N M , . /
  Black keys: W E   T Y U   O P   [ ] \\ (for sharps/flats)
"""

import pygame
import sys
import os

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 400
BACKGROUND_COLOR = (30, 30, 30)

# Piano layout
# Two octaves: C4 to B5 (24 keys total: 14 white + 10 black per 2 octaves)
# White keys pattern: C D E F G A B (7 per octave)
# Black keys pattern: C# D# F# G# A# (5 per octave)

# Note frequencies (C4 = 261.63 Hz)
NOTE_NAMES = [
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5"
]

# Frequencies for 2 octaves starting from C4
def get_frequencies():
    """Generate frequencies for 2 octaves starting from C4 (A4 = 440 Hz)."""
    freqs = []
    for i in range(24):
        # C4 is 9 semitones below A4 (440 Hz)
        # frequency = 440 * 2^((n - 57) / 12) where n is MIDI note number
        # C4 = MIDI 60, so n = 60 + i
        freq = 440.0 * (2.0 ** ((i - 9) / 12.0))
        freqs.append(freq)
    return freqs

FREQUENCIES = get_frequencies()

# Key mappings (computer keyboard to piano note index)
# Lower octave (C4-B4): indices 0-11
# Upper octave (C5-B5): indices 12-23

# White keys - lower octave
# C4 D4 E4 F4 G4 A4 B4
# A  S  D  F  G  H  J
LOWER_WHITE_MAP = {
    pygame.K_a: 0,   # C4
    pygame.K_s: 2,   # D4
    pygame.K_d: 4,   # E4
    pygame.K_f: 5,   # F4
    pygame.K_g: 7,   # G4
    pygame.K_h: 9,   # A4
    pygame.K_j: 11,  # B4
}

# White keys - upper octave
# C5 D5 E5 F5 G5 A5 B5
# K  L  ;  '  Z  X  C
UPPER_WHITE_MAP = {
    pygame.K_k: 12,  # C5
    pygame.K_l: 14,  # D5
    pygame.K_SEMICOLON: 16,  # E5
    pygame.K_QUOTE: 17,  # F5
    pygame.K_z: 19,  # G5
    pygame.K_x: 21,  # A5
    pygame.K_c: 23,  # B5
}

# Black keys - lower octave
# C#4 D#4 F#4 G#4 A#4
# W   E   T   Y   U
LOWER_BLACK_MAP = {
    pygame.K_w: 1,   # C#4
    pygame.K_e: 3,   # D#4
    pygame.K_t: 6,   # F#4
    pygame.K_y: 8,   # G#4
    pygame.K_u: 10,  # A#4
}

# Black keys - upper octave
# C#5 D#5 F#5 G#5 A#5
# O   P   [   ]   \\
UPPER_BLACK_MAP = {
    pygame.K_o: 13,  # C#5
    pygame.K_p: 15,  # D#5
    pygame.K_LEFTBRACKET: 18,  # F#5
    pygame.K_RIGHTBRACKET: 20,  # G#5
    pygame.K_BACKSLASH: 22,  # A#5
}

# Combine all mappings
KEY_TO_NOTE = {}
KEY_TO_NOTE.update(LOWER_WHITE_MAP)
KEY_TO_NOTE.update(UPPER_WHITE_MAP)
KEY_TO_NOTE.update(LOWER_BLACK_MAP)
KEY_TO_NOTE.update(UPPER_BLACK_MAP)

# Reverse mapping: note_index -> keyboard key label
NOTE_TO_KEY_LABEL = {}
for key, note_idx in KEY_TO_NOTE.items():
    label = {
        pygame.K_a: "A", pygame.K_s: "S", pygame.K_d: "D", pygame.K_f: "F",
        pygame.K_g: "G", pygame.K_h: "H", pygame.K_j: "J", pygame.K_k: "K",
        pygame.K_l: "L", pygame.K_SEMICOLON: ";", pygame.K_QUOTE: "'",
        pygame.K_z: "Z", pygame.K_x: "X", pygame.K_c: "C",
        pygame.K_w: "W", pygame.K_e: "E", pygame.K_t: "T", pygame.K_y: "Y",
        pygame.K_u: "U", pygame.K_o: "O", pygame.K_p: "P",
        pygame.K_LEFTBRACKET: "[", pygame.K_RIGHTBRACKET: "]", pygame.K_BACKSLASH: "\\",
    }.get(key, "?")
    NOTE_TO_KEY_LABEL[note_idx] = label


class PianoKey:
    def __init__(self, note_index, is_black, rect):
        self.note_index = note_index
        self.is_black = is_black
        self.rect = rect
        self.is_pressed = False
        self.note_name = NOTE_NAMES[note_index]
        self.key_label = NOTE_TO_KEY_LABEL.get(note_index, "")

    def draw(self, screen, font, key_font):
        if self.is_black:
            if self.is_pressed:
                color = (100, 100, 180)  # Pressed black key
            else:
                color = (20, 20, 20)  # Normal black key
            pygame.draw.rect(screen, color, self.rect)
            pygame.draw.rect(screen, (60, 60, 60), self.rect, 1)

            # Draw keyboard key label above black key
            if self.key_label:
                text = key_font.render(self.key_label, True, (200, 200, 200))
                text_rect = text.get_rect(center=(self.rect.centerx, self.rect.top - 12))
                screen.blit(text, text_rect)
        else:
            if self.is_pressed:
                color = (200, 220, 255)  # Pressed white key
            else:
                color = (255, 255, 255)  # Normal white key
            pygame.draw.rect(screen, color, self.rect)
            pygame.draw.rect(screen, (0, 0, 0), self.rect, 1)

            # Draw keyboard key label above white key
            if self.key_label:
                text = key_font.render(self.key_label, True, (180, 180, 180))
                text_rect = text.get_rect(center=(self.rect.centerx, self.rect.top - 12))
                screen.blit(text, text_rect)

            # Draw note name at bottom of white key
            if self.rect.height > 60:
                text = font.render(self.note_name, True, (100, 100, 100))
                text_rect = text.get_rect(center=(self.rect.centerx, self.rect.bottom - 20))
                screen.blit(text, text_rect)


class Piano:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.keys = []
        self.pressed_notes = {}  # note_index -> pygame.mixer.Sound
        self._create_keys()

    def _create_keys(self):
        """Create the piano key layout."""
        # Layout parameters
        num_white_keys = 14  # 2 octaves
        white_key_width = self.screen_width // num_white_keys
        white_key_height = int(self.screen_height * 0.85)
        black_key_width = int(white_key_width * 0.6)
        black_key_height = int(white_key_height * 0.6)

        # White key positions (7 per octave)
        # Pattern: C D E F G A B
        white_note_indices = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23]

        # Create white keys
        white_rects = []
        for i, note_idx in enumerate(white_note_indices):
            x = i * white_key_width
            y = self.screen_height - white_key_height
            rect = pygame.Rect(x, y, white_key_width - 1, white_key_height)
            key = PianoKey(note_idx, False, rect)
            self.keys.append(key)
            white_rects.append(rect)

        # Black key positions relative to white keys
        # Octave pattern: C#(after C), D#(after D), F#(after F), G#(after G), A#(after A)
        # Black keys sit between white keys
        black_key_positions = [
            # Octave 1 (C4-B4)
            (1, 0),   # C#4 between C4 and D4 (white key 0 and 1)
            (3, 1),   # D#4 between D4 and E4 (white key 1 and 2)
            (6, 3),   # F#4 between F4 and G4 (white key 3 and 4)
            (8, 4),   # G#4 between G4 and A4 (white key 4 and 5)
            (10, 5),  # A#4 between A4 and B4 (white key 5 and 6)
            # Octave 2 (C5-B5)
            (13, 7),   # C#5 between C5 and D5 (white key 7 and 8)
            (15, 8),   # D#5 between D5 and E5 (white key 8 and 9)
            (18, 10),  # F#5 between F5 and G5 (white key 10 and 11)
            (20, 11),  # G#5 between G5 and A5 (white key 11 and 12)
            (22, 12),  # A#5 between A5 and B5 (white key 12 and 13)
        ]

        for note_idx, white_idx in black_key_positions:
            # Position black key between the two white keys
            left_white = white_rects[white_idx]
            right_white = white_rects[white_idx + 1]
            x = left_white.right - black_key_width // 2
            y = self.screen_height - white_key_height
            rect = pygame.Rect(x, y, black_key_width, black_key_height)
            key = PianoKey(note_idx, True, rect)
            self.keys.append(key)

        # Sort keys by note_index for consistent ordering
        self.keys.sort(key=lambda k: k.note_index)

    def press_note(self, note_index):
        """Play a note and mark it as pressed."""
        if note_index < 0 or note_index >= len(FREQUENCIES):
            return

        # Mark key as pressed
        for key in self.keys:
            if key.note_index == note_index:
                key.is_pressed = True
                break

        # Play the sound
        if note_index not in self.pressed_notes:
            freq = FREQUENCIES[note_index]
            sound = self._generate_sound(freq)
            sound.play(-1)  # Loop until released
            self.pressed_notes[note_index] = sound

    def release_note(self, note_index):
        """Stop a note and mark it as released."""
        if note_index < 0 or note_index >= len(FREQUENCIES):
            return

        # Mark key as released
        for key in self.keys:
            if key.note_index == note_index:
                key.is_pressed = False
                break

        # Stop the sound
        if note_index in self.pressed_notes:
            self.pressed_notes[note_index].stop()
            del self.pressed_notes[note_index]

    def _generate_sound(self, frequency, duration=1.0, sample_rate=44100):
        """Generate a sine wave sound at the given frequency."""
        import numpy as np

        # Calculate number of samples
        num_samples = int(sample_rate * duration)

        # Generate sine wave
        t = np.linspace(0, duration, num_samples, False)
        wave = np.sin(2 * np.pi * frequency * t)

        # Apply envelope (attack and release) to avoid clicks
        attack = int(sample_rate * 0.01)  # 10ms attack
        release = int(sample_rate * 0.05)  # 50ms release

        # Attack envelope
        if attack > 0:
            wave[:attack] *= np.linspace(0, 1, attack)

        # Release envelope
        if release > 0:
            wave[-release:] *= np.linspace(1, 0, release)

        # Add harmonics for richer sound
        # Second harmonic
        wave2 = np.sin(2 * np.pi * frequency * 2 * t) * 0.3
        # Third harmonic
        wave3 = np.sin(2 * np.pi * frequency * 3 * t) * 0.15
        wave = wave + wave2 + wave3

        # Normalize
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val

        # Convert to 16-bit PCM
        wave = (wave * 32767 * 0.5).astype(np.int16)

        # Make it 2D for stereo (duplicate mono to both channels)
        stereo_wave = np.column_stack((wave, wave))

        # Create sound from array
        sound = pygame.sndarray.make_sound(stereo_wave)
        return sound

    def draw(self, screen, font, key_font):
        """Draw all piano keys."""
        # Draw white keys first
        for key in self.keys:
            if not key.is_black:
                key.draw(screen, font, key_font)

        # Draw black keys on top
        for key in self.keys:
            if key.is_black:
                key.draw(screen, font, key_font)

    def get_key_for_note(self, note_index):
        """Get the key object for a given note index."""
        for key in self.keys:
            if key.note_index == note_index:
                return key
        return None


def main():
    """Main game loop."""
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Piano")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)       # For note names (C4, D4, etc.)
    key_font = pygame.font.Font(None, 18)   # For keyboard key labels (A, S, W, etc.)

    # Initialize the mixer
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()

    piano = Piano(WINDOW_WIDTH, WINDOW_HEIGHT)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in KEY_TO_NOTE:
                    note_index = KEY_TO_NOTE[event.key]
                    piano.press_note(note_index)

            elif event.type == pygame.KEYUP:
                if event.key in KEY_TO_NOTE:
                    note_index = KEY_TO_NOTE[event.key]
                    piano.release_note(note_index)

        # Draw everything
        screen.fill(BACKGROUND_COLOR)
        piano.draw(screen, font, key_font)

        pygame.display.flip()
        clock.tick(60)

    # Stop all sounds before quitting
    for note_index in list(piano.pressed_notes.keys()):
        piano.release_note(note_index)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
