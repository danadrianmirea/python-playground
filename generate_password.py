#!/usr/bin/env python3
"""
Random Password Generator

Usage: python password_generator.py <length>
Example: python password_generator.py 12
"""

import secrets
import string
import sys

# Define character pools
LOWERCASE = string.ascii_lowercase
UPPERCASE = string.ascii_uppercase
DIGITS = string.digits
SPECIAL = "!@#$%^&*()-_=+[]{}|;:,.<>?/~"   # Customize as needed
ALL_CHARS = LOWERCASE + UPPERCASE + DIGITS + SPECIAL

def generate_password(length: int) -> str:
    """
    Generate a random password with the specified length.
    The password will contain at least one character from each pool.
    """
    if length < 4:
        raise ValueError("Password length must be at least 4 to include one of each character type.")

    # Ensure at least one of each required type
    password_chars = [
        secrets.choice(LOWERCASE),
        secrets.choice(UPPERCASE),
        secrets.choice(DIGITS),
        secrets.choice(SPECIAL),
    ]

    # Fill the remaining length with random characters from the full pool
    remaining = length - 4
    password_chars.extend(secrets.choice(ALL_CHARS) for _ in range(remaining))

    # Shuffle the list to avoid predictable order
    secrets.SystemRandom().shuffle(password_chars)

    return ''.join(password_chars)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python password_generator.py <length>")
        sys.exit(1)

    try:
        length = int(sys.argv[1])
    except ValueError:
        print("Error: Length must be an integer.")
        sys.exit(1)

    try:
        password = generate_password(length)
        print(password)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)