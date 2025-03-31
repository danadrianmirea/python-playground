import random

dice_art = {
    1: '''
┌─────────┐
│         │
│    ●    │
│         │
└─────────┘''',
    2: '''
┌─────────┐
│  ●      │
│         │
│      ●  │
└─────────┘''',
    3: '''
┌─────────┐
│  ●      │
│    ●    │
│      ●  │
└─────────┘''',
    4: '''
┌─────────┐
│  ●   ●  │
│         │
│  ●   ●  │
└─────────┘''',
    5: '''
┌─────────┐
│  ●   ●  │
│    ●    │
│  ●   ●  │
└─────────┘''',
    6: '''
┌─────────┐
│  ●   ●  │
│  ●   ●  │
│  ●   ●  │
└─────────┘'''
}

num_of_dice = 6

while True:
    dice = []
    total = 0
    
    for i in range(num_of_dice):
        dice.append(random.randint(1,6))

    # Split each die's art into lines and print them side by side
    for line in range(6):  # Each die art has 5 lines
        for die in dice:
            print(dice_art[die].split('\n')[line], end='  ')  # Add space between dice
        print()  # New line after each row of dice

    for die in dice:
        total += die

    print("The total is", total)
    
    roll_again = input("Would you like to roll again? (y/n): ").lower()
    if roll_again != 'y':
        break

print("Thanks for playing!") 