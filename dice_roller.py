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

while True:
    dice = []
    total = 0
    num_of_dice = int(input("How many dice do you want to roll? "))

    for i in range(num_of_dice):
        dice.append(random.randint(1,6))

    for die in dice:
        print(dice_art[die])
        
    for die in dice:
        total += die

    print("The total is", total)
    
    roll_again = input("Would you like to roll again? (y/n): ").lower()
    if roll_again != 'y':
        break

print("Thanks for playing!") 