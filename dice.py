import random

def roll_dice():
    total = 0
    num_of_dice = int(input("How many dice do you want to roll? "))

    dice = []

    for i in range(num_of_dice):
        dice.append(random.randint(1,6))

    for die in dice:
        total += die

    print("The total is", total)

roll_dice()
