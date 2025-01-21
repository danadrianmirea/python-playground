import random
import math

print("Guess the number")

theNumber = random.randint(0, 99)
theGuess = -1

numberOfAttempts = math.ceil(math.log2(100))
running = 1

lowerBound = 0
higherBound = 99

while running:
    print("Attempts left: " + str(numberOfAttempts))
    try:
        theGuess = int(input("Enter your guess, range is ("+str(lowerBound)+","+str(higherBound)+"): ") )
    except Exception as e:    
        print("Invalid input, try again")
        continue
    
    numberOfAttempts = numberOfAttempts - 1
    if theGuess == theNumber:
        print("Congratulations! you won, the number was " + str(theGuess))
        running = 0
    elif numberOfAttempts == 0:
        print("You lost! ran out of attempts")
        running = 0
    elif theGuess < theNumber:
        print("The number is greater than your guess, try again")
        lowerBound = theGuess + 1
    elif theGuess > theNumber:
        higherBound = theGuess - 1 
        print("The number is smaller than your guess, try again")

        
        
