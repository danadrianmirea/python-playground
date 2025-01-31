age = 0

while age<=0 or age>150:
    age = int(input("Please enter your age: "))
    
if age < 18:
    print("You are under 18")
else:
    print("You are over 18")