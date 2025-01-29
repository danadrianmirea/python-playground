from datetime import datetime, date


today = date.today()

name = input("What is your name?: ")
age = int(input("How old are you?: "))
date_str = input("Enter your birthdate (YYYY-MM-DD): ")


print(f"Hello {name}!")

try:
    birthdate = datetime.strptime(date_str, "%Y-%m-%d").date()  # Convert to date object
    if birthdate.month == today.month and birthdate.day == today.day:
        print("Happy Birthday! 🎉")
    else:
        print("It's not your birthday today.")
except ValueError:
    print("Invalid date format. Please use YYYY-MM-DD.")

print(f"You are {age} years old")