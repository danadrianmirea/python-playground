def total_digits(n):
    total = 0
    length = len(str(n))  # Number of digits in n

    # Iterate over each number of digits (1 digit, 2 digits, etc.)
    for digits in range(1, length):
        start = 10**(digits - 1)
        end = 10**digits - 1
        total += (end - start + 1) * digits
    
    # Add digits for the last range (with the same number of digits as n)
    start = 10**(length - 1)
    total += (n - start + 1) * length
    
    return total

# Example usage:
n = int(input("Enter a number: "))
print("Total number of digits required:", total_digits(n))