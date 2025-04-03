def add(num1, num2):
    return num1 + num2

def sub(num1, num2):
    return num1 - num2

def mul(num1, num2):
    return num1 * num

def calculateSalary(baseSalary, taxPercent=0.45):
    return baseSalary-baseSalary*taxPercent

if __name__ == "__main__":
    print(add(1, 2))
    print(sub(1, 2))
    print(mul(1, 2))
    print(add(mul(1, 2), add(1, 2)))
    print(calculateSalary(5000))

