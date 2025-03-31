def add(num1, num2):
    return num1 + num2

def sub(num1, num2):
    return num1 - num2

def mul(num1, num2):
    return num1 * num2

if __name__ == "__main__":
    print(add(1, 2))
    print(sub(1, 2))
    print(mul(1, 2))
    print(add(mul(1, 2), add(1, 2)))
