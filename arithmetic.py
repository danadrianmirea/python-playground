import math

count = 0
count = count + 1
count += 1
count -= 2

mul = 1
mul = mul*3
mul = mul/3
mul *= 5
#mul /= 2
#mul = int(mul)
mul = (int)(mul/2)
mul = 1
mul = 2
mul **= 4

print(f"count: {count}, mul: {mul}")

x = 5.3
res = round(x)
resceil = math.ceil(x)
resfloor = math.floor(x)

print(f"res: {res}, resceil: {resceil}, resfloor: {resfloor}")

x = 16
ressqrt = math.sqrt(x)
print(f"ressqrt: {ressqrt}")


nutsRadius = float(input("Enter the radius of deez nuts: "))
areaNuts = math.pi * pow(nutsRadius, 2)
print(f"The area of deez nuts is: {areaNuts}")5