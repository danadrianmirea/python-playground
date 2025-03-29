from decimal import Decimal, getcontext
from math import factorial

def compute_pi(precision):
    # Set decimal precision
    getcontext().prec = precision + 1
    
    # Using Chudnovsky algorithm which converges much faster than other series
    C = 426880 * Decimal(10005).sqrt()
    L = 13591409
    X = 1
    M = 1
    K = 6
    S = L
    for i in range(1, precision):
        M = M * (K ** 3 - 16 * K) // (i ** 3)
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X
        K += 12
    pi = C / S
    return str(pi)

# Calculate first 100 decimals of pi
pi_digits = compute_pi(100)
print(f"First 100 decimals of pi:\n{pi_digits}")
