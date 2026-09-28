import numpy as np
import matplotlib.pyplot as plt

# Create x values from -10 to 10
x = np.linspace(-10, 10, 1000)

# Calculate y = sin(x)
y = np.sin(x)

# Draw the function
plt.plot(x, y)

# Draw coordinate axes
plt.axhline(0)
plt.axvline(0)

plt.xlabel("x")
plt.ylabel("y")
plt.title("y = sin(x)")
plt.grid()

plt.show()