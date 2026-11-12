import sympy as sp

x = sp.symbols('x')

sp.integrate(x**2, x)
# x**3/3

result=sp.integrate(sp.sin(x), (x, 0, sp.pi))

print(result)