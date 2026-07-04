import sympy

J, w = sympy.symbols("J,w")

J = w**3 + 3 * w

print(J)

res = sympy.diff(J, w)
print(res)
