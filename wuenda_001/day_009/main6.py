import math
import numpy as np

AT = np.array([[200, 17]])
W = np.array([[1, -3, 5], [-2, 4, -6]])
b = np.array([[-1, 1, 2]])


def g(z):
    return z
    # return 1 / (1 + math.exp(-z))


def dense(AT, W, b):
    z = np.matmul(AT, W) + b
    print(z)
    a_out = g(z)
    return a_out


print(dense(AT, W, b))
