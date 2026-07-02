import math
import numpy as np
# import tensorflow as tf

x = np.array([200, 17])
W = np.array([[1, -3, 5], [-2, 4, -6]])
b = np.array([-1, 1, 2])


def pr_shape(M):
    print(f"[M.shape]: {M.shape}")


def g(z):
    # 激活函数
    # sigmoid activate function code
    return 1 / (1 + math.exp(-z))


def dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out


res = dense(x, W, b)
print(f"[res]:{res}")
