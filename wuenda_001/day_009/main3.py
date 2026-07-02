# import numpy as np
#
#
# def g(_in):
#     print(_in)
#     pass
#
#
# def dense(a_in, W, b):
#     units = W.shape[1]
#     a_out = np.zeros(units)
#     for j in range(units):
#         w = W[:, j]
#         z = np.dot(w, a_in) + b
#         a_out[j] = g(z)
#     return a_out
#
#
# def sequential(x):
#     a1 = dense(w, W1, b1)
#     a2 = dense(a1, W2, b2)
#     a3 = dense(a2, W3, b3)
#     a4 = dense(a3, W4, b4)
#     f_x = a4
#     return f_x
