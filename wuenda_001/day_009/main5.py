# 矩阵乘法
import numpy as np


def mul(A, B):
    _shape = A.shape
    _x, _y = _shape[0], _shape[1]
    # print(_x, _y)
    res = np.zeros(_x * _x).reshape(_x, _x)
    # 2,3
    for x in range(_x):
        _x_matrix = A[x]
        for b_y in range(_x):
            _y_matrix = np.zeros(_y)
            for y in range(_y):
                _y_matrix[y] = B[y][b_y]
            _res = 0
            for y in range(_y):
                _res += _x_matrix[y] * _y_matrix[y]
            res[x][b_y] = _res
    # print(res)
    return res


A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])
res = mul(A, B)
print(f"[res]:\n{res}")

# [res]:
# [[22. 28.]
#  [49. 64.]]
