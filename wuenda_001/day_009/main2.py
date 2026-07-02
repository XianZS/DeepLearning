import tensorflow as tf
import numpy as np

# 标准二维输入：(样本数, 特征数)，不要多一层中括号
x = np.array([[200.0, 17.0]])
x = tf.convert_to_tensor(x)  # 显式转换，解决类型提示

layer_1 = tf.keras.layers.Dense(units=3, activation="sigmoid")
a1 = layer_1(x)

layer_2 = tf.keras.layers.Dense(units=1, activation="sigmoid")
a2 = layer_2(a1)
print(a2.numpy())
