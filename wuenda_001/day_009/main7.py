# Tensorflow 实现
# 创建模型API导入
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 损失函数API导入，二元交叉熵损失函数
from tensorflow.keras.losses import BinaryCrossentropy

model = Sequential(
    [
        Dense(units=25, activation="sigmoid"),
        Dense(units=15, activation="sigmoid"),
        Dense(units=1, activation="sigmoid"),
    ]
)

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
model.compile(loss=BinaryCrossentropy())
model.fit(X, Y, epochs=1000)
