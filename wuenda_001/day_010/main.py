import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model = Sequential([Dense(25, "relu"), Dense(15, "relu"), Dense(10, "softmax")])
model.compile(loss=SparseCategoricalCrossentropy())
