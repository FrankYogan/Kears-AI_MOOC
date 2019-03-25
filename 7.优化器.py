# -*- coding: utf-8 -*-
'''
直接从4.交叉熵复制过来，导入优化器，设置优化器，训练就可以了
'''

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (60000,28,28)
print('x_shape:', x_train.shape)
# (60000)
print('y_shape:', y_train.shape)

# (60000,28,28)->(60000,784)
x_train = x_train.reshape(x_train.shape[0], -1)/255.0
x_test = x_test.reshape(x_test.shape[0], -1)/255.0
# 换成 one hot格式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型，输入784个神经元，输出10个神经元
model = Sequential([
    Dense(units=10, input_dim=784, bias_initializer='one', activation='softmax')  # 输出神经元数量、输入神经元数量、偏置值、激活函数
])

# 定义优化器
sgd = SGD(lr=0.2)
adam = Adam(lr=0.001)

# 定义优化器，loss function， 训练过程中计算准确率
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',  # 交叉熵 收敛速度比较快
    metrics=['accuracy']
)

# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=10)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss', loss)
print('accuracy', accuracy)








