# -*- coding: utf-8 -*-

'''
# 载入模型
model = load_model('model.h5')
载入之后可以直接对模型进行评估
'''

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model

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

# 载入模型
model = load_model('model.h5')

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss', loss)
print('accuracy', accuracy)

'''
加载模型基础上，继续训练两代
'''

# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=2)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss', loss)
print('accuracy', accuracy)


# 保存模型
model.save('model.h5')  #HDF5文件，pip intall h5py








