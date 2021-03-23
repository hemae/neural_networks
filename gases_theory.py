import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt


def get_data(N, M):
    R = 8.314
    # создаём обучающую выборку N векторов входных данных

    substance_amount = np.random.rand(N) * 2 + 4
    substance_amount = substance_amount + (np.random.rand(N) - 0.5) * 2 * 0.05  # имитация ошибки при измерении

    temperature = np.random.rand(N) * 200 + 200
    temperature = temperature + (np.random.rand(N) - 0.5) * 200 * 0.05  # имитация ошибки при измерении

    volume = np.random.rand(N) * 9 + 1
    volume = volume + (np.random.rand(N) - 0.5) * 9 * 0.1  # имитация ошибки при измерении

    pressure = R * substance_amount * temperature / volume

    x_train = np.transpose(np.array([substance_amount, temperature, volume]))
    y_train = pressure

    # print(x_train)
    # print(y_train)

    # создаём тестовую выборку M векторов входных данных
    substance_amount = np.random.rand(M) * 2 + 6
    substance_amount = substance_amount + (np.random.rand(M) - 0.5) * 2 * 0.07  # имитация ошибки при измерении

    temperature = np.random.rand(M) * 250 + 200
    temperature = temperature + (np.random.rand(M) - 0.5) * 250 * 0.04  # имитация ошибки при измерении

    volume = np.random.rand(M) * 10 + 2
    volume = volume + (np.random.rand(M) - 0.5) * 10 * 0.08  # имитация ошибки при измерении

    pressure = R * substance_amount * temperature / volume

    x_test = np.transpose(np.array([substance_amount, temperature, volume]))
    y_test = pressure

    # print(x_test)
    # print(y_test)

    return x_train, x_test, y_train, y_test


def normalization(np_list):
    np_list = np.transpose(np_list)
    for i in range(len(np_list)):
        np_list[i] = np_list[i] / max(np_list[i])
    return np.transpose(np_list)


N = 1000  # размер обучающей выборки
M = 100   # размер тестовой выборки
x_train, x_test, y_train, y_test = get_data(N, M)

x_train = normalization(x_train)
x_test = normalization(x_test)
y_mean = np.mean(y_train)
y_train = y_train / max(y_train)

# print(x_train)
# print(x_test)

# случайным образом формируем обучающую и валидационную выборки с помощью sklearn (20%)
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, test_size=0.2)

model = keras.Sequential()
model.add(Dense(units=10, input_shape=(3,), activation='relu'))
model.add(Dense(units=10, input_shape=(10,), activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
# print(model.summary())

model.fit(x_train_split, y_train_split, batch_size=5, epochs=5, validation_data=(x_val_split, y_val_split))

pred = np.transpose(model.predict(x_test))[0]
# print(pred)
# print(y_test)
# y_test = np.transpose(y_test)

percent_difference = (y_test - pred) / y_test * 100
mean_percent_difference = np.mean(percent_difference)

print(mean_percent_difference)