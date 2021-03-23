import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from tensorflow import keras

from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization

def get_data(N, M):
    def get_random_data(N, from_value=0, to_value=1):
        data_range = to_value - from_value
        random_data = np.random.rand(N) * data_range + from_value

        return random_data

    R = 8.314

    # создаём обучающую выборку N векторов входных данных
    error = 0.05    # имитированная ошибка при измерении всех величин 5%
    substance_amount = get_random_data(N, 4, 6)
    temperature = get_random_data(N, 200, 400)
    volume = get_random_data(N, 1, 10)
    pressure = R * substance_amount * temperature / volume

    x_train = np.transpose(np.array([substance_amount, temperature, volume]))
    y_train = pressure

    # создаём тестовую выборку M векторов входных данных
    error = 0.06  # имитированная ошибка при измерении всех величин 5%
    substance_amount = get_random_data(M, 6, 8)
    temperature = get_random_data(M, 200, 450)
    volume = get_random_data(M, 2, 12)
    pressure = R * substance_amount * temperature / volume

    x_test = np.transpose(np.array([substance_amount, temperature, volume]))
    y_test = pressure

    return x_train, x_test, y_train, y_test


N = 10000  # размер обучающей выборки
M = 1000   # размер тестовой выборки
x_train, x_test, y_train, y_test = get_data(N, M)

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)

# print(x_train)
# print(x_test)

# случайным образом формируем обучающую и валидационную выборки с помощью sklearn (20%)
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, test_size=0.2)

# model = keras.Sequential()
# model.add(Dense(units=1, input_shape=(3,), activation='linear'))
# model.compile(loss='mean_squared_error', optimizer='adam')
# # print(model.summary())

model = keras.Sequential([
    Dense(units=1, input_shape=(3,), activation='linear'),
    Dense(1, activation='linear'),
    # Dense(300, activation='relu'),
    # BatchNormalization(),
    # Dropout(0.8)
])
print(model.summary())

# компиляция модели: оптимизация по Адаму, потери: наименьшие квадраты
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

model.fit(x_train_split, y_train_split, batch_size=100, epochs=10, validation_data=(x_val_split, y_val_split))

pred = np.transpose(model.predict(x_test))[0]
# print(pred)
# print(y_test)
# y_test = np.transpose(y_test)

percent_difference = (y_test - pred) / y_test * 100
mean_percent_difference = np.mean(percent_difference)

print(mean_percent_difference)