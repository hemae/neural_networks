# Офф. док. Keras: https://keras.io/api/optimizers/adam/​
# Русскоязычная по Keras: https://ru-keras.com/optimizer/


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# загружаем данные из модуля mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# нормируем данные на 1
x_train = x_train / 255
x_test = x_test / 255

# трансофрмируем выходные сигналы в список следующим образом: 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] - и тд
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

limit = 5000
x_train_data = x_train[:limit]
y_train_data = y_train_cat[:limit]
x_valid = x_train[limit:limit * 2]
y_valid = y_train_cat[limit:limit * 2]

# вручную формируем обучающую выборку и выборку валидации
# size_val = 10000
# x_val_split = x_train[:size_val]
# y_val_split = y_train_cat[:size_val]
# x_train_split = x_train[size_val:]
# y_train_split = y_train_cat[size_val:]

# случайным образом формируем обучающую и валидационную выборки с помощью sklearn (20%)
# x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_cat, test_size=0.2)

# plt.figure(figsize=(10,5))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#
# plt.show()

# строим модель: полносвязная нейронная сеть, картинка 28x28, 28*28+1=785 входных сигналов, 1 скрытый слой: 128 нейронов с функцией активации ReLu,
# выходной слой: 10 нейронов с функцией активации SoftMax - потому что 10 возможных цифр, предсказываем вектор типа [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
# где каждый элемент - вероятность того, что предсказана цифра, в данном случае 5
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    # Dense(128, activation='relu'),
    Dense(300, activation='relu'),
    BatchNormalization(),
    # Dropout(0.8),
    Dense(10, activation='softmax')
])

# вывести модель сети в текстовом виде, полезно посмотреть
# 100480 весов связей между входным слоем и 1-м скрытым, 1290 ((128 + 1) * 10) весов связей между 1-м скрытым и выходным слоями
print(model.summary())

# компиляция модели: оптимизация по Адаму, потери: категориальная кросс-энтропия, метрика по точности
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# компиляция модели: оптимизация SGD по Нестерову, потери: категориальная кросс-энтропия, метрика по точности
myOpt = keras.optimizers.SGD(
    learning_rate=0.1,
    momentum=0.0,
    nesterov=True
)
model.compile(
    optimizer=myOpt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# обучаем сеть: делаем 32 батча, 50 эпох, валидационную выборку из (x_valid, y_valid)
# всего берём в обучающую выборку limit данных, в валидационную следующие limit данных
his = model.fit(x_train_data, y_train_data, batch_size=32, epochs=50, validation_data=(x_valid, y_valid))

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.show()

# обучаем сеть: делаем 32 батча, 5 эпох, 20% данных из обучающей выборки берём в валидационную случайным образом
# model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

# обучаем сеть: делаем 32 батча, 5 эпох, обучающая и валидационная выборки созданы вручную или с помощью sklearn
# model.fit(x_train_split, y_train_split, batch_size=32, epochs=5, validation_data=(x_val_split, y_val_split))

# model.evaluate(x_test, y_test_cat)

n = 2
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f'Распознанная цифра: {np.argmax(res)}')

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

# print(pred.shape)

# print(pred[:20])
# print(y_test[:20])

mask = pred == y_test
# print(mask[:10])

x_false = x_test[~mask]
p_false = pred[~mask]

print(x_false.shape)

# for i in range(10):
#     print('Значение сети: ' + str(p_false[i]))
#     plt.imshow(x_false[i], cmap=plt.cm.binary)
#     plt.show()