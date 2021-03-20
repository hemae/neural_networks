import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

def g():
    return random.random()

x_all = []
f_all = []
for j in range (10):
    c = np.array([5, 12, 7, 342, 85, 6, 0, 78, 65, 1254])
    f = np.array([0, 7, 2, 337, 80, 1, -5, 73, 60, 1249])

    model = keras.Sequential()
    model.add(Dense(units=1, input_shape=(1,), activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

    history = model.fit(c, f, epochs=500, verbose=0)

    # plt.plot(history.history['loss'])
    # plt.grid(True)
    # plt.show()
    print(model.get_weights())
    print(model.predict([45])[0])

    x = []
    f = []
    for i in range(10):
        f.append((model.predict([i * 10]))[0])
        x.append(i * 10)

    x_all.append(x)
    f_all.append(f)

for j in range(10):
    plt.plot(x_all[j], f_all[j])
plt.grid(True)
plt.show()