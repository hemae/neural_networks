import numpy as np
import matplotlib.pyplot as plt
import random


def log_func(x):
    return 1 / (1 + np.exp(-x))


def der_log_func(x):
    return log_func(x) * (1 - log_func(x))


def loc_gradient(summ_in_signal, err):
    return err * der_log_func(summ_in_signal)


def neuron_out_signal(w, x):
    summ_in_signal = np.dot(w, x)
    out_signal = log_func(summ_in_signal)

    return out_signal


def my_network(x, w11, w12, w21, f):  # wij - i номер слоя, к которому подходят связи, j - номер нейрона, к которому подходят связи, wij - вектор (набор весов приходящих связей)
    f[0][0] = neuron_out_signal(w11, x)
    f[0][1] = neuron_out_signal(w12, x)
    f[1] = neuron_out_signal(w21, f[0])

    return f[1]  # float


def back_prop_my_network(x, w11, w12, w21, f, err, converg_step):
    def update_w(w, step, grad, f_lay):
        return w - step * grad * f_lay

    delta = err * f[1] * (1 - f[1])
    w21 = update_w(w21, converg_step, delta, f[0])
    # print(w21)

    delta1 = delta * w21 * f[1] * (1 - f[1])
    # print(delta1)
    w11 = update_w(w11, converg_step, delta1[0], x)
    w12 = update_w(w12, converg_step, delta1[1], x)

    return w11, w12, w21


lamd = 1  # шаг сходимости при корректировке весов
N = 1000000  # количество итераций

w11 = np.array([random.random() - 0.5, random.random() - 0.5,
                random.random() - 0.5])  # 3 # вектора начальных весов каждого нейрона сети
w12 = np.array([random.random() - 0.5, random.random() - 0.5, random.random() - 0.5])  # 3

w21 = np.array([random.random() - 0.5, random.random() - 0.5])  # 2

f = [np.array([1.0, 1.0]), 1.0]
# print(f)

input_data = [
    [np.array([0, 0, 0]), 0],
    [np.array([0, 0, 1]), 1],
    [np.array([0, 1, 0]), 0],
    [np.array([0, 1, 1]), 1],
    [np.array([1, 0, 0]), 0],
    [np.array([1, 0, 1]), 1],
    [np.array([1, 1, 0]), 0],
    [np.array([1, 1, 1]), 0]
]

error_array = []

for i in range(N):
    index = random.randint(0, len(input_data) - 1)
    x = input_data[index][0]
    d = input_data[index][1]
    f[1] = my_network(x, w11, w12, w21, f)  # выходное значение сети
    error = f[1] - d
    error_array.append(error)
    w11, w12, w21 = back_prop_my_network(x, w11, w12, w21, f, error,
                                         lamd)  # корректировка весов (изменение глобальных переменных)
    # print(f)
    # print(w11)

# summ = 0
# for i in range(len(input_data)):
#     x = input_data[i][0]
#     d = input_data[i][1]
#     f[1] = my_network(x, w11, w12, w21, f)
#     print(f[1])
#     error = f[1] - d
#     summ += error

# average_error = summ / len(input_data)
# print(average_error)

x = np.array([1, 1, 1])
y = my_network(x, w11, w12, w21, f)
print(y)

plt.plot(error_array)
plt.show()
