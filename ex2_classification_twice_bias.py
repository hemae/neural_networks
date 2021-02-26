import numpy as np


def act(x):
    return 0 if x < 0 else 1

def go(vect, w1, w2, w11, c):
    x = np.array(vect)
    weight1 = np.array([w1, w2]) # матрица 2х3
    weight2 = np.array([w11[0], w11[1]]) # вектор 1х3

    sum_hidden = np.dot(weight1, x) # вычисляем сумму на нейронах срытого слоя
    print("Суммы на нейронах сткрытого слоя: " + str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden]) # считаем функцию активации на каждой сумме
    print("Значения функции активации на выходе нейронов скрытого слоя" + str(out_hidden))

    sum_end = np.dot(weight2, out_hidden) + w11[2] * c
    y = act(sum_end)

    if y == 0:
        print("C2")
    elif y == 1:
        print("C1")
    else:
        print("Перепроверь прогу, чувак")

b1 = 1.5
b2 = 0.5
xb = 1
c = 1
z = -0.5

w11 = 1
w21 = w11
w31 = -w21 * b1 / xb

w12 = 1
w22 = w12
w32 = -w22 * b2 / xb

weight11 = [w11, w21, w31]
weight21 = [w12, w22, w32]

wa = -1
wb = 1
wc = z / c

weight12 = [wa, wb, wc]

x = 0
y = 0
init_vect = [0, 1, xb]
go(init_vect, weight11, weight21, weight12, c)