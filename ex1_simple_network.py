import numpy as np


def act(x):
    return 0 if x < 0.5 else 1

def go(vect):
    x = np.array(vect)
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12]) # матрица 2х3
    weight2 = np.array([-1, 1]) # вектор 1х3

    sum_hidden = np.dot(weight1, x) # вычисляем сумму на нейронах срытого слоя
    print("Суммы на нейронах сткрытого слоя: " + str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden]) # считаем функцию активации на каждой сумме
    print("Значения функции активации на выходе нейронов скрытого слоя" + str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)

    if y == 0:
        print("Облом")
    elif y == 1:
        print("Фак е")
    else:
        print("Перепроверь прогу, чувак")


init_vect = [1, 1, 0]
go(init_vect)