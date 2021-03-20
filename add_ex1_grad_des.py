import numpy as np
import matplotlib.pyplot as plt
import time


def f(x):
    return x * x - 5 * x + 5

def df(x):
    return 2 * x - 5


N = 100  # число итераций
xn = 0  # начальное значение x
lmd = 0.8   # шаг сходимости (коэффициент)

x_plt = np.arange(0, 5.0, 0.1)
f_plt = [f(x) for x in x_plt]

plt.ion()   # включаем интерактивный режим
fig, ax = plt.subplots()    # создаем окно и оси для графика
ax.grid(True)   # отображаем сетку на графике (осях)

ax.plot(x_plt, f_plt)   # отображаем начальный график на осях
point = ax.scatter(xn, f(xn), c='r')    # отображаем начальную точку на графике

mn = 100
for i in range(N):
    lmd = 1 / (min(i + 1, mn))
    xn = xn - lmd * np.sign(df(xn))  # изменяем аргумент согласно алгоритму град.спуска
    point.set_offsets([xn, f(xn)])  # обновляем положение точки

    # перерисовываем график при задержке 20 мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.002)

plt.ioff()  # выключаем интерактивный режим
print(xn)
ax.scatter(xn, f(xn), c='g')    # отрисовываем конечное положение точки
plt.show()
