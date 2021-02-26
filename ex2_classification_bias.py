import numpy as np
import matplotlib.pyplot as plt


N = 5
b = 3
xb = 1

x1 = np.random.random(N) # генерим иксы точек класса С1
x2 = x1 + [np.random.randint(10) / 10 for i in range(N)] + b# генерим игреки
C1 = [x1, x2] # иксы и игреки точек класса C1

x1 = np.random.random(N) # аналогично
x2 = x1 - [np.random.randint(10) / 10 for i in range(N)] - 0.1 + b
C2 = [x1, x2]

f = [0 + b, 1 + b] # просто прямая

w1 = 1 # веса выбраны так, чтобы была прямая с углом наклона 45
w2 = - w1   # поднятая на b3
w3 = - w2 * b / xb
w = np.array([w1, w2, w3])
for i in range(N): #
    x = np.array([C2[0][i], C2[1][i], xb])
    y = np.dot(w, x) # вычисляем выходное значение y для i-точки
    if y >= 0:
        print("C1")
    else:
        print("С2")

plt.scatter(C1[0][:], C1[1][:], s=10, c="red")
plt.scatter(C2[0][:], C2[1][:], s=10, c="green")
plt.plot(f)
plt.grid(True)
plt.show()