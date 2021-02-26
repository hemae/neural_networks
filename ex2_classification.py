import numpy as np
import matplotlib.pyplot as plt


N = 5

x1 = np.random.random(N) # генерим иксы точек класса С1
x2 = x1 + [np.random.randint(10) / 10 for i in range(N)] # генерим игреки
C1 = [x1, x2] # иксы и игреки точек класса C1

x1 = np.random.random(N) # аналогично
x2 = x1 - [np.random.randint(10) / 10 for i in range(N)] - 0.1
C2 = [x1, x2]

f = [0, 1] # просто прямая

w = np.array([-1, 1]) # веса
for i in range(N): #
    x = np.array([C2[0][i], C2[1][i]])
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