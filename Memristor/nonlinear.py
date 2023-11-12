from math import exp
from random import randrange
import matplotlib.pyplot as plt

""" Currently under development """

l, g, k, f, f2, f1, f3 = [], [], [], [], [], [], []
for i in range(100):
    g.append(i)
    y = i + 0
    l.append(y)

for i in range(100):
    y = 100 * (1 - exp(-i / 15))
    y1 = 100 * (1 - exp(-i / 12))
    y2 = 100 * (1 - exp(-i / 20))
    f1.append(y1)
    f3.append(y2)
    f2.append(y)
    y += randrange(-10, 10)
    f.append(y)
j1, j2 = [], []
for i in range(100, 0, -1):
    y = i + 0
    j1.append(y)
gg = []
for i in range(100, 200):
    gg.append(i)
for i in range(100):
    y = -100 * (1 - exp(-i / 15)) + 100
    j2.append(y)

plt.xlim(0, 210)

plt.plot(g, l, "--", f1)
plt.plot(g, f, f2)
plt.plot(g, f3)
plt.plot(gg, j1, "--")
plt.plot(gg, j2)

plt.show()
