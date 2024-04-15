import matplotlib.pyplot as plt
v = []
I = []
Ron = 100
Roff = 16000
mu = 1 * 10 ** (-14)
w = 1 * 10 ** (-9)
D = 1 * 10 ** (-8)

for i in range(1000):
    v.append(i / 1000)
for i in v:
    I.append(i**2 / (Ron * w / D + Roff * (1 - w)))

plt.plot(v,I)
plt.show()