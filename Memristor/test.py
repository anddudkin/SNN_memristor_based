import badcrossbar
import numpy as np
import matplotlib.pyplot as plt
import math

# plt.imshow(solution.voltages.word_line, cmap='gray_r', vmin=0, vmax=1, interpolation='None')
# plt.show()
r = 1000
u = []
for i in range(0, 10000):
    u.append(i / 10000)
I_l = []
for i in u:
    I_l.append(i / r)
plt.plot(u, I_l)
plt.show()

I_nl = []
for i in u:
    if i < 0.2:
        I_nl.append(i / r)
    else:
        I_nl.append((0.0002 * i * 2 ** (math.sqrt(i * 30))))
plt.plot(u, I_nl)
plt.show()

I_i = []
for i in u:
    I_i.append((0.0002 * i * 2 ** (math.sqrt(i * 30))))

plt.plot(u, I_i)
plt.show()
R=[0]
for i in range(1,len(u)):
    R.append(u[i]/I_i[i])
plt.plot(u,R)
plt.show()
