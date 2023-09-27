import torch

import numpy as np
from matplotlib import pyplot as plt
scale = 1
A_plus = 0.8  # time difference is positive i.e negative reinforcement
A_minus = 0.3 # 0.01 # time difference is negative i.e positive reinforcement
tau_plus = 8
tau_minus = 5
w_max = 1.5*scale
w_min = -1.2*scale
sigma = 0.1 #0.02
# STDP reinforcement learning curve
a,b=[],[]
def rl(t):
    if t > 0:
        return -A_plus * np.exp(-float(t) / tau_plus)
    if t <= 0:
        return A_minus * np.exp(float(t) / tau_minus)

for i in range(-50,50,1):
    a.append(rl(i))
    b.append(i)
plt.plot(b,a,'.')
plt.show()

# STDP weight update rule
def update(w, del_w):
    if del_w < 0:
        return w + sigma * del_w * (w - abs(w_min)) * scale
    elif del_w > 0:
        return w + sigma * del_w * (w_max - w) * scale


# if __name__ == '__main__':
#     print
#     rl(-20) * par.sigma