import torch

import numpy as np
from matplotlib import pyplot as plt

scale = 1
A_plus = 0.01  # positive reinforcement
A_minus = 0.001  # negative reinforcement
tau_plus = 5
tau_minus = 5
w_max = 1.5 * scale
w_min = -1.5 * scale


#


def compute_dw(t):
    """
    Computes dw
    """
    if t <= 0:
        return A_plus * np.exp(t / tau_plus)
    else:
        return -A_minus * np.exp(-t / tau_minus)


def plot_stdp():
    """
    Plots STDP reinforcement learning curve
    """
    a, b = [], []
    for i in range(-20, 20, 1):
        a.append(compute_dw(i))
        b.append(i)
    plt.plot(b, a, '.')
    plt.show()


# STDP weight update rule
"""def update(w, del_w):
    if del_w < 0:
        return w + sigma * del_w * (w - abs(w_min)) * scale
    elif del_w > 0:
        return w + sigma * del_w * (w_max - w) * scale
"""