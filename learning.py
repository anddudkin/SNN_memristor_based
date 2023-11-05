import torch

import numpy as np
from matplotlib import pyplot as plt


A_plus = 0.005  # positive reinforcement
A_minus = 0.005  # negative reinforcement
tau_plus = 10
tau_minus = 10



def compute_dw(t):
    """
    Computes dw
    """
    # if t < -tau_plus:
    #     return -0.0005
    if -10 <= t <= 0:
        return A_plus * np.exp(t / tau_plus)
    else:
        return -0.005



def plot_stdp():
    """
    Plots STDP reinforcement learning curve
    """
    a, b = [], []
    for i in range(-100, 100, 1):
        a.append(compute_dw(i))
        b.append(i)
    plt.plot(b, a, '.')
    plt.show()
plot_stdp()

# STDP weight update rule
"""def update(w, del_w):
    if del_w < 0:
        return w + sigma * del_w * (w - abs(w_min)) * scale
    elif del_w > 0:
        return w + sigma * del_w * (w_max - w) * scale
"""
