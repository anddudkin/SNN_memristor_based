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
        return -0.003


def plot_simple_stdp():
    """
    Plots simple STDP reinforcement learning curve
    """

    def compute_dw_plot(t, tau):
        """
        Computes dw
        """
        # if t < -tau_plus:
        #     return -0.0005
        if -10 <= t <= 0:
            return A_plus * np.exp(t / tau)
        else:
            return -0.003

    a, b = [], []
    for j in (1, 2, 4, 8, 10, 20, 30, 40, 10000):
        for i in range(-20, 10, 1):
            a.append(compute_dw_plot(i, j))
            b.append(i)
        plt.plot(b, a, label=f'tau={j}')
        a, b = [], []
    plt.axvline(x=0, color='k')
    plt.axhline(y=0, color='k')
    plt.legend()
    plt.show()


plot_simple_stdp()

# STDP weight update rule
"""def update(w, del_w):
    if del_w < 0:
        return w + sigma * del_w * (w - abs(w_min)) * scale
    elif del_w > 0:
        return w + sigma * del_w * (w_max - w) * scale
"""
