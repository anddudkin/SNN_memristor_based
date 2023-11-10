import torch

import numpy as np
from matplotlib import pyplot as plt

""" Simple STDP learning rule"""

A_plus = 0.005  # positive reinforcement
tau_plus = 10


def compute_dw(t):
    """ Computes dw """
    if -10 <= t <= 0:
        return A_plus * np.exp(t / tau_plus)
    else:
        return -0.003


def plot_simple_stdp():
    """Plots simple STDP reinforcement learning curve"""

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
    gg = np.linspace(-20, 20, num=500)
    for j in (1, 2, 4, 8, 10, 20, 30, 40):
        for i in gg:
            a.append(compute_dw_plot(i, j))
            b.append(i)
        plt.plot(b, a, label=f'tau={j}')
        a, b = [], []
    plt.axvline(x=0, color='k')
    plt.axhline(y=0, color='k')
    plt.xlabel(r't$_{\mathrm{pre}}$ - t$_{\mathrm{post}}$ (ms)')
    plt.ylabel(r'$\Delta$W', fontsize=12)
    plt.legend()
    plt.show()





def plot_classic_STDP():
    A_plus_ = 0.005  # positive reinforcement
    A_minus = 0.003  # negative reinforcement
    gg = np.linspace(-20, 20, num=500)
    def compute_dw_classic(t, tt):
        """
        Computes dw
        """
        if t <= 0:
            return A_plus_ * np.exp(t / tt)
        else:
            return -A_minus * np.exp(-t / tt)

    a, b = [], []
    for j in (1, 4, 8, 10, 30):
        for i in gg:
            a.append(compute_dw_classic(i, j))
            b.append(i)
        plt.plot(b, a, "--", label=f'tau={j}')
        a, b = [], []
    plt.axvline(x=0, color='k')
    plt.axhline(y=0, color='k')
    plt.xlabel(r't$_{\mathrm{pre}}$ - t$_{\mathrm{post}}$ (ms)')
    plt.ylabel(r'$\Delta$W', fontsize=12)
    plt.legend()
    plt.show()



