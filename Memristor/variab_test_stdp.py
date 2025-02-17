import pickle

import numpy as np
from matplotlib import pyplot as plt

from Network.learning import plot_simple_stdp
with open('Res_states.pkl', 'rb') as f:
    r = pickle.load(f)
print(r)
A_plus = 20  # positive reinforcement
print(len(r))
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
            return -4

    a, b = [], []
    gg = np.linspace(-20, 20, num=500)

    for i in gg:
        a.append(compute_dw_plot(i, 5))
        b.append(i)
    plt.plot(b, a, label=f'tau={5}')

    plt.axvline(x=0, color='k')
    plt.axhline(y=0, color='k')
    for i in range(-4,21):
        plt.axhline(y=i, color='r', linestyle="--", linewidth=1)
    plt.xlabel(r't$_{\mathrm{pre}}$ - t$_{\mathrm{post}}$ (ms)')
    plt.ylabel(r'$\Delta$W', fontsize=12)
    plt.legend()
    plt.show()
    return [a,b]
x=plot_simple_stdp()
y1=x[0]
x1=x[1]
print(y1)
print(x1)
y2=[]
for i in y1:
    y2.append(i/max(y1))
plt.plot(x1,y2)
plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')

plt.show()