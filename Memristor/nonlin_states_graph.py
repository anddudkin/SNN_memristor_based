import math
import pickle
import statistics
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import exp

import badcrossbar
from Memristor import compute_crossbar
from compute_crossbar import TransformToCrossbarBase
import torch



with open('interp_coeff.pkl', 'rb') as f:
    args2 = pickle.load(f)
print(args2)
v=[]
for i in range(7000):
    v.append(i / 10000)
res_std=1.9546164134240556*10**-6
fig, ax = plt.subplots()
k = 1
k1 = 1
k2 = 1
x = np.array(range(100))
new, new1 = [], []
f=0.9
ff=1
f1,f2=[],[]
for i in range(20):
    t1, t2, t3, t4, t5, t6 = [], [], [], [], [], []

    for i in v:
        t1.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2] + (k1 - 1) * res_std)
    for i in v:
        t2.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2] + k1 * res_std)
    new.append(t2)
    for i in v:
        t3.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2] + (k1 + 1) * res_std)

    t1[0] = 0
    t2[0] = 0
    t3[0] = 0
    v[0]=0
    r, g, b = None, None, None
    for ggg in range(11):
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
        y = ggg + np.random.rand(100) * 0.25
    p1=ax.semilogy(v[1:], t2[1:], linestyle='dashed', linewidth=f, color=[r, g, b],label=str(ff) + " состояние")
    #ax.plot(v, t2, '--', color="black", linewidth=0.5)
    ax.fill_between(v[1:], t2[1:], t1[1:], alpha=0.1, color=[r, g, b])
    ax.fill_between(v[1:], t3[1:], t2[1:], alpha=0.1, color=[r, g, b])
    #ax.legend()
    p2= ax.fill(np.NaN, np.NaN, color=[r, g, b], alpha=0.5)

    f1.append((p2[0], p1[0]))
    ax.legend([(p2[0], p1[0]),])
    f2.append(str(ff) + " состояние")
    k1 += 2
    f-=0.02
    ff+=1
ax.legend(f1,f2,ncol=2)
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.tick_params(which='minor', direction="in")
#plt.legend(ncol=3)
plt.xlabel("Напряжение, В")
plt.ylabel("Ток, А")
plt.tight_layout()
plt.savefig("6", dpi=300)
plt.show()