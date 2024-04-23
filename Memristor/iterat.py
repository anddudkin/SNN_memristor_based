import math

import badcrossbar
import numpy as np
import torch
from compute_crossbar import TransformToCrossbarBase
import matplotlib.pyplot as plt
applied_voltages = np.ones([196, 1])
torch.set_printoptions(threshold=10_000)
#w = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")
w = torch.load("G:/Другие компьютеры/Ноутбук/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")

# plt.imshow(w,cmap='gray', vmin=0, vmax=1, interpolation='None')
# plt.show()
# w1 = TransformToCrossbarBase(w,5000,25000,1)
# plt.imshow(w1.weights,cmap='gray_r', vmin=5000, vmax=25000, interpolation='None')
# plt.show()
#
# print(w1.weights)

V = [[0.5], [0.5], [0.5], [0.5]]
cr = [[15000, 24000], [18000, 22000], [18000, 22000], [18000, 22000]]

solution = badcrossbar.compute(V, cr, 500)
volt = solution.voltages.word_line
volt1 = solution.voltages.bit_line
cur = solution.currents.device


# print(cur)
# print(volt)
# print(torch.div(torch.tensor(volt), torch.tensor(cur)))
#
# for r in range(5000, 25000, 1000):
#     u, ii, i_lin = [], [], []
#     for i in range(0, 500):
#         i = i / 1000
#         u.append(i)
#         i_lin.append(i / r)
#         ii.append(2 / r * i * 10 ** (-1) * math.exp(5 * math.sqrt(i / 4)))
#     plt.plot(u, ii, color="k")
#     plt.plot(u, i_lin, "--", label=r)
# # plt.plot(u,ii)
# plt.legend(loc="upper left")
# plt.show()


# for r in range(5000, 25000, 1000):
#     u, ii, i_lin = [], [], []
#     for i in range(0, 500):
#         i = i / 1000
#         u.append(i)
#
#         i_lin.append(1 / r * i * 10 ** (-1) * math.exp(5 * math.sqrt(i / 4)))
#         ii.append(i/r)
#     plt.plot(u, ii, label=r)
#     plt.plot(u, i_lin, label=r)
# # plt.plot(u,ii)
# plt.legend(loc="upper left")
# plt.show()


def rtog(x):
    return 1 / float(x)
def gtor(x):
    return 1 / float(x)


V = np.ones([196, 1]) / 2
G= TransformToCrossbarBase(w, 5000, 25000, 1)
cr=G.weights
print(G.weights)
print(G.weights_Om)

# V = [[0.4], [0.2], [0.3], [0.5], [0.5]]
# cr = torch.tensor([[15000, 24000, 10000, 7000, 19000],
#                    [15000, 24000, 10000, 7000, 19000],
#                    [15000, 24000, 10000, 7000, 19000],
#                    [15000, 24000, 10000, 7000, 19000],
#                    [15000, 24000, 10000, 7000, 19000]], dtype=torch.float)
crG = torch.clone(cr)
crG.apply_(rtog)
o = 10 ** (-6)
k = 0
eps = 0
print("iterations stars......")
cr0 = torch.clone(crG)
flag = True
ll = []
while flag:
    if cr0[0][0] > 1:
        g_g = torch.clone(cr0)
    else:
        g_g = torch.clone(cr0.apply_(gtor))
    solution = badcrossbar.compute(V, g_g, 1)
    voltage = solution.voltages.word_line
    currents = solution.currents.device
    for i in range(len(cr0)):
        for j in range(len(cr0[0])):
            cr0[i][j] = 2 * 1 / crG[i][j] * 0.1 * math.exp(5 * math.sqrt(voltage[i][j] / 4))
    ll.append(solution.currents.device[1][1])
    det_g = torch.subtract(cr0, g_g)

    det_g = torch.abs(det_g)

    eps = torch.max(det_g) / (torch.max(g_g))

    print(eps)

    if eps < o:
        flag = False
        #print(solution.voltages.word_line)
        print(solution.currents.device)
plt.semilogy(ll[1:])
plt.show()

#print(solution.currents.device)
