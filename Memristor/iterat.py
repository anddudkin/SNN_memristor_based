import math

import badcrossbar
import numpy as np
import torch
from compute_crossbar import TransformToCrossbarBase
import matplotlib.pyplot as plt

applied_voltages = np.ones([196, 1])
torch.set_printoptions(threshold=10_000)
w = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")
# w = torch.load("G:/Другие компьютеры/Ноутбук/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")

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


u, ii, i_lin,g = [], [], [],[]
for i in range(1, 500000):
    i = i / 1000000
    u.append(i)

    ii.append(1 / 25000 * i * 10 ** (-1) * math.exp(5 * math.sqrt(i / 4)))
    i_lin.append(i / 25000)
    g.append((1 / 25000 * i * 10 ** (-1) * math.exp(5 * math.sqrt(i / 4)))/i)
vax=dict(zip(u, ii))
print(vax.keys())
# plt.plot(u, ii)
plt.plot(u, i_lin)
plt.plot(u,ii)

# plt.plot(u,ii)


def rtog(x):
    return 1 / float(x)


def gtor(x):
    return 1 / float(x)


V = np.ones([196, 1]) / 2
# w = torch.tensor(np.random.rand(196, 50),dtype=torch.float)
G = TransformToCrossbarBase(w, 5000, 25000, 0)

# print(G.weights_Siemens)
# print(G.weights_Om)
# print("1111")

# V = [[0.4], [0.2], [0.3], [0.5], [0.5]]
# cr = torch.tensor([[15000, 24000, 10000, 7000, 19000],
#                    [15000, 24000, 10000, 7000, 19000],
#                    [15000, 24000, 10000, 7000, 19000],
#                    [15000, 24000, 10000, 7000, 19000],
#                    [15000, 24000, 10000, 7000, 19000]], dtype=torch.float)
crR = G.weights_Om
o = 10 ** (-5)
k = 0
eps = 0
print("iterations stars......")
cr0 = crR.clone().detach()
flag = True
sol = None
g_iter = None
gr_v, gr_i, gr_g = [], [], []


while flag:
    if cr0[0][0] > 1:
        g_g = cr0.clone().detach()
    else:
        g_g = (cr0.apply_(gtor)).clone().detach()
    solution = badcrossbar.compute(V, g_g, 1)
    voltage = torch.subtract(torch.tensor(solution.voltages.word_line, dtype=torch.float),
                             torch.tensor(solution.voltages.bit_line, dtype=torch.float))
    currents = torch.tensor(solution.currents.device, dtype=torch.float)
    ####
    gr_v.append(voltage[0][0])
    gr_i.append(currents[0][0])
    for i in range(len(cr0)):
        for j in range(len(cr0[0])):
            # i_lin.append(1 / 25000 * i * 10 ** (-1) * math.exp(5 * math.sqrt(i / 4)))
            #cr0[i][j] = (crR[i][j]* 0.1 * math.exp(5 * math.sqrt(voltage[i][j] / 4)))/voltage[i][j]
            cr0[i][j] = voltage[i][j]/vax[round(float(voltage[i][j]),6)]
    gr_g.append(cr0[0][0])
    det_g = torch.subtract(cr0, g_g)

    det_g = torch.abs(det_g)

    eps = torch.max(det_g) / (torch.max(g_g))

    print(eps)

    if eps < o:
        flag = False
        # print(solution.voltages.word_line)
        sol = solution
        g_iter = cr0
        print(solution.currents.device)
        print(g_iter)
plt.plot(gr_v, gr_i,"--")
n=[]
for i in range(len(gr_v)):
    n.append(i)
for i, txt in enumerate(n):
    plt.annotate(txt, (gr_v[i], gr_i[i]),fontsize=12)
plt.show()
print(torch.add(torch.tensor(solution.voltages.word_line, dtype=torch.float),
                torch.tensor(solution.voltages.bit_line, dtype=torch.float)))
print(torch.mul(torch.tensor(solution.currents.device, dtype=torch.float), torch.tensor(g_iter, dtype=torch.float)))
# print(solution.currents.device)
i_d = torch.tensor(solution.currents.device, dtype=torch.float)
i_bl = torch.tensor(solution.currents.bit_line, dtype=torch.float)
i_wl = torch.tensor(solution.currents.word_line, dtype=torch.float)
r_l = torch.ones([196, 50], dtype=torch.float)
print(g_iter)
v_d = torch.mul(i_d, g_iter)
v_bl = torch.mul(i_bl, r_l)
v_wl = torch.mul(i_wl, r_l)
v_s = 0
# for 1st bitline
for i in range(196):
    v_s += v_bl[i][0]
print(v_s + v_d[0][0] + v_wl[0][0])
first_col = []
for i in range(196):
    v_s = 0
    for j in range(0 + i, 196):
        v_s += v_bl[j][0]
    first_col.append(v_s + v_d[i][0] + v_wl[i][0])
print(first_col)
