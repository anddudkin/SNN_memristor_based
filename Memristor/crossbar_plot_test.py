import badcrossbar
import numpy as np
import matplotlib.pyplot as plt


def g():
    in_n = 130
    out_n = 100
    s = []
    s1 = []
    x = [5, 10, 20, 40, 50, 60, 70, 80, 90, 100, 150, 200]
    for j in (5, 10, 50, 100, 200, 300, 500):
        for i in (5, 10, 20, 40, 50, 60, 70, 80, 90, 100, 150, 200):
            applied_voltages = np.ones([j, 1])
            w = np.random.randint(5000, 25000, (j, i))
            r_i = 1
            solution = badcrossbar.compute(applied_voltages, w, r_i)
            s.append(round(np.sum(solution.voltages.word_line) / i / j, 3))
            s1.append(round(np.min(solution.voltages.word_line), 3))
        plt.plot(x, s1)
        col = (np.random.random(), np.random.random(), np.random.random())
        plt.plot(x, s, '--', label=j, c=col)
        print(s)
        print(s1)
        s = []
        s1 = []
    print(s)
    print(s1)
    plt.legend()
    plt.show()


# g()
def g1():
    in_n = 50
    out_n = 100
    s = []
    s1 = []
    x = [5, 10, 20, 40, 50, 60, 70, 80, 90, 100, 150, 200]
    for j in (0, 0.05, 0.1,0.5,1,1.5, 2):
        for i in (5, 10, 20, 40, 50, 60, 70, 80, 90, 100, 150, 200):
            applied_voltages = np.ones([in_n, 1])
            w = np.random.randint(5000, 25000, (in_n, i))

            solution = badcrossbar.compute(applied_voltages, w, j)
            s.append(round(np.sum(solution.voltages.word_line) / i / in_n, 3))
            s1.append(round(np.min(solution.voltages.word_line), 3))
        plt.plot(x, s,label=j)
        col = (np.random.random(), np.random.random(), np.random.random())
        #plt.plot(x, s, '--', label=j, c=col)
        print(s)
        print(s1)
        s = []
        s1 = []
    print(s)
    print(s1)
    plt.legend()
    plt.show()
g1()

import torch

import seaborn as sns
"""распределение весов"""
#w = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/weights_tensor.pt")
w = torch.load("G:/Другие компьютеры/Ноутбук/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")
w = torch.squeeze(w.reshape(1, 9800))
sns.histplot(w, stat="percent",bins = 100)
plt.show()
sns.kdeplot(w,cut = 0)
plt.show()
g = []
for i in w:
    if i > 0.75:
        g.append(float(i))
print(g)
sns.histplot(g, stat="percent", bins = 100)
plt.show()
sns.kdeplot(g, cut = 0)
plt.show()