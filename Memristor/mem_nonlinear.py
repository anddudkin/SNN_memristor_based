import torch
import matplotlib.pyplot as plt
import seaborn as sns

w = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/weights_tensor.pt")
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
'''
plt.hist(w.reshape(1,9800), color='lightgreen', ec='black', bins=15)
plt.show()
y, x = [], []
for j, i in enumerate(range(0, 1000)):
        x.append(i / 1000)
        y.append((i / 1000) ** 2/100)
plt.plot(x, y)
plt.show()
import numpy as np
import math
'''
