import math
import pickle
import statistics
import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import exp

w = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")
j1 = plt.imshow(w, cmap='gray_r', vmin=0, vmax=1,
                        interpolation='None')
plt.colorbar(j1, fraction=0.12, pad=0.04)
plt.show()
d = {}
from Memristor import compute_crossbar
from compute_crossbar import TransformToCrossbarBase

r = [4681, 2444, 1654, 1004, 506, 390, 267, 254]
c = TransformToCrossbarBase(w, 5000, 25000, 1)
print(c.weights_Om)
c.plot_crossbar_weights()

c.transform_with_experemental_data(r)
print(c.weights_Om)
c.plot_crossbar_weights()
print(r)
g = []
for i in range(len(c.weights_Om)):
    for j in range(len(c.weights_Om[0])):
        if c.weights_Om[i][j] not in g:
            g.append(int(c.weights_Om[i][j]))
print("эксперимент: ",r)
print("в кроссбаре: ", sorted(g))

# Поиск ближайшего значения
def nearest_value(R_real, value):
    '''Поиск ближайшего значения до value в списке items'''
    found = R_real[0]  # найденное значение (первоначально первое)
    for i, item in enumerate(R_real):
        if abs(item - value) < abs(found - value):
            found = item
    return found
