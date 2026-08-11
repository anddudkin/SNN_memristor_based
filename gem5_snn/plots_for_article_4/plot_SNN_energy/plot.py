from matplotlib import pyplot as plt


from Network.datasets import encoding_to_spikes
from badcrossbar.compute import compute
import numpy as np
import torch
from tqdm import tqdm
import pickle

n_neurons = [10,20,30,40,50,60,70,80,90,100,150,200]

e_crossbar = [86,112,148,169,188,195,220,235, 261,281,325,365]
e_crossbar = [x *25/1000 for x in e_crossbar]
plt.figure(figsize=(8, 4))
imp_mean = 130
e_neurons = []
e_neuron_act =[]
e_neuron_ref = []
for i in n_neurons:
    e_neurons.append(i*(1250*10**-6)*(5.6*10**-6)*10**6/50)
plt.semilogy(n_neurons, e_crossbar, 'k-', label='Кроссбар-массив', linewidth=1.5)           # сплошная
plt.semilogy(n_neurons, e_neurons, 'k--', label='Нейроны', linewidth=1.5)            # пунктирная
# plt.plot(n_values, energy_b_line, 'k:', label='Элементы столбцов', linewidth=1.5)             # точки
# plt.plot(n_values, sum_energy, 'k-', label='Общая энергия', linewidth=2)
plt.xlabel('Кол-во выходных нейронов', fontsize=12)
plt.ylabel('Энергия, мкДж/изображение', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.show()
e_sum=[]
for i in n_neurons:
    e_neuron_act.append((5.6*10**-6)*((1250*i-15*imp_mean)*10**-6)*10**9/50)
    e_neuron_ref.append(imp_mean*15*(10.5*10**-6)*10**-6*10**9/50)
    e_sum.append((5.6*10**-6)*((1250*i-15*imp_mean)*10**-6)*10**9+imp_mean*15*(10.5*10**-6)*10**-6*10**9/50)

plt.plot(n_neurons, e_sum, 'k-', label='Кроссбар-массив', linewidth=1.5)           # сплошная
plt.plot(n_neurons, e_neuron_ref, 'k--', label='Нейроны', linewidth=1.5)            # пунктирная
plt.plot(n_neurons, e_neuron_act, 'k:', label='Элементы столбцов', linewidth=1.5)
plt.show()