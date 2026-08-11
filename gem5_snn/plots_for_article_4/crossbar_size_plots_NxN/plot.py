from matplotlib import pyplot as plt


import numpy as np


data = np.load('energy_data.npz')
n_values = data['n_values']
energy_devices = data['energy_devices']
energy_w_line = data['energy_w_line']
energy_b_line = data['energy_b_line']
sum_energy = data['sum_energy']

plt.plot(n_values, energy_devices, 'k-.', label='Мемристивные элементы', linewidth=1.5)           # сплошная
plt.plot(n_values, energy_w_line, 'k-', label='Элементы строк', linewidth=3.5, alpha=0.3)            # пунктирная
plt.plot(n_values, energy_b_line, 'k:', label='Элементы столбцов', linewidth=1.5)             # точки
plt.plot(n_values, sum_energy, 'k-', label='Общая энергия', linewidth=2)
plt.xlabel('Размерность кроссбар-массива (n x n)', fontsize=12)
plt.ylabel('Энергия, нДж', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.show()
