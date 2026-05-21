from matplotlib import pyplot as plt

import badcrossbar
from Network.datasets import encoding_to_spikes
from badcrossbar.compute import compute
import numpy as np
import torch
from tqdm import tqdm
import pickle
def compute_e_crossbar(input_spikes = None, w = None):
    r_i = 1
    t_pulse = 1 * 10 ** -6
    # g = TransformToCrossbarBase(w, R_min=1000, R_max=20000)
    # return g.weights_Om

    r = w
    #print( torch.max(r), torch.min(r),torch.mean(r) ) можно построить график с уменьшением среднего значения весов

    solution = badcrossbar.compute(input_spikes*0.6, r, r_i)
    #print(solution.currents.word_line, len(solution.currents.word_line),len(solution.currents.word_line[0]))
    cur_devices = torch.from_numpy(solution.currents.device)
    cur_w_line = torch.from_numpy(solution.currents.word_line)
    cur_b_line = torch.from_numpy(solution.currents.bit_line)

    energy_devices = torch.sum( cur_devices ** 2 * r * t_pulse) * 10**9
    energy_w_line = torch.sum( cur_w_line ** 2 * r_i * t_pulse) * 10 ** 9
    energy_b_line = torch.sum( cur_b_line ** 2 * r_i * t_pulse) * 10 ** 9
    sum_energy = energy_devices + energy_w_line + energy_b_line
    # print(f"-------{torch.sum(input_spikes)}--------")
    # print(f"{energy_devices} нДж device")
    # print(f"{energy_w_line} нДж word line")
    # print(f"{energy_b_line} нДж bit line")
    # print(f"{sum_energy} нДж sum")
    return energy_devices,energy_w_line,energy_b_line,sum_energy

energy_devices,energy_w_line,energy_b_line,sum_energy =[],[],[],[]
n_values = range(1,784)

for n in tqdm(n_values):
    arr = np.random.uniform(1000, 20000, size=(n, n))
    applied_voltages = np.ones([n, 1])
    g = compute_e_crossbar(applied_voltages,arr)
    energy_devices.append(g[0])
    energy_w_line.append(g[1])
    energy_b_line.append(g[2])
    sum_energy.append(g[3])



# После вычислений сохраняем все данные
np.savez('energy_data.npz',
         n_values=n_values,
         energy_devices=energy_devices,
         energy_w_line=energy_w_line,
         energy_b_line=energy_b_line,
         sum_energy=sum_energy)

print("Данные сохранены в energy_data.npz")


