import os
import torch
import h5py
import numpy as np
import badcrossbar

V = [[0.5], [0.5]]
cr = [[15000, 24000], [18000, 22000]]
# V = np.ones([196, 1]) /2
# print(V)
# cr= np.random.randint(1000,4000,(196, 50))
print(cr)
solution = badcrossbar.compute(V, cr, 2)
# print(solution.voltages.word_line)
# print(solution.voltages.bit_line)
# print(torch.add(torch.tensor(solution.voltages.word_line,dtype=torch.float),torch.tensor(solution.voltages.bit_line,dtype=torch.float)))
# print(torch.mul(torch.tensor(solution.currents.device,dtype=torch.float),torch.tensor(cr,dtype=torch.float)))
i_d = torch.tensor(solution.currents.device, dtype=torch.float)
i_bl = torch.tensor(solution.currents.bit_line, dtype=torch.float)
i_wl = torch.tensor(solution.currents.word_line, dtype=torch.float)
r_l = torch.tensor([[2, 2], [2, 2]], dtype=torch.float)
r_d = torch.tensor(cr, dtype=torch.float)

v_d = torch.mul(i_d, r_d)
v_bl = torch.mul(i_bl, r_l)
v_wl = torch.mul(i_wl, r_l)

print(v_d)
print(solution.voltages.bit_line)
print(solution.voltages.word_line)
print(torch.subtract(torch.tensor(solution.voltages.word_line, dtype=torch.float),
                     torch.tensor(solution.voltages.bit_line, dtype=torch.float)))
print(v_wl[0][0] + v_bl[0][0])
