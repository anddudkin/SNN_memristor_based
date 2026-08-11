
import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt

applied_voltages = np.ones([196, 1])
w =  np.full((196, 50), 10000, dtype=np.int64)
result = np.matmul(np.ones(196), w)
print(result)

r_i = 0

solution0 = badcrossbar.compute(applied_voltages, w, r_i)
print(solution0.currents.output)

r_i = 1

solution1 = badcrossbar.compute(applied_voltages, w, r_i)
print(solution1.currents.output)

r_i_word_line = 1
r_i_bit_line = 1

solution2 = badcrossbar.compute(applied_voltages, w, r_i_word_line=r_i_word_line, r_i_bit_line=r_i_bit_line)

print(solution2.currents.output)

from new_algo import crossbar_distributed


applied_voltages = np.ones([196])
Rrow_seg = 10  # сопротивление участка строки между столбцами
Rcol_seg = 10    # сопротивление участка столбца между строками
Rrow_in = 0.0000000001    # сопротивление на входе строки (от источника к первому столбцу)
Rcol_out = 0.0000000001   # сопротивление на выходе столбца (на землю)
V_S, V_C, I_elem, I_col = crossbar_distributed(w, applied_voltages, Rrow_seg, Rcol_seg,
                                                   Rrow_in, Rcol_out)


print("Токи столбцов:", I_col)