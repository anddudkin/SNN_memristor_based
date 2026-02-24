import numpy as np
import matplotlib.pyplot as plt

sol_mean_all = np.load('data_mean.npy')
err_all = np.load('data_std.npy')
percents = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]

sol_mean_all = np.insert(sol_mean_all,0,0)
err_all = np.insert(err_all,0,0)
# Нормализация данных (деление на 40)
mean_normalized = sol_mean_all / 40
err_normalized = err_all / 40

# Построение графика в оттенках серого
fig = plt.figure(figsize=(6, 6))

# Закрашиваем область между mean+err и mean-err (светло-серый)
plt.fill_between(percents,
                 mean_normalized - err_normalized,
                 mean_normalized + err_normalized,
                 alpha=0.4, color='#CCCCCC', label='mean STD')  # светло-серый

# Линия среднего значения (темно-серый/черный)
plt.plot(percents, mean_normalized, '-', marker='.', markersize=8,
         linewidth=2, label='mean', color='#333333')  # темно-серый

# Точки с ошибками (средне-серый)
plt.errorbar(percents, mean_normalized, err_normalized,
             fmt='s', markersize=1, capsize=2, linewidth=0.6,
             color='#666666', alpha=0.4)  # средне-серый

# Настройки графика
# Настройки графика
plt.xlabel("Кол-во залипших элементов, %", fontsize=20)
plt.ylabel(r'$\overline{\Delta \mathtt{I}}$, %', fontsize=20)
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0.0001, top = 20)
plt.xlim(left = 0, right = 7)

# Легенда

plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
fig.savefig("fig0_grayscale", dpi=300)

