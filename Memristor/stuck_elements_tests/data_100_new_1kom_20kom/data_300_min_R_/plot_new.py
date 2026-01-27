# Загрузка данных
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

# Построение графика
fig = plt.figure(figsize=(6, 6))

# Закрашиваем область между mean+err и mean-err
plt.fill_between(percents,
                 mean_normalized - err_normalized,
                 mean_normalized + err_normalized,
                 alpha=0.1, color='blue', label='mean STD')

# Линия среднего значения
plt.plot(percents, mean_normalized, 'b-', marker ='.', markersize = 8, linewidth=2, label='mean')

#Точки с ошибками (опционально, можно убрать если нужно)
plt.errorbar(percents, mean_normalized, err_normalized,
             fmt='s', markersize=1, capsize=2, linewidth=0.6,
             color='darkblue', alpha=0.4)

# Настройки графика
plt.xlabel("Stuck elements, %", fontsize=20)
plt.ylabel("Deviation, %", fontsize=20)
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0.0001, top = 20)
plt.xlim(left = 0, right = 7)

# Легенда
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
fig.savefig("fig0")
plt.show()