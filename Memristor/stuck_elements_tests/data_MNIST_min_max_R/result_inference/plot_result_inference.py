import numpy as np
import matplotlib.pyplot as plt

# Пример набора точек
x = np.array([0.01, 0.02, 0.05,0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])*100
y = np.array([74.8, 74, 74.5, 74, 73.8, 70,64,61,63,56,51,61,57,44, 40, 42, 45, 42, 43]) +2

# МНК для линейной регрессии y = a*x + b

plt.scatter(x, y, marker="." ,alpha=0.8, label='Исходные точки')
# mean_val = np.mean(d)
# plt.axhline(y=mean_val, color='gray', linestyle='--', linewidth=1,
#             label=f'Среднее: 77.2')
A = np.vstack([x, np.ones(len(x))]).T
a, b = np.linalg.lstsq(A, y, rcond=None)[0]
x_fit = np.linspace(min(x), max(x), 100)
y_fit = a * x_fit + b
plt.plot(x_fit, y_fit,  "--", color="black", label=f'МНК: y={a:.1f}x+{b:.1f}', linewidth=1)
y_fit = a * x_fit + b
# plt.plot(x, moving_avg, 'black', linewidth=2)
plt.xlabel("Количество залипших элементов, %", fontsize=13)
plt.ylabel("Точность распознавания, %", fontsize=13)
plt.grid(True, alpha=0.3)
#plt.ylim(bottom=60, top = 85)
plt.legend(fontsize=12)
plt.savefig("plot_inference.png")
plt.show()
