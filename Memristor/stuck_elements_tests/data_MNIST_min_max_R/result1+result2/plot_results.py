
import matplotlib.pyplot as plt
import numpy as np
d = []
for i in range(1,51):
    with open(f"result{i}.txt","r") as f:
        f = f.readlines()
        d.append(float(f[11].lstrip("Final result:")))
print(d)
x = range(1,51)
window_size = 5
moving_avg = np.convolve(d, np.ones(window_size)/window_size, mode='same')
moving_avg[0] = 75
moving_avg[1] = 76
moving_avg[-1] = 76
moving_avg[-2] = 76
print(d)
d1 = [76.95, 77.6, 76.0, 76.5, 77.13, 76.95, 75.56, 74.55, 76.4, 75.71, 74.23, 74.0, 73.21, 75.98, 72.55, 74.0, 72.85, 75.4, 72.04, 74.18, 73.17, 74.21, 73.75, 74.19, 72.3, 73.53, 74.4, 72.35, 72.35, 74.8, 73.41, 71.98, 72.12, 71.79, 73.0, 71.18, 72.76, 71.41, 69.05, 70.18, 69.05, 69.84, 70.21, 71.6, 69.6, 70.6, 71.41, 70.97, 69.8, 70.8]
plt.scatter(x, d1, marker="." ,alpha=0.8, label='Исходные точки')
# mean_val = np.mean(d)
# plt.axhline(y=mean_val, color='gray', linestyle='--', linewidth=1,
#             label=f'Среднее: 77.2')
A = np.vstack([x, np.ones(len(x))]).T
a, b = np.linalg.lstsq(A, d1, rcond=None)[0]
x_fit = np.linspace(min(x), max(x), 100)
y_fit = a * x_fit + b
plt.plot(x_fit, y_fit,  "--", color="black", label=f'МНК: y={a:.1f}x+{b:.1f}', linewidth=1)
y_fit = a * x_fit + b
# plt.plot(x, moving_avg, 'black', linewidth=2)
plt.xlabel("Количество залипших элементов, %", fontsize=13)
plt.ylabel("Точность распознавания, %", fontsize=13)
plt.grid(True, alpha=0.3)
plt.ylim(bottom=60, top = 82)
plt.legend(fontsize=12)
plt.savefig("plot_result.png")
plt.show()

