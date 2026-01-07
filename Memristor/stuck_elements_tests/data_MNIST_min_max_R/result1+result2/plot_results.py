
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
plt.scatter(x, d, marker="." ,alpha=0.8, label='Исходные точки')
mean_val = np.mean(d)
plt.axhline(y=mean_val, color='gray', linestyle='--', linewidth=1,
            label=f'Среднее: 77.2')

plt.plot(x, moving_avg, 'black', linewidth=2)
plt.legend()
plt.show()

