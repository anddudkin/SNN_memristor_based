import pandas as pd
import matplotlib.pyplot as plt

# Чтение первых 4 столбцов из файла
df = pd.read_excel('1-3 LRS stuck.xlsx', usecols=[0, 1, 2, 3])

# Альтернативно, можно указать диапазон
fig = plt.figure(figsize=(10, 3))

# Вывод данных

print(df.iloc[:,0])

plt.scatter(df.iloc[240:800,0], df.iloc[240:800,1], marker='o', s = 9, label = "HRS")
plt.scatter(df.iloc[240:800,2], df.iloc[240:800,3], marker='x', s = 9, label = "LRS")
plt.ylabel("Resistance, Ohm", fontsize=13)
plt.xlabel("Switching cycle", fontsize=13)
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0)
plt.xlim(left = 239)

# Легенда
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tight_layout()
fig.savefig("1-3 LRS stuck")
plt.show()