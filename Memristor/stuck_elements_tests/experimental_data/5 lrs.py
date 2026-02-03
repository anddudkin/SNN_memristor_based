import pandas as pd
import matplotlib.pyplot as plt

# Чтение первых 4 столбцов из файла
df = pd.read_excel('5 lrs v.xlsx', usecols=[0, 1])

# Альтернативно, можно указать диапазон
fig = plt.figure(figsize=(8, 5))

# Вывод данных

print(df.iloc[:,0])

plt.scatter(df.iloc[:,0], df.iloc[:,1], marker='o', s = 7, label = "HRS")

plt.xlabel("Номер импульса", fontsize=13)
plt.ylabel("Напряжение, В", fontsize=13)
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0)
# plt.xlim(left = 239)

# Легенда
plt.legend(fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tight_layout()
fig.savefig("5 lrs")
plt.show()