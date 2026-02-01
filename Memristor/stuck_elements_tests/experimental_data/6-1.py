import pandas as pd
import matplotlib.pyplot as plt

# Чтение первых 4 столбцов из файла
df = pd.read_excel('6-1.xlsx', usecols=list(range(0,117)))
print(df)
# Альтернативно, можно указать диапазон
fig = plt.figure(figsize=(8, 5))

# Вывод данных

print(df.iloc[2:,0])
for i in range(1,117):
    plt.plot(df.iloc[2:,0], df.iloc[2:,i])

plt.xlabel("Цикл", fontsize=13)
plt.ylabel("Сопротивление, Ом", fontsize=13)
plt.grid(True, alpha=0.3)
# plt.ylim(bottom=-2)
# plt.xlim(left = 239)

# Легенда
# plt.legend()
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tight_layout()
fig.savefig("6-1")
plt.show()