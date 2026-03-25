import torch
import torchvision
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from Network.datasets import encoding_to_spikes

with open("weights.pkl", 'rb') as f:
    W = torch.load(f)

W.requires_grad_(False)
W = [[],[]]
# Параметры
n_train = 50
N_INPUT = 28 * 28
N_NEURONS = 100
TIME_STEPS = 25
LR_STDP = 0.005
TAU_M = 15  # мембранная постоянная
TAU_TRACE = 20.0  # постоянная следов STDP
THRESHOLD = 2
REST = 0.0
REFRACTORY_PERIOD = 15  # длительность рефрактерного периода в тиках
inh_coef = 0.9
# Инициализация весов и следов


# Загрузка MNIST
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
train_set = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)


def lif_update(v, refractory_counter, spikes_in, w_col):
    """Обновление мембранного потенциала LIF нейрона с рефрактерным периодом"""
    if refractory_counter > 0:
        # Нейрон в рефрактерном периоде
        return v, 0.0, refractory_counter - 1
    i_in = torch.dot(spikes_in, w_col)
    v = v * (1 - 1 / TAU_M) + i_in
    if v >= THRESHOLD:
        spike = 1.0
        v = REST
        refractory_counter = REFRACTORY_PERIOD
    else:
        spike = 0.0
    return v, spike, refractory_counter


for batch_idx, (data, label) in enumerate(tqdm(train_loader, total= n_train)):

    if batch_idx >= n_train:
        break


    img = data.view(-1)
    input_spikes = encoding_to_spikes(img, TIME_STEPS)

    v = torch.zeros(N_NEURONS)
    refractory_counters = torch.zeros(N_NEURONS, dtype=torch.long)
    spike_history = []

    for i, t in enumerate(range(TIME_STEPS)):
        # Обновление всех нейронов
        spikes_out = torch.zeros(N_NEURONS)

        new_refractory = torch.zeros(N_NEURONS, dtype=torch.long)

        for j in range(N_NEURONS):

            #spikes_input = (torch.rand(N_INPUT) < img).float()
            spikes_input = input_spikes[i].squeeze()

            v[j], s, new_refractory[j] = lif_update(
                v[j], refractory_counters[j], spikes_input, W[:, j]
            )
            spikes_out[j] = s

        refractory_counters = new_refractory

        # Латеральное торможение (ингибирование соседей)
        active = torch.where((spikes_out > 0))[0]

        if len(active) > 0:

            # Выбираем самый активный нейрон среди не-рефрактерных
            winner = active[torch.argmax(v[active])]

            # Подавляем все остальные спайки
            spikes_out = torch.zeros(N_NEURONS)
            spikes_out[winner] = 1.0

            # Все остальные нейроны получают дополнительное торможение
            # (опционально, можно добавить негативный импульс)
            for j in range(N_NEURONS):
                if j != winner and refractory_counters[j] == 0:
                    v[j] = max(v[j] * inh_coef, REST)  # дополнительное торможение

        spike_history.append(spikes_out)

# Вывод и сохранение
print(f"\nФинальная матрица весов (первые 10x10):\n{W[:10, :10]}")

with open('weights.pkl', 'wb') as f:
    pickle.dump(W.numpy(), f)
print("\nВеса сохранены в weights.pkl")

# Визуализация обученных рецептивных полей
import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i in range(min(100, N_NEURONS)):
    row, col = i // 10, i % 10
    weight_map = W[:, i].reshape(28, 28).numpy()
    axes[row, col].imshow(weight_map, cmap='hot', interpolation='nearest')
    axes[row, col].axis('off')
plt.suptitle('Рецептивные поля обученных нейронов')
plt.tight_layout()
plt.savefig('receptive_fields.png', dpi=100)
plt.show()

# Дополнительная статистика
print(f"\nСтатистика:")
print(f"Среднее значение весов: {W.mean():.4f}")
print(f"Количество нулевых весов: {(W == 0).sum().item()}")
print(f"Количество максимальных весов: {(W == 1).sum().item()}")