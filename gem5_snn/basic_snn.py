import torch
import torchvision
import numpy as np
import pickle
from tqdm import tqdm

# Параметры
N_INPUT = 28 * 28
N_NEURONS = 100
TIME_STEPS = 20
LR_STDP = 1e-3
TAU_M = 20.0  # мембранная постоянная
TAU_TRACE = 20.0  # постоянная следов STDP
THRESHOLD = 1.0
REST = 0.0

# Инициализация весов и следов
W = torch.randn(N_INPUT, N_NEURONS) * 0.01
W.requires_grad_(False)

trace_pre = torch.zeros(N_INPUT, N_NEURONS)
trace_post = torch.zeros(N_NEURONS)

# Загрузка MNIST
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
train_set = torchvision.datasets.MNIST('../data/MNIST', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)


def lif_update(v, spikes_in, w_col):
    """Обновление мембранного потенциала LIF нейрона"""
    # Входной ток от спайков
    i_in = torch.dot(spikes_in, w_col)
    # Обновление потенциала
    v = v * (1 - 1 / TAU_M) + i_in
    # Генерация спайков
    spike = (v >= THRESHOLD).float()
    v = v * (1 - spike) + REST * spike
    return v, spike


def stdp_update(w, trace_pre_col, trace_post_val, spikes_pre, spike_post):
    """Обновление весов по правилу STDP"""
    # Обновление следов
    trace_pre_col = trace_pre_col * (1 - 1 / TAU_TRACE) + spikes_pre
    trace_post_val = trace_post_val * (1 - 1 / TAU_TRACE) + spike_post

    if spike_post > 0:
        # Пост-синаптический спайк: LTP для активных пре-синапсов
        delta_w = LR_STDP * trace_pre_col
        w += delta_w

    # Ограничение весов
    w = torch.clamp(w, 0, 1)

    return w, trace_pre_col, trace_post_val


# Обучение
print("Обучение...")
for batch_idx, (data, label) in enumerate(tqdm(train_loader, total=100)):
    if batch_idx >= 1000:
        break

    # Преобразование изображения в спайки (интенсивность -> частота)
    img = data.view(-1)
    spikes_input = (torch.rand(N_INPUT) < img).float()

    # Инициализация состояния
    v = torch.zeros(N_NEURONS)

    for t in range(TIME_STEPS):
        # Обновление всех нейронов
        spikes_out = torch.zeros(N_NEURONS)
        for j in range(N_NEURONS):
            v[j], s = lif_update(v[j], spikes_input, W[:, j])
            spikes_out[j] = s

        # Латеральное торможение (ингибирование соседей)
        active = torch.where(spikes_out > 0)[0]
        if len(active) > 0:
            # Выбираем самый активный нейрон
            winner = active[torch.argmax(v[active])]
            spikes_out = torch.zeros(N_NEURONS)
            spikes_out[winner] = 1.0
            v = torch.where(torch.arange(N_NEURONS) == winner, v, torch.tensor(REST))

            # STDP обучение для нейрона-победителя
            for j in range(N_NEURONS):
                if spikes_out[j] > 0:
                    # Обновляем веса для победителя
                    W[:, j], trace_pre[:, j], trace_post[j] = stdp_update(
                        W[:, j], trace_pre[:, j], trace_post[j],
                        spikes_input, spikes_out[j]
                    )

        # Обновление следов для всех нейронов
        trace_post = trace_post * (1 - 1 / TAU_TRACE) + spikes_out
        trace_pre = trace_pre * (1 - 1 / TAU_TRACE) + spikes_input.unsqueeze(1)

# Вывод и сохранение
print(f"\nФинальная матрица весов (первые 10x10):\n{W[:10, :10]}")

with open('weights.pkl', 'wb') as f:
    pickle.dump(W.numpy(), f)
print("\nВеса сохранены в weights.pkl")

# Визуализация обученных рецептивных полей
import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i in range(100):
    row, col = i // 10, i % 10
    weight_map = W[:, i].reshape(28, 28).numpy()
    axes[row, col].imshow(weight_map, cmap='hot', interpolation='nearest')
    axes[row, col].axis('off')
plt.suptitle('Рецептивные поля обученных нейронов')
plt.tight_layout()
plt.savefig('receptive_fields.png', dpi=100)
plt.show()