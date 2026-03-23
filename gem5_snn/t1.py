import torch
import torchvision
import numpy as np
import pickle
from tqdm import tqdm

# Параметры
n_train = 30
N_INPUT = 28 * 28
N_NEURONS = 100
TIME_STEPS = 25
LR_STDP = 0.003
TAU_M = 15  # мембранная постоянная
TAU_TRACE = 20.0  # постоянная следов STDP
THRESHOLD = 5
REST = 0.0
REFRACTORY_PERIOD = 10  # длительность рефрактерного периода в тиках

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
train_set = torchvision.datasets.MNIST('../data/MNIST', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)


def lif_update(v, refractory_counter, spikes_in, w_col):
    """Обновление мембранного потенциала LIF нейрона с рефрактерным периодом"""
    if refractory_counter > 0:
        # Нейрон в рефрактерном периоде
        return v, 0.0, refractory_counter - 1

    # Входной ток от спайков
    i_in = torch.dot(spikes_in, w_col)
    # Обновление потенциала
    v = v * (1 - 1 / TAU_M) + i_in

    # Генерация спайков
    if v >= THRESHOLD:
        spike = 1.0
        v = REST
        refractory_counter = REFRACTORY_PERIOD
    else:
        spike = 0.0

    return v, spike, refractory_counter


def stdp_update(w, trace_pre_col, trace_post_val, spikes_pre, spike_post, refractory_active):
    """Обновление весов по правилу STDP"""
    # Обновление следов
    trace_pre_col = trace_pre_col * (1 - 1 / TAU_TRACE) + spikes_pre
    trace_post_val = trace_post_val * (1 - 1 / TAU_TRACE) + spike_post

    # STDP только если нейрон не в рефрактерном периоде
    if spike_post > 0 and not refractory_active:
        # Пост-синаптический спайк: LTP для активных пре-синапсов
        delta_w = LR_STDP * trace_pre_col
        w += delta_w

    # Ограничение весов
    w = torch.clamp(w, 0, 1)

    return w, trace_pre_col, trace_post_val


# Обучение
print("Обучение...")
for batch_idx, (data, label) in enumerate(tqdm(train_loader, total= n_train)):
    if batch_idx >= n_train:
        break

    # Преобразование изображения в спайки (интенсивность -> частота)
    img = data.view(-1)
    spikes_input = (torch.rand(N_INPUT) < img).float()

    # Инициализация состояния
    v = torch.zeros(N_NEURONS)
    refractory_counters = torch.zeros(N_NEURONS, dtype=torch.long)
    spike_history = []

    for t in range(TIME_STEPS):
        # Обновление всех нейронов
        spikes_out = torch.zeros(N_NEURONS)
        new_refractory = torch.zeros(N_NEURONS, dtype=torch.long)

        for j in range(N_NEURONS):
            v[j], s, new_refractory[j] = lif_update(
                v[j], refractory_counters[j], spikes_input, W[:, j]
            )
            spikes_out[j] = s

        refractory_counters = new_refractory

        # Латеральное торможение (ингибирование соседей)
        active = torch.where((spikes_out > 0) & (refractory_counters == 0))[0]

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
                    v[j] = max(v[j] - 0.5, REST)  # дополнительное торможение

        spike_history.append(spikes_out)

        # STDP обучение для активных нейронов
        for j in range(N_NEURONS):
            if spikes_out[j] > 0:
                # Проверяем, не в рефрактерном ли периоде был нейрон до спайка
                refractory_before = (refractory_counters[j] == REFRACTORY_PERIOD - 1)

                W[:, j], trace_pre[:, j], trace_post[j] = stdp_update(
                    W[:, j], trace_pre[:, j], trace_post[j],
                    spikes_input, spikes_out[j], refractory_before
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