import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from scipy.ndimage import zoom
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Проверка наличия CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# Загрузка и предобработка данных MNIST через torchvision
print("Загрузка данных MNIST...")

# Загрузка тренировочных данных
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
)

# Загрузка тестовых данных
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
)

# Преобразование в numpy массивы
x_train_full = train_dataset.data.numpy()
y_train_full = train_dataset.targets.numpy()
x_test_full = test_dataset.data.numpy()
y_test_full = test_dataset.targets.numpy()

# Нормализация
x_train_full = x_train_full.astype(np.float32)
x_test_full = x_test_full.astype(np.float32)


# Преобразование в спайковые сигналы
def convert_to_spikes(images, time_steps=20, method='rate'):
    """
    Преобразование изображений в спайковые последовательности
    method: 'rate' - кодирование интенсивностью, 'latency' - временное кодирование
    """
    n_samples = len(images)
    height, width = images[0].shape
    spikes = np.zeros((n_samples, time_steps, height, width), dtype=np.int8)

    if method == 'rate':
        # Кодирование интенсивностью (rate coding)
        for i in range(n_samples):
            img = images[i] / 255.0  # нормализация в [0, 1]
            for t in range(time_steps):
                # Вероятность спайка пропорциональна интенсивности пикселя
                spike_prob = img
                spikes[i, t] = np.random.random((height, width)) < spike_prob
    elif method == 'latency':
        # Временное кодирование (latency coding)
        for i in range(n_samples):
            img = images[i] / 255.0
            for h in range(height):
                for w in range(width):
                    if img[h, w] > 0.1:  # только значимые пиксели
                        # Чем ярче пиксель, тем раньше спайк
                        latency = int((1 - img[h, w]) * (time_steps - 1))
                        if latency < time_steps:
                            spikes[i, latency, h, w] = 1

    return spikes


# Уменьшаем размерность для ускорения обучения
def downsample_images(images, size=(14, 14)):
    """Уменьшение разрешения изображений с использованием интерполяции"""
    n_samples = len(images)
    h_ratio = size[0] / images.shape[1]
    w_ratio = size[1] / images.shape[2]

    downsampled = np.zeros((n_samples, size[0], size[1]))
    for i in range(n_samples):
        downsampled[i] = zoom(images[i], (h_ratio, w_ratio), order=1)

    return downsampled


# Подготовка данных
print("Подготовка данных...")
x_train_down = downsample_images(x_train_full, (14, 14))
x_test_down = downsample_images(x_test_full, (14, 14))

# Используем часть данных для ускорения обучения
n_train_samples = 2000
n_test_samples = 200

x_train_small = x_train_down[:n_train_samples]
y_train_small = y_train_full[:n_train_samples]
x_test_small = x_test_down[:n_test_samples]
y_test_small = y_test_full[:n_test_samples]

# Преобразование в спайки
time_steps = 15
encoding_method = 'rate'  # 'rate' или 'latency'
print(f"Преобразование в спайковые последовательности ({time_steps} временных шагов, метод: {encoding_method})...")
x_train_spikes = convert_to_spikes(x_train_small, time_steps, encoding_method)
x_test_spikes = convert_to_spikes(x_test_small, time_steps, encoding_method)

print(f"Форма тренировочных данных: {x_train_spikes.shape}")
print(f"Форма тестовых данных: {x_test_spikes.shape}")


class LIFNeuron:
    """Нейрон с моделью Leaky Integrate-and-Fire"""

    def __init__(self, threshold=1.0, decay=0.9, reset_potential=0.0, refractory_period=0):
        self.threshold = threshold
        self.decay = decay
        self.reset_potential = reset_potential
        self.refractory_period = refractory_period
        self.potential = 0.0
        self.spike_count = 0
        self.refractory_counter = 0

    def update(self, input_current):
        """
        Обновление состояния нейрона
        Возвращает: спайк (0 или 1)
        """
        # Рефрактерный период
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            return 0

        # Утечка потенциала
        self.potential *= self.decay

        # Добавление входного тока
        self.potential += input_current

        # Проверка порога
        if self.potential >= self.threshold:
            self.potential = self.reset_potential
            self.spike_count += 1
            self.refractory_counter = self.refractory_period
            return 1
        return 0

    def reset(self):
        """Сброс состояния нейрона"""
        self.potential = 0.0
        self.spike_count = 0
        self.refractory_counter = 0


class SNNLayer:
    """Слой импульсной нейронной сети с обучаемыми весами"""

    def __init__(self, n_neurons, input_size, threshold=1.0, decay=0.9,
                 learning_rate=0.01, use_stdp=True):
        self.n_neurons = n_neurons
        self.input_size = input_size
        self.threshold = threshold
        self.decay = decay
        self.learning_rate = learning_rate
        self.use_stdp = use_stdp

        # Инициализация весов (методом Ксавье)
        self.weights = np.random.randn(n_neurons, input_size) * np.sqrt(2.0 / input_size)
        self.weights = np.clip(self.weights, 0, 1)  # только возбуждающие связи

        # Инициализация нейронов
        self.neurons = [LIFNeuron(threshold, decay, refractory_period=2)
                        for _ in range(n_neurons)]

        # Для STDP
        self.last_spike_time = np.zeros((n_neurons, input_size))

    def forward(self, input_spikes, time_step=None):
        """
        Прямой проход
        input_spikes: бинарная матрица входных спайков
        Возвращает: выходные спайки слоя
        """
        output_spikes = np.zeros(self.n_neurons)

        for i, neuron in enumerate(self.neurons):
            # Вычисление входного тока как взвешенной суммы спайков
            input_current = np.sum(self.weights[i] * input_spikes)
            output_spikes[i] = neuron.update(input_current)

            # STDP обучение
            if self.use_stdp and time_step is not None:
                self._update_stdp(i, input_spikes, output_spikes[i], time_step)

        return output_spikes

    def _update_stdp(self, neuron_idx, input_spikes, output_spike, time_step):
        """Обновление весов по правилу STDP"""
        if output_spike == 1:
            # Пост-синаптический спайк: усиление недавно активных синапсов
            for j in range(self.input_size):
                if input_spikes[j] == 1:
                    # Потенцирование (LTP)
                    self.weights[neuron_idx, j] += self.learning_rate
                else:
                    # Небольшое депрессирование для неактивных синапсов
                    self.weights[neuron_idx, j] -= self.learning_rate * 0.1
        else:
            # Пре-синаптический спайк без пост-синаптического: депрессирование
            for j in range(self.input_size):
                if input_spikes[j] == 1:
                    self.weights[neuron_idx, j] -= self.learning_rate * 0.5

        # Клиппинг весов
        self.weights[neuron_idx] = np.clip(self.weights[neuron_idx], 0, 1)

    def reset(self):
        """Сброс состояния всех нейронов"""
        for neuron in self.neurons:
            neuron.reset()


class SNNWithInhibition:
    """Импульсная нейронная сеть с латеральным ингибированием"""

    def __init__(self, input_size, n_classes, time_steps,
                 threshold=1.0, decay=0.95, inhibition_strength=0.5,
                 inhibition_type='winner_takes_all'):
        self.input_size = input_size
        self.n_classes = n_classes
        self.time_steps = time_steps
        self.threshold = threshold
        self.decay = decay
        self.inhibition_strength = inhibition_strength
        self.inhibition_type = inhibition_type  # 'winner_takes_all' или 'lateral'

        # Создание выходного слоя
        self.output_layer = SNNLayer(n_classes, input_size, threshold, decay)

        # Матрица латерального ингибирования
        self.inhibition_matrix = np.ones((n_classes, n_classes)) * inhibition_strength
        np.fill_diagonal(self.inhibition_matrix, 0)

        # История активности для анализа
        self.activity_history = []

    def forward(self, input_spikes, return_activity=False):
        """
        Прямой проход через всю сеть
        input_spikes: матрица спайков [time_steps, input_size]
        """
        spike_counts = np.zeros(self.n_classes)
        membrane_potentials = []

        for t in range(self.time_steps):
            # Входные спайки на текущем временном шаге
            current_input = input_spikes[t]

            # Вычисление выходных спайков
            output_spikes = self.output_layer.forward(current_input, time_step=t)

            # Применение ингибирования
            if self.inhibition_type == 'winner_takes_all':
                output_spikes = self._winner_takes_all(output_spikes)
            elif self.inhibition_type == 'lateral':
                output_spikes = self._lateral_inhibition(output_spikes, t)

            # Накопление количества спайков
            spike_counts += output_spikes

            # Запись потенциалов для визуализации
            if return_activity:
                potentials = [neuron.potential for neuron in self.output_layer.neurons]
                membrane_potentials.append(potentials)

        if return_activity:
            return spike_counts, np.array(membrane_potentials)
        return spike_counts

    def _winner_takes_all(self, output_spikes):
        """Механизм 'победитель получает всё'"""
        if np.sum(output_spikes) > 0:
            winner_idx = np.argmax([neuron.spike_count for neuron in self.output_layer.neurons])
            # Оставляем только спайк победителя
            new_output = np.zeros_like(output_spikes)
            new_output[winner_idx] = output_spikes[winner_idx]
            return new_output
        return output_spikes

    def _lateral_inhibition(self, output_spikes, time_step):
        """Латеральное ингибирование с учетом времени"""
        if np.sum(output_spikes) > 0:
            # Находим нейроны с наибольшей активностью
            activities = np.array([neuron.spike_count for neuron in self.output_layer.neurons])
            if np.max(activities) > 0:
                # Ингибируем все нейроны, кроме наиболее активного
                for i in range(self.n_classes):
                    if activities[i] < np.max(activities) and output_spikes[i] == 0:
                        # Ингибирование через снижение потенциала
                        inhibition = self.inhibition_strength * (1 - activities[i] / np.max(activities))
                        self.output_layer.neurons[i].potential *= (1 - inhibition)
        return output_spikes

    def train(self, x_train, y_train, epochs=5, learning_rate=0.01, verbose=True):
        """
        Обучение сети с использованием supervised STDP
        """
        print("Начало обучения...")
        training_history = []

        for epoch in range(epochs):
            if verbose:
                print(f"\n{'=' * 50}")
                print(f"Эпоха {epoch + 1}/{epochs}")
                print(f"{'=' * 50}")

            correct_predictions = 0
            epoch_losses = []

            for i in range(len(x_train)):
                # Получаем спайки для текущего образца
                input_spikes = x_train[i]
                input_flat = input_spikes.reshape(self.time_steps, -1)
                target = y_train[i]

                # Прямой проход
                spike_counts = self.forward(input_flat)

                # Определение предсказанного класса
                predicted = np.argmax(spike_counts)

                # Обновление весов
                if predicted == target:
                    correct_predictions += 1
                    loss = 0
                    # Подкрепление правильного ответа
                    for j in range(self.input_size):
                        avg_input_activity = np.mean(input_flat[:, j])
                        self.output_layer.weights[target, j] += learning_rate * avg_input_activity
                else:
                    loss = 1
                    # Коррекция: усиление правильного класса, ослабление неправильного
                    for j in range(self.input_size):
                        avg_input_activity = np.mean(input_flat[:, j])
                        self.output_layer.weights[predicted, j] -= learning_rate * avg_input_activity * 0.5
                        self.output_layer.weights[target, j] += learning_rate * avg_input_activity

                epoch_losses.append(loss)

                # Нормализация весов
                self.output_layer.weights = np.clip(self.output_layer.weights, 0, 1)

                # Сброс состояния сети для следующего образца
                self.reset()

                # Вывод прогресса
                if verbose and (i + 1) % 1000 == 0:
                    accuracy = correct_predictions / (i + 1) * 100
                    print(f"  Обработано {i + 1:4d}/{len(x_train)} образцов, "
                          f"точность: {accuracy:.2f}%, "
                          f"потери: {np.mean(epoch_losses):.4f}")

            epoch_accuracy = correct_predictions / len(x_train) * 100
            training_history.append({
                'epoch': epoch + 1,
                'accuracy': epoch_accuracy,
                'loss': np.mean(epoch_losses)
            })

            if verbose:
                print(f"\nРезультаты эпохи {epoch + 1}:")
                print(f"  Точность: {epoch_accuracy:.2f}%")
                print(f"  Потери: {np.mean(epoch_losses):.4f}")

        return training_history

    def predict(self, x_test, return_confidence=False):
        """
        Предсказание для тестовых данных с возможностью возврата уверенности
        """
        predictions = []
        confidences = []

        for i in range(len(x_test)):
            input_spikes = x_test[i]
            input_flat = input_spikes.reshape(self.time_steps, -1)

            spike_counts = self.forward(input_flat)
            predicted = np.argmax(spike_counts)
            predictions.append(predicted)

            if return_confidence:
                # Нормализованная уверенность на основе количества спайков
                if np.sum(spike_counts) > 0:
                    confidence = spike_counts[predicted] / np.sum(spike_counts)
                else:
                    confidence = 0
                confidences.append(confidence)

            self.reset()

        if return_confidence:
            return np.array(predictions), np.array(confidences)
        return np.array(predictions)

    def reset(self):
        """Сброс состояния всей сети"""
        self.output_layer.reset()
        self.activity_history = []


def calculate_metrics(y_true, y_pred):
    """Вычисление метрик качества"""
    # Точность
    accuracy = np.mean(y_true == y_pred) * 100

    # Матрица ошибок
    n_classes = len(np.unique(y_true))
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1

    # Precision, Recall, F1 для каждого класса
    precision = []
    recall = []
    f1 = []

    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)

    # Macro average
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }


def visualize_results(snn, x_test_spikes, y_test_small, predictions, metrics):
    """Визуализация результатов обучения"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Матрица ошибок
    im = axes[0, 0].imshow(metrics['confusion_matrix'], cmap='Blues', interpolation='nearest')
    axes[0, 0].set_title('Матрица ошибок')
    axes[0, 0].set_xlabel('Предсказанный класс')
    axes[0, 0].set_ylabel('Истинный класс')
    plt.colorbar(im, ax=axes[0, 0])

    # Добавляем числа в ячейки
    for i in range(metrics['confusion_matrix'].shape[0]):
        for j in range(metrics['confusion_matrix'].shape[1]):
            axes[0, 0].text(j, i, metrics['confusion_matrix'][i, j],
                            ha="center", va="center", color="white" if metrics['confusion_matrix'][i, j] > metrics[
                    'confusion_matrix'].max() / 2 else "black")

    # 2. Примеры изображений и их спайковой активности
    sample_idx = 0
    sample_image = x_test_small[sample_idx]
    sample_spikes = x_test_spikes[sample_idx]

    axes[0, 1].imshow(sample_image, cmap='gray')
    axes[0, 1].set_title(f'Исходное изображение\nИстинная метка: {y_test_small[sample_idx]}')
    axes[0, 1].axis('off')

    # Визуализация спайковой активности
    spike_raster = sample_spikes.reshape(snn.time_steps, -1)
    axes[0, 2].imshow(spike_raster.T, aspect='auto', cmap='binary',
                      interpolation='nearest', extent=[0, snn.time_steps, 0, spike_raster.shape[1]])
    axes[0, 2].set_title('Спайковая активность\n(время × нейроны)')
    axes[0, 2].set_xlabel('Время')
    axes[0, 2].set_ylabel('Входные нейроны')

    # 3. Распределение точности по классам
    class_accuracy = []
    for i in range(10):
        mask = y_test_small == i
        if np.sum(mask) > 0:
            acc = np.sum(predictions[mask] == i) / np.sum(mask) * 100
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)

    bars = axes[1, 0].bar(range(10), class_accuracy, color='skyblue', edgecolor='navy')
    for bar, acc in zip(bars, class_accuracy):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    axes[1, 0].set_title('Точность по классам')
    axes[1, 0].set_xlabel('Класс')
    axes[1, 0].set_ylabel('Точность (%)')
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Веса обученной сети
    weights_vis = snn.output_layer.weights.reshape(snn.n_classes, 14, 14)
    axes[1, 1].imshow(weights_vis[0], cmap='hot')
    axes[1, 1].set_title('Веса для класса 0 (наиболее активные входы)')
    axes[1, 1].axis('off')

    # 5. Примеры неправильных предсказаний
    wrong_indices = np.where(predictions != y_test_small)[0]
    if len(wrong_indices) > 0:
        wrong_idx = wrong_indices[0]
        wrong_image = x_test_small[wrong_idx]
        axes[1, 2].imshow(wrong_image, cmap='gray')
        axes[1, 2].set_title(f'Пример ошибки\n{y_test_small[wrong_idx]} → {predictions[wrong_idx]}')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].text(0.5, 0.5, 'Нет ошибок!', ha='center', va='center', fontsize=14)
        axes[1, 2].set_title('Примеры ошибок')
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # Дополнительная визуализация активности нейронов
    fig, ax = plt.subplots(figsize=(10, 6))

    # Тестируем на одном примере и показываем активность нейронов
    test_sample = x_test_spikes[0]
    input_flat = test_sample.reshape(snn.time_steps, -1)

    snn.reset()
    _, membrane_potentials = snn.forward(input_flat, return_activity=True)

    im = ax.imshow(membrane_potentials.T, aspect='auto', cmap='viridis',
                   interpolation='bilinear', extent=[0, snn.time_steps, 0, snn.n_classes])
    ax.set_xlabel('Время')
    ax.set_ylabel('Выходные нейроны (классы)')
    ax.set_title('Динамика потенциалов мембраны во времени\n(чем ярче, тем выше потенциал)')
    plt.colorbar(im, ax=ax, label='Потенциал мембраны')

    # Добавляем контуры для наглядности
    for i in range(snn.n_classes):
        ax.axhline(y=i + 0.5, color='white', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


# Создание и обучение сети
print("\n" + "=" * 60)
print("Создание импульсной нейронной сети")
print("=" * 60)

input_size = 14 * 14  # размер входного слоя (14x14 пикселей)
n_classes = 10
time_steps = 15

snn = SNNWithInhibition(
    input_size=input_size,
    n_classes=n_classes,
    time_steps=time_steps,
    threshold=0.8,
    decay=0.95,
    inhibition_strength=0.4,
    inhibition_type='winner_takes_all'  # 'winner_takes_all' или 'lateral'
)

print(f"Параметры сети:")
print(f"  Входной размер: {input_size} нейронов")
print(f"  Выходной размер: {n_classes} нейронов")
print(f"  Временных шагов: {time_steps}")
print(f"  Порог срабатывания: {snn.threshold}")
print(f"  Коэффициент затухания: {snn.decay}")
print(f"  Сила ингибирования: {snn.inhibition_strength}")
print(f"  Тип ингибирования: {snn.inhibition_type}")

# Обучение сети
print("\n" + "=" * 60)
print("Обучение сети")
print("=" * 60)

training_history = snn.train(
    x_train_spikes,
    y_train_small,
    epochs=3,
    learning_rate=0.008,
    verbose=True
)

# Тестирование
print("\n" + "=" * 60)
print("Тестирование сети")
print("=" * 60)

predictions, confidences = snn.predict(x_test_spikes, return_confidence=True)

# Оценка качества
metrics = calculate_metrics(y_test_small, predictions)

print(f"\nРезультаты тестирования:")
print(f"  Точность: {metrics['accuracy']:.2f}%")
print(f"  Macro Precision: {metrics['macro_precision']:.3f}")
print(f"  Macro Recall: {metrics['macro_recall']:.3f}")
print(f"  Macro F1-Score: {metrics['macro_f1']:.3f}")

print(f"\nТочность по классам:")
for i in range(10):
    print(f"  Класс {i}: {metrics['precision'][i] * 100:.2f}% (Precision: {metrics['precision'][i]:.3f}, "
          f"Recall: {metrics['recall'][i]:.3f}, F1: {metrics['f1'][i]:.3f})")

# Визуализация результатов
visualize_results(snn, x_test_spikes, y_test_small, predictions, metrics)

# Вывод статистики обучения
print("\n" + "=" * 60)
print("Статистика обучения")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

epochs = [h['epoch'] for h in training_history]
accuracies = [h['accuracy'] for h in training_history]
losses = [h['loss'] for h in training_history]

ax1.plot(epochs, accuracies, 'b-o', linewidth=2, markersize=8)
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Точность (%)')
ax1.set_title('Динамика точности обучения')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(epochs)

ax2.plot(epochs, losses, 'r-s', linewidth=2, markersize=8)
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Потери')
ax2.set_title('Динамика потерь обучения')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs)

plt.tight_layout()
plt.show()
with open('snn_weights.pkl', 'wb') as f:
    pickle.dump(snn.output_layer.weights, f)
print("Веса сохранены в snn_weights.pkl")

print("\nОбучение завершено!")
print(f"Финальная точность: {metrics['accuracy']:.2f}%")
print(f"Средняя уверенность предсказаний: {np.mean(confidences):.3f}")

# Дополнительная информация
print("\n" + "=" * 60)
print("Информация о датасете")
print("=" * 60)
print(f"Размер тренировочной выборки: {len(x_train_spikes)}")
print(f"Размер тестовой выборки: {len(x_test_spikes)}")
print(f"Количество классов: 10 (цифры 0-9)")
print(f"Размер изображений после downsampling: 14x14 пикселей")
print(f"Метод кодирования: {encoding_method}")
print(f"Временных шагов: {time_steps}")