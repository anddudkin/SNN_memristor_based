#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 10
#define BATCH_SIZE 2500
#define EPOCHS 1
#define LEARNING_RATE 0.25
#define NUM_TRAIN 5000   // Количество тренировочных примеров

double weights1[INPUT_SIZE * HIDDEN_SIZE];
double bias1[HIDDEN_SIZE];
double weights2[HIDDEN_SIZE * OUTPUT_SIZE];
double bias2[OUTPUT_SIZE];
double fixed_image[INPUT_SIZE];
int fixed_label = 5;
// Генерация случайных данных вместо MNIST
double train_data[NUM_TRAIN][INPUT_SIZE];
int train_labels[NUM_TRAIN];

double sigmoid(double x) {
    x = fmax(-10, fmin(10, x));
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_deriv(double x) {
    return x * (1.0 - x);
}

void init_weights() {
    srand(42);
    double scale1 = 0.1 / sqrt(INPUT_SIZE);
    double scale2 = 0.1 / sqrt(HIDDEN_SIZE);

    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        weights1[i] = (rand() / (double)RAND_MAX - 0.5) * scale1;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        weights2[i] = (rand() / (double)RAND_MAX - 0.5) * scale2;
    }
    memset(bias1, 0, sizeof(bias1));
    memset(bias2, 0, sizeof(bias2));
}

// Генерация случайных данных
void create_fixed_image() {
    // Простое фиксированное изображение - градиент
    for (int i = 0; i < INPUT_SIZE; i++) {
        fixed_image[i] = 0.5;
    }
}
void setup_training_data() {
    create_fixed_image();

    for (int i = 0; i < NUM_TRAIN; i++) {
        // Копируем одно и то же изображение для всех примеров
        memcpy(train_data[i], fixed_image, INPUT_SIZE * sizeof(double));
        train_labels[i] = fixed_label;  // Все примеры имеют одинаковую метку
    }
}

void forward(double* input, double* hidden, double* output) {
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        double sum = bias1[h];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += input[i] * weights1[i * HIDDEN_SIZE + h];
        }
        hidden[h] = sigmoid(sum);
    }

    for (int o = 0; o < OUTPUT_SIZE; o++) {
        double sum = bias2[o];
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            sum += hidden[h] * weights2[h * OUTPUT_SIZE + o];
        }
        output[o] = sum;
    }
}

double softmax_ce(double* logits, int target, double* probs) {
    double max_logit = logits[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    double sum = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        probs[i] = exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) probs[i] /= sum;

    return -log(probs[target] + 1e-8);
}

void train_batch(int start_idx) {
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];
    double probs[OUTPUT_SIZE];
    double dw1[INPUT_SIZE * HIDDEN_SIZE] = {0};
    double db1[HIDDEN_SIZE] = {0};
    double dw2[HIDDEN_SIZE * OUTPUT_SIZE] = {0};
    double db2[OUTPUT_SIZE] = {0};

    for (int b = 0; b < BATCH_SIZE && (start_idx + b) < NUM_TRAIN; b++) {
        int idx = start_idx + b;
        double* input = train_data[idx];
        int target = train_labels[idx];

        forward(input, hidden, output);
        softmax_ce(output, target, probs);

        // Градиенты выходного слоя
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double grad = (probs[o] - (target == o ? 1.0 : 0.0));
            db2[o] += grad;
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                dw2[h * OUTPUT_SIZE + o] += hidden[h] * grad;
            }
        }

        // Градиенты скрытого слоя
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            double error = 0;
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                error += (probs[o] - (target == o)) * weights2[h * OUTPUT_SIZE + o];
            }
            double grad = error * sigmoid_deriv(hidden[h]);
            db1[h] += grad;
            for (int i = 0; i < INPUT_SIZE; i++) {
                dw1[i * HIDDEN_SIZE + h] += input[i] * grad;
            }
        }
    }

    // Обновляем веса
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        weights1[i] -= dw1[i] * LEARNING_RATE / BATCH_SIZE;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        bias1[i] -= db1[i] * LEARNING_RATE / BATCH_SIZE;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        weights2[i] -= dw2[i] * LEARNING_RATE / BATCH_SIZE;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias2[i] -= db2[i] * LEARNING_RATE / BATCH_SIZE;
    }
}

int main() {
    init_weights();
    setup_training_data();

    double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE], probs[OUTPUT_SIZE];
    //double total_loss = 0.0;

    //printf("Начало обучения (%d эпох)...\n", EPOCHS);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Мини-батчи
        for (int batch = 0; batch < NUM_TRAIN / BATCH_SIZE; batch++) {
            train_batch(batch * BATCH_SIZE);
        }

        // Вычисляем loss на всей выборке
        //total_loss = 0.0;
        //for (int i = 0; i < 1000; i++) { // Для скорости считаем loss на первых 1000 примерах
        //    forward(train_data[i], hidden, output);
        //    double loss = softmax_ce(output, train_labels[i], probs);
        //    total_loss += loss;
        //}

        //printf("Эпоха %d: Средний Loss = %.4f\n", epoch + 1, total_loss / 1000.0);
    }

    //printf("Обучение завершено!\n");
    return 0;
}