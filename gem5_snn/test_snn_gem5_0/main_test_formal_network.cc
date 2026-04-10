#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define NUM_TRAIN 1000
#define NUM_TEST 200
#define EPOCHS 50
#define LEARNING_RATE 0.1

double weights1[INPUT_SIZE * HIDDEN_SIZE];
double bias1[HIDDEN_SIZE];
double weights2[HIDDEN_SIZE * OUTPUT_SIZE];
double bias2[OUTPUT_SIZE];

double sigmoid(double x) { return 1.0 / (1.0 + exp(-fmax(-500, fmin(500, x)))); }
double sigmoid_deriv(double x) { return x * (1.0 - x); }

void init_weights() {
    srand(42);
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) weights1[i] = (rand() / (double)RAND_MAX - 0.5) * 0.2;
    for (int i = 0; i < HIDDEN_SIZE; i++) bias1[i] = 0.0;
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) weights2[i] = (rand() / (double)RAND_MAX - 0.5) * 0.2;
    for (int i = 0; i < OUTPUT_SIZE; i++) bias2[i] = 0.0;
}

void generate_data(double train_in[NUM_TRAIN][INPUT_SIZE], int train_labels[NUM_TRAIN], double test_in[NUM_TEST][INPUT_SIZE], int test_labels[NUM_TEST]) {
    for (int i = 0; i < NUM_TRAIN; i++) {
        train_labels[i] = rand() % 10;
        for (int j = 0; j < INPUT_SIZE; j++) {
            train_in[i][j] = (rand() / (double)RAND_MAX > 0.5) ? 1.0 : 0.0;  // Бинарное изображение
        }
    }
    for (int i = 0; i < NUM_TEST; i++) {
        test_labels[i] = rand() % 10;
        for (int j = 0; j < INPUT_SIZE; j++) {
            test_in[i][j] = (rand() / (double)RAND_MAX > 0.5) ? 1.0 : 0.0;
        }
    }
}

void forward(double *input, double *hidden, double *output) {
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        double sum = bias1[h];
        for (int i = 0; i < INPUT_SIZE; i++) sum += input[i] * weights1[i * HIDDEN_SIZE + h];
        hidden[h] = sigmoid(sum);
    }
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        double sum = bias2[o];
        for (int h = 0; h < HIDDEN_SIZE; h++) sum += hidden[h] * weights2[h * OUTPUT_SIZE + o];
        output[o] = sigmoid(sum);
    }
}

double train_loss = 0.0;
void backward(double *input, double *hidden, double *output, double *target, double *d_weights1, double *d_bias1, double *d_weights2, double *d_bias2) {
    double output_error[OUTPUT_SIZE];
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        output_error[o] = output[o] - target[o];
        train_loss += output_error[o] * output_error[o];
    }
    train_loss /= OUTPUT_SIZE;

    double hidden_error[HIDDEN_SIZE];
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        double err = 0.0;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            err += output_error[o] * weights2[h * OUTPUT_SIZE + o];
        }
        hidden_error[h] = err * sigmoid_deriv(hidden[h]);
    }

    for (int h = 0; h < HIDDEN_SIZE; h++) {
        d_bias1[h] += hidden_error[h] * LEARNING_RATE;
        for (int i = 0; i < INPUT_SIZE; i++) {
            d_weights1[i * HIDDEN_SIZE + h] += input[i] * hidden_error[h] * LEARNING_RATE;
        }
    }
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        d_bias2[o] += output_error[o] * LEARNING_RATE;
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            d_weights2[h * OUTPUT_SIZE + o] += hidden[h] * output_error[o] * LEARNING_RATE;
        }
    }
}

int predict(double *input, double *hidden, double *output) {
    forward(input, hidden, output);
    int pred = 0;
    double maxp = output[0];
    for (int o = 1; o < OUTPUT_SIZE; o++) {
        if (output[o] > maxp) {
            maxp = output[o];
            pred = o;
        }
    }
    return pred;
}

void update_weights(double *d_weights1, double *d_bias1, double *d_weights2, double *d_bias2) {
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        weights1[i] -= d_weights1[i];
        d_weights1[i] = 0.0;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        bias1[i] -= d_bias1[i];
        d_bias1[i] = 0.0;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        weights2[i] -= d_weights2[i];
        d_weights2[i] = 0.0;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias2[i] -= d_bias2[i];
        d_bias2[i] = 0.0;
    }
}

int main() {
    init_weights();

    double train_in[NUM_TRAIN][INPUT_SIZE];
    int train_labels[NUM_TRAIN];
    double test_in[NUM_TEST][INPUT_SIZE];
    int test_labels[NUM_TEST];
    generate_data(train_in, train_labels, test_in, test_labels);

    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];
    double target[OUTPUT_SIZE];
    double d_weights1[INPUT_SIZE * HIDDEN_SIZE] = {0};
    double d_bias1[HIDDEN_SIZE] = {0};
    double d_weights2[HIDDEN_SIZE * OUTPUT_SIZE] = {0};
    double d_bias2[OUTPUT_SIZE] = {0};

    double hidden_test[HIDDEN_SIZE], output_test[OUTPUT_SIZE];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        train_loss = 0.0;
        for (int i = 0; i < NUM_TRAIN; i++) {
            for (int o = 0; o < OUTPUT_SIZE; o++) target[o] = (train_labels[i] == o) ? 1.0 : 0.0;
            forward(train_in[i], hidden, output);
            backward(train_in[i], hidden, output, target, d_weights1, d_bias1, d_weights2, d_bias2);
        }
        update_weights(d_weights1, d_bias1, d_weights2, d_bias2);

        int correct = 0;
        for (int i = 0; i < NUM_TEST; i++) {
            int pred = predict(test_in[i], hidden_test, output_test);
            if (pred == test_labels[i]) correct++;
        }
        double acc = (double)correct / NUM_TEST * 100.0;
        printf("Эпоха %d: Loss=%.4f, Test Acc=%.1f%%\n", epoch + 1, train_loss / NUM_TRAIN, acc);
    }

    return 0;
}