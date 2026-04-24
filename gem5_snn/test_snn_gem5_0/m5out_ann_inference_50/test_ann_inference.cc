#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 10
#define NUM_INFERENCE 50

double weights1[INPUT_SIZE * HIDDEN_SIZE];
double bias1[HIDDEN_SIZE];
double weights2[HIDDEN_SIZE * OUTPUT_SIZE];
double bias2[OUTPUT_SIZE];

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void init_weights() {
    srand(42);
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
        weights1[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
        weights2[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
}

int main() {
    double test_image[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++)
        test_image[i] = 1.0;

    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];

    for (int img = 0; img < NUM_INFERENCE; img++) {
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            double sum = 0;
            for (int i = 0; i < INPUT_SIZE; i++)
                sum += test_image[i] * weights1[i * HIDDEN_SIZE + h];
            hidden[h] = sigmoid(sum);
        }

        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double sum = 0;
            for (int h = 0; h < HIDDEN_SIZE; h++)
                sum += hidden[h] * weights2[h * OUTPUT_SIZE + o];
            output[o] = sum;
        }
    }

    return 0;
}