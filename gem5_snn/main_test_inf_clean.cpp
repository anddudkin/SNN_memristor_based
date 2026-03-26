
#include <vector>
#include <cmath>

const int N_INPUT = 28 * 28;
const int N_NEURONS = 100;
const int TIME_STEPS = 25;
const int N_IMAGES = 50;
const float TAU_M = 15.0f;
const float THRESHOLD = 2.0f;  //
const float REST = 0.0f;
const int REFRACTORY_PERIOD = 15;
const float INH_COEF = 0.9f;

struct LIFNeuron {
    float v = 0.0f;
    int refractory = 0;
};

class SNN {
private:
    std::vector<std::vector<float>> W;
    std::vector<LIFNeuron> neurons;

public:
    SNN() {
        W.resize(N_NEURONS, std::vector<float>(N_INPUT, 0.0f));
        neurons.resize(N_NEURONS);

        for(int j = 0; j < N_NEURONS; j++) {
            for(int i = 0; i < N_INPUT; i++) {
                W[j][i] = 0.01f * (rand() / (float)RAND_MAX);  // ↑ Средние веса
            }
        }
    }

    float lif_update(int j, const float* spikes_in) {
        LIFNeuron& neuron = neurons[j];

        if (neuron.refractory > 0) {
            neuron.refractory--;
            return 0.0f;
        }

        float i_in = 0.0f;
        for(int i = 0; i < N_INPUT; i++) {
            i_in += spikes_in[i] * W[j][i];
        }

        neuron.v = neuron.v * (1.0f - 1.0f / TAU_M) + i_in;  // ↑ Масштаб ввода

        if (neuron.v >= THRESHOLD) {
            neuron.v = REST;
            neuron.refractory = REFRACTORY_PERIOD;
            return 1.0f;
        }
        return 0.0f;
    }

    int forward(const float* input_spikes, float* spikes_out) {
        std::fill(spikes_out, spikes_out + N_NEURONS, 0.0f);

        int raw_spikes = 0;
        for(int j = 0; j < N_NEURONS; j++) {
            spikes_out[j] = lif_update(j, input_spikes);
            if(spikes_out[j] > 0) raw_spikes++;
        }

        static int total_raw = 0;

        int winner = -1;
        float max_v = -1.0f;
        for(int j = 0; j < N_NEURONS; j++) {
            if(spikes_out[j] > 0.0f && neurons[j].v > max_v) {
                max_v = neurons[j].v;
                winner = j;
                total_raw += 1;
            }
        }

        if(winner >= 0) {
            std::fill(spikes_out, spikes_out + N_NEURONS, 0.0f);
            spikes_out[winner] = 1.0f;

            for(int j = 0; j < N_NEURONS; j++) {
                if(j != winner && neurons[j].refractory == 0) {
                    neurons[j].v *= INH_COEF;
                }
            }
        }
        return (winner >= 0) ? 1 : 0;
    }
};

void generate_image_spikes(float* spikes, float avg_rate = 0.3f) {  // ↑ Больше спайков
    for(int i = 0; i < N_INPUT; i++) {
        float rate = 0.08f + 0.1f * (rand() / (float)RAND_MAX);
        spikes[i] = (rand() / (float)RAND_MAX < rate) ? 1.0f : 0.0f;
    }
}

int main() {
    float* input_spikes = new float[N_INPUT];
    float* spikes_out = new float[N_NEURONS];

    SNN snn;
    long total_wta_spikes = 0;

    for(int img = 0; img < N_IMAGES; img++) {

        for(int t = 0; t < TIME_STEPS; t++) {
            generate_image_spikes(input_spikes, 0.3f);

            total_wta_spikes += snn.forward(input_spikes, spikes_out);
        }
    }

    delete[] input_spikes;
    delete[] spikes_out;
    return 0;
}