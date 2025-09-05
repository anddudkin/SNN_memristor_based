import torch
from tqdm import tqdm
from Network.assigment import MnistAssignment, MnistEvaluation
from Network.visuals import plot_weights_square
from Network.topology import Connections
from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14
from Network.NeuronModels import NeuronLifAdaptiveThresh
import matplotlib.pyplot as plt
import time as t
import multiprocessing
import numpy as np


def snn(ar):

    n_neurons_out = 50  # number of neurons in input layer
    n_neurons_in = 196  # number of output in input layer
    n_train = 2000  # number of images for training
    n_test = 3000 # number of images for testing
    time = 350  # time of each image presentation during training
    time_test = 200  # time of each image presentation during testing
    test = True  # do testing or not
    plot = False  # plot graphics or not
    x = ar
    probabil = ar
    mask = np.random.binomial(n=1, p=probabil / 2, size=[196, n_neurons_out])
    mask1 = np.random.binomial(n=1, p=probabil / 2, size=[196, n_neurons_out])

    # модель с реальными физическими величинами и линейно
    # дискретным диапазоном значений проводимости
    out_neurons = NeuronLifAdaptiveThresh(n_neurons_in,
                                          n_neurons_out,
                                          train=True,
                                          U_mem=0,
                                          decay=0.92,
                                          U_tr=20 / 100,
                                          U_rest=0,
                                          refr_time=5,
                                          traces=True,
                                          inh=True)  # activate literal inhibition

    conn = Connections(n_neurons_in, n_neurons_out, "all_to_all", w_min=0.00005, w_max=0.01)
    conn.all_to_all_conn()
    conn.initialize_weights("normal")

    data_train = MNIST_train_test_14x14()[0]
    data_test = MNIST_train_test_14x14()[1]

    assig = MnistAssignment(n_neurons_out)


    train_labels = [0, 1, 2, 9, 5]


   #
    assig = MnistAssignment(n_neurons_out)
    # assig.load_assignment(
    #     '/home/anddudkin/PycharmProjects/SNN_memristor_based/Memristor/stuck_elements_tests/data_MNIST_min_max_R/results3/assignments50.pkl')
    # conn.load_weights(
    #     '/home/anddudkin/PycharmProjects/SNN_memristor_based/Memristor/stuck_elements_tests/data_MNIST_min_max_R/results3/weights_tensor50.pt')
    # out_neurons.load_U_thresh(
    #     '/home/anddudkin/PycharmProjects/SNN_memristor_based/Memristor/stuck_elements_tests/data_MNIST_min_max_R/results3/thresh50.pt')
    assig.load_assignment('assignments2048.pkl')
    conn.load_weights('weights_tensor2048.pt')
    out_neurons.load_U_thresh('thresh2048.pt')
    evall = MnistEvaluation(n_neurons_out)
    if ar > 0.01:
        for rr in range(196):
            for rr1 in range(n_neurons_out):
                if mask[rr][rr1] == 1:
                    conn.weights[rr][rr1] = np.max(conn.w_min)
                if mask1[rr][rr1] == 1:
                    conn.weights[rr][rr1] = np.max(conn.w_max)



    out_neurons.train = False
    out_neurons.reset_variables(True, True, True)
    d = []
    if test:
        for i in tqdm(range(n_test), desc='test', colour='green', position=0):
            d.append(data_train[i][1])
            if data_train[i][1] in train_labels:
                input_spikes = encoding_to_spikes(data_train[i][0], time_test)

                for j in range(time_test):
                    out_neurons.compute_U_mem(input_spikes[j].reshape(196), conn.weights)
                    out_neurons.check_spikes1()
                    evall.count_spikes(out_neurons.spikes)
                evall.conclude(assig.assignments, data_train[i][1])

    evall.final()
    print(d)
    # with open('result' + str(int(ar*100)) + '.txt', 'a+') as f:
    #     f.write("\ntrain: " + str(train_labels))
    #     f.write("\ntrain: " + str(n_train))
    #     f.write("\ntest: " + str(n_test))
    #     f.write("\ntime_train: " + str(time))
    #     f.write("\ntime_train: " + str(time_test))
    #     f.write("\nneurons out: " + str(n_neurons_out))
    #     f.write("\ntrain images: " + str(0))
    #
    #     f.write(evall.final())
    x = evall.final(only_result_number=True)
    plt.imshow(conn.weights, cmap = "gray")
    plt.title(str(int(ar*100))+"% stuck elements, "+x + " %, test result" )
    plt.savefig("fid"+str(int(ar*100))+".png")
if __name__ == "__main__":

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    #result = pool.map(snn, [x/100 for x in list(range(30,31))])
    result = pool.map(snn, [0.01, 0.02, 0.05,0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
#5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9, 4