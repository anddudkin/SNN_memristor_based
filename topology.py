from random import random

import torch


def construct_matrix_connections(n_in_neurons, n_out_neurons, type):
    """ type of connection: 1) "all_to_all" 2).....
    """

    matrix_conn = torch.zeros([n_in_neurons * n_out_neurons, 3], dtype=torch.float32)

    if type == "all_to_all":
        while matrix_conn[n_in_neurons * n_out_neurons - 1][0] == 0:
            ind = 0
            for i in range(n_out_neurons):
                for j in range(n_in_neurons):
                    matrix_conn[ind][0] = i  # [out,in,w]
                    matrix_conn[ind][1] = j
                    ind += 1

    return matrix_conn


def inicialize_weights(martix_conn):
    for i in range(len(martix_conn)):
        martix_conn[i][2] = random()
    return martix_conn


def compute_det_w(matrix_conn, I_in):
    pass
