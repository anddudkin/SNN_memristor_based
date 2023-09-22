import torch



def construct_matrix_connections(n_in_neurons, n_out_neurons, type):
    """" type of connection """

    matrix_conn = torch.zeros([n_in_neurons * n_out_neurons, 3], dtype=torch.float)

    if type == "all_to_all":
        while matrix_conn[n_in_neurons * n_out_neurons - 1][0] == 0:
            ind = 0
            for i in range(n_in_neurons):
                for j in range(n_out_neurons):
                    matrix_conn[ind][0] = i
                    matrix_conn[ind][1] = j
                    ind += 1

    return matrix_conn


b = construct_matrix_connections(4, 10, "all_to_all")
print(b)
