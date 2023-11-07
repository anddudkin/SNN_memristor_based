import torch


class MnistAssigment:
    def __init__(self, n_neurons_out):
        self.dict_labels = {}
        self.n_neurons_out = n_neurons_out
        self.assigments = {}

        for n in range(self.n_neurons_out):
            self.dict_labels[n] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    def count_spikes_train(self, spikes, label):
        """Count how much each neuron reacted for each label

            Args:
                spikes : current spikes
                label : current label
        """
        for j, i in enumerate(spikes, start=0):
            if i == 1:
                self.dict_labels[j][int(label)] += 1

    def get_assigment(self):
        """Return assigned label for each neuron expl: {0: 9, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1}"""
        for n in range(self.n_neurons_out):
            self.assigments[n] = list(self.dict_labels[n].keys())[
                list(self.dict_labels[n].values()).index(max(self.dict_labels[n].values()))]


class MnistEvaluation:
    def __init__(self, n_neurons_out):
        self.n_neurons_out = n_neurons_out
        self.spikes_counter = torch.zeros([self.n_neurons_out], dtype=torch.int)
        self.good = 0
        self.bad = 0

    def count_spikes(self, spikes):
        self.spikes_counter += spikes

    def conclude(self, assigment, label):
        if assigment[int(torch.argmax(self.spikes_counter))] == int(label):
            self.good += 1
        else:
            self.bad += 1
        self.spikes_counter.fill_(0)

    def final(self):
        print("Correctly defined", self.good)
        print("Incorrectly defined", self.bad)
        print("Final result", round((self.good / (self.bad + self.good) * 100), 2))
