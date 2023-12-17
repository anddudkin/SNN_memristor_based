import torch
import pickle


class MnistAssignment:
    """Class for lables assignment of SNN with MNIST dataset"""

    def __init__(self, n_neurons_out):
        self.dict_labels = {}
        self.n_neurons_out = n_neurons_out
        self.assignments = {}

        for n in range(self.n_neurons_out):
            self.dict_labels[n] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    def count_spikes_train(self, spikes, label):
        """Counts how much each neuron spiked for each label

            Args:
                spikes : current spikes
                label : current label
        """
        for j, i in enumerate(spikes, start=0):
            if i == 1:
                self.dict_labels[j][int(label)] += 1

    def get_assignment(self):
        """Return assigned label for each neuron expl: {0: 9, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1}"""
        for n in range(self.n_neurons_out):
            self.assignments[n] = list(self.dict_labels[n].keys())[
                list(self.dict_labels[n].values()).index(max(self.dict_labels[n].values()))]

    def save_assignment(self):
        with open('assignments.pkl', 'wb') as f:
            pickle.dump(self.assignments, f)

    def load_assignment(self, path="assignments.pkl"):
        with open(path, 'rb') as f:
            self.assignments = pickle.load(f)


class MnistEvaluation:
    """Class for result evaluation of SNN with MNIST dataset"""

    def __init__(self, n_neurons_out):
        self.n_neurons_out = n_neurons_out
        self.spikes_counter = torch.zeros([self.n_neurons_out], dtype=torch.int)
        self.good = 0
        self.bad = 0

    def count_spikes(self, spikes):
        """Counts how much each neuron spiked for presented image"""
        self.spikes_counter += spikes

    def conclude(self, assigment, label):
        """Counts how much images were defined correctly and incorrectly
        Args:
                assigment : assignment for each neuron (from MnistAssignment.assignments )
                label : current image label
        """
        if assigment[int(torch.argmax(self.spikes_counter))] == int(label) and int(torch.max(self.spikes_counter)) != 0:
            self.good += 1
        elif int(torch.max(self.spikes_counter)) == 0:
            pass
        else:
            self.bad += 1
            # print(self.spikes_counter, int(label))

        self.spikes_counter.fill_(0)

    def final(self):
        """Prints test results"""
        print("Test Completed")
        print("Correctly defined images:", self.good)
        print("Incorrectly defined images:", self.bad)
        print(f"Final result: {round((self.good / (self.bad + self.good) * 100), 2)} %")
        return "\nTest Completed\n" + "Correctly defined images:" + str(
            self.good) + "\nIncorrectly defined images:" + str(self.bad) + "\nFinal result:" + str(
            round((self.good / (self.bad + self.good) * 100), 2))
