import torch

from datasets import test_values
from NeuronModels import Neuron_IF
from compute_crossbar import compute_ideal
x=compute_ideal(test_values()[0],test_values()[1])[0]
print(Neuron_IF(x,40,40))
