from compute import compute_ideal
from tests import test_values
def Neuron_Integrator(I_in, U_tr):
     '''I_in - Input current
        U_tr -  max capacity of neuron (Treshold). If U_mem > U_tr neuron spikes
          '''
     a,b=test_values()[0],test_values()[1],
     I_out = compute_ideal(a,b)
     ''' add neurons
          compute'''


Neuron_Integrator(1)
