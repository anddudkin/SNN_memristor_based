from main import compute_ideal
from tests import test_values
def Neuron_Integrator(threshold):
     a,b=test_values()[0],test_values()[1],
     I_out = compute_ideal(a,b)
     ''' add neurons
          compute'''


Neuron_Integrator(1)
