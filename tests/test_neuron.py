import unittest
from modules.neuron import Neuron

class TestNeuron(unittest.TestCase):
    def test_neuron_potential_update(self):
        neuron = Neuron(neuron_id=1)
        neuron.update_potential(input_current=10, dt=1.0)
        self.assertNotEqual(neuron.v_m, neuron.v_rest)

    def test_neuron_spike(self):
        neuron = Neuron(neuron_id=1, v_threshold=-50.0)
        neuron.update_potential(input_current=100, dt=1.0)
        self.assertTrue(neuron.spike)
