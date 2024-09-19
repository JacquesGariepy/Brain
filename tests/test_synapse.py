import unittest
from modules.synapse import Synapse
from modules.neuron import Neuron

class TestSynapse(unittest.TestCase):
    def setUp(self):
        """Crée deux neurones et une synapse pour les tests."""
        self.pre_neuron = Neuron(neuron_id=0)
        self.post_neuron = Neuron(neuron_id=1)
        self.synapse = Synapse(self.pre_neuron, self.post_neuron)

    def test_synapse_weight_initialization(self):
        """Teste que le poids initial de la synapse est correct."""
        self.assertEqual(self.synapse.weight, 0.5)

    def test_synapse_transmission(self):
        """Teste que la synapse transmet correctement les spikes."""
        self.pre_neuron.spike = True
        self.synapse.transmit_spike()
        self.assertEqual(len(self.synapse.spike_times), 1)

    def test_stdp_weight_update(self):
        """Teste la mise à jour du poids synaptique selon la règle STDP."""
        delta_t = 10
        initial_weight = self.synapse.weight
        self.synapse.update_weight_stdp(delta_t)
        self.assertNotEqual(self.synapse.weight, initial_weight)

    def test_synapse_current(self):
        """Teste que la synapse génère correctement le courant synaptique."""
        self.synapse.spike_times.append(1)
        current = self.synapse.get_current()
        self.assertGreater(current, 0)
