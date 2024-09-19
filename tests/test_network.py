import unittest
from modules.network import Network
from modules.neuron import Neuron
from modules.synapse import Synapse

class TestNetwork(unittest.TestCase):
    def setUp(self):
        """Initialise un réseau neuronal pour les tests."""
        self.network = Network()
        self.neuron1 = Neuron(neuron_id=0)
        self.neuron2 = Neuron(neuron_id=1)
        self.network.add_neuron(self.neuron1)
        self.network.add_neuron(self.neuron2)

    def test_add_neuron(self):
        """Teste l'ajout de neurones au réseau."""
        self.assertEqual(len(self.network.neurons), 2)

    def test_connect_neurons(self):
        """Teste la création de synapses entre deux neurones."""
        self.network.connect_neurons(self.neuron1, self.neuron2, weight=0.7, delay=1.5)
        self.assertEqual(len(self.network.synapses), 1)
        synapse = self.network.synapses[0]
        self.assertEqual(synapse.pre_neuron, self.neuron1)
        self.assertEqual(synapse.post_neuron, self.neuron2)
        self.assertEqual(synapse.weight, 0.7)

    def test_network_update(self):
        """Teste la mise à jour du réseau après un pas de temps."""
        self.network.connect_neurons(self.neuron1, self.neuron2)
        self.neuron1.spike = True
        self.network.update(dt=1.0)
        self.assertFalse(self.neuron1.spike)  # Le neurone doit être réinitialisé après un spike
