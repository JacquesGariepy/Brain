import unittest
import numpy as np
from modules.learning import LearningModule
from modules.network import Network
from modules.memory import MemoryModule
from modules.neuron import Neuron

class TestLearningModule(unittest.TestCase):
    def setUp(self):
        """Initialise le réseau et le module d'apprentissage pour les tests."""
        self.network = Network()
        self.memory = MemoryModule()
        self.learning_module = LearningModule(self.network, self.memory)

        # Crée des neurones et les ajoute au réseau
        for i in range(5):
            neuron = Neuron(neuron_id=i)
            self.network.add_neuron(neuron)

    def test_supervised_learning(self):
        """Teste l'apprentissage supervisé avec des exemples étiquetés."""
        inputs = np.random.rand(len(self.network.neurons))
        targets = np.random.randint(0, 2, len(self.network.neurons))
        initial_weights = [synapse.weight for synapse in self.network.synapses]
        
        self.learning_module.supervised_learning(inputs, targets)
        
        # Vérifie que les poids synaptiques ont changé après l'apprentissage
        for synapse, initial_weight in zip(self.network.synapses, initial_weights):
            self.assertNotEqual(synapse.weight, initial_weight)

    def test_unsupervised_learning(self):
        """Teste l'apprentissage non supervisé basé sur le clustering."""
        inputs = np.random.rand(len(self.network.neurons))
        self.learning_module.unsupervised_learning(inputs)

        # Vérifie que les poids synaptiques ont été ajustés
        for synapse in self.network.synapses:
            self.assertTrue(0 <= synapse.weight <= 1)

    def test_reinforcement_learning(self):
        """Teste l'apprentissage par renforcement avec une récompense."""
        reward = 1  # Récompense positive
        initial_weights = [synapse.weight for synapse in self.network.synapses]
        
        self.learning_module.reinforcement_learning(reward)
        
        # Vérifie que les poids synaptiques ont été mis à jour
        for synapse, initial_weight in zip(self.network.synapses, initial_weights):
            self.assertNotEqual(synapse.weight, initial_weight)
