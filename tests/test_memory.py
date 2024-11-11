import unittest
from modules.memory import MemoryModule
from modules.synapse import Synapse
from modules.neuron import Neuron

class TestMemoryModule(unittest.TestCase):
    def setUp(self):
        """Initialise le module de mémoire pour les tests."""
        self.memory_module = MemoryModule()
        self.pre_neuron = Neuron(neuron_id=0)
        self.post_neuron = Neuron(neuron_id=1)
        self.synapse = Synapse(self.pre_neuron, self.post_neuron)

    def test_long_term_potentiation(self):
        """Teste la potentialisation à long terme (LTP)."""
        initial_weight = self.synapse.weight
        self.memory_module.long_term_potentiation(self.synapse)
        self.assertGreater(self.synapse.weight, initial_weight)

    def test_hippocampal_involvement(self):
        """Teste l'implication de l'hippocampe dans la consolidation de la mémoire."""
        data = "test_data"
        self.memory_module.hippocampal_involvement(data)
        self.assertIn(data, self.memory_module.short_term_memory)
        if len(self.memory_module.short_term_memory) == self.memory_module.short_term_memory.maxlen:
            self.assertIn("consolidated_memory", self.memory_module.long_term_memory)

    def test_memory_consolidation(self):
        """Teste la consolidation de la mémoire."""
        data = "test_data"
        self.memory_module.store_short_term(data)
        self.memory_module.memory_consolidation()
        self.assertIn(f"consolidated_{data}", self.memory_module.long_term_memory)

    def test_distributed_storage(self):
        """Teste le stockage distribué des souvenirs."""
        data = "test_data"
        self.memory_module.distributed_storage(data)
        regions = ["region_1", "region_2", "region_3"]
        for region in regions:
            self.assertIn(f"{region}_{data}", self.memory_module.long_term_memory)

    def test_neurogenesis(self):
        """Teste la neurogenèse en ajoutant de nouveaux neurones à la mémoire."""
        new_neurons = ["neuron_1", "neuron_2"]
        self.memory_module.neurogenesis(new_neurons)
        self.assertIn("new_neurons", self.memory_module.long_term_memory)
        self.assertEqual(self.memory_module.long_term_memory["new_neurons"], new_neurons)

if __name__ == '__main__':
    unittest.main()
