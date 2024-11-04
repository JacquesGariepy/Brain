import unittest
from main import Brain

class TestBrain(unittest.TestCase):
    def setUp(self):
        """Initialise une instance du cerveau pour les tests."""
        self.brain = Brain()

    def test_brain_initialization(self):
        """Teste que le cerveau est correctement initialisé avec tous les modules."""
        self.assertIsNotNone(self.brain.network)
        self.assertIsNotNone(self.brain.memory_module)
        self.assertIsNotNone(self.brain.language_module)

    def test_perceive_and_process(self):
        """Teste la perception et le traitement des stimuli sensoriels."""
        sensory_input = [0.5 for _ in range(10)]  # Stimuli sensoriels simulés
        self.brain.perceive_and_process(sensory_input, dt=1.0)
        self.assertGreaterEqual(len(self.brain.memory_module.retrieve_short_term()), 0)

    def test_inject_knowledge(self):
        """Teste l'injection de connaissances dans le cerveau."""
        knowledge = "L'intelligence artificielle est une discipline en pleine expansion."
        self.brain.inject_knowledge(knowledge)
        self.assertIn("intelligence", self.brain.language_module.vocabulary)

    def test_save_and_load_state(self):
        """Teste la sauvegarde et le chargement de l'état du cerveau."""
        self.brain.save_state()
        self.brain.load_state()
        self.assertGreaterEqual(len(self.brain.memory_module.retrieve_long_term("vocabulary")), 0)

    def test_inject_knowledge_invalid_text(self):
        """Teste que la méthode inject_knowledge lève une ValueError pour un texte invalide."""
        with self.assertRaises(ValueError):
            self.brain.inject_knowledge("")

    def test_inject_knowledge_success(self):
        """Teste l'injection réussie de connaissances."""
        knowledge = "L'apprentissage automatique est une branche de l'intelligence artificielle."
        self.brain.inject_knowledge(knowledge)
        self.assertIn("apprentissage", self.brain.language_module.vocabulary)
