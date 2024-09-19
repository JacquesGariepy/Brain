import unittest
from modules.language import LanguageModule
from modules.memory import MemoryModule

class TestLanguageModule(unittest.TestCase):
    def setUp(self):
        """Initialise le module de langage pour les tests."""
        self.memory = MemoryModule()
        self.language_module = LanguageModule(self.memory)

    def test_learn_text(self):
        """Teste l'apprentissage de nouveaux mots et l'ajout au vocabulaire."""
        text = "Le cerveau humain est un organe fascinant."
        self.language_module.learn_text(text)
        self.assertIn("cerveau", self.language_module.vocabulary)

    def test_generate_sentence(self):
        """Teste la génération de phrases."""
        sentence = self.language_module.generate_sentence(prompt="Le cerveau")
        self.assertTrue(len(sentence) > 0)

    def test_understand_sentence(self):
        """Teste la compréhension de phrases."""
        sentence = "Le cerveau humain est fascinant."
        comprehension = self.language_module.understand_sentence(sentence)
        self.assertIn("comprise", comprehension)
