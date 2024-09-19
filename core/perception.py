# core/perception.py

from .interfaces import BrainModule

class PerceptionModule(BrainModule):
    def process(self, data):
        print("Module de Perception traite les données.")
        # Implémenter la logique de perception ici
        return "données_perçues"

# core/language.py

from .interfaces import BrainModule

class LanguageModule(BrainModule):
    def process(self, data):
        print("Module de Langage traite les données.")
        # Implémenter la logique du langage ici
        return "données_langage"

# core/reasoning.py

from .interfaces import BrainModule

class ReasoningModule(BrainModule):
    def process(self, data):
        print("Module de Raisonnement traite les données.")
        # Implémenter la logique de raisonnement ici
        return "données_raisonnées"
