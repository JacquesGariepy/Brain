# core/brain.py

import importlib
import os
from typing import Dict
from .interfaces import BrainModule

class Brain:
    def __init__(self):
        self.modules: Dict[str, BrainModule] = {}
        self.load_core_modules()
        self.load_plugins()

    def load_core_modules(self):
        from .perception import PerceptionModule
        from .language import LanguageModule
        from .reasoning import ReasoningModule

        self.modules['perception'] = PerceptionModule()
        self.modules['language'] = LanguageModule()
        self.modules['reasoning'] = ReasoningModule()

    def load_plugins(self):
        plugin_folder = 'plugins'
        for filename in os.listdir(plugin_folder):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                module = importlib.import_module(f'plugins.{module_name}')
                plugin_class = getattr(module, 'Plugin')
                plugin_instance = plugin_class()
                self.modules[module_name] = plugin_instance
                print(f'Plugin chargé : {module_name}')

    def process(self, data):
        # Flux de traitement de base
        data = self.modules['perception'].process(data)
        data = self.modules['language'].process(data)
        data = self.modules['reasoning'].process(data)
        # Traitement avec les plugins
        for module_name, module in self.modules.items():
            if module_name not in ['perception', 'language', 'reasoning']:
                data = module.process(data)
        return data

    def inject_knowledge(self, text):
        """
        Injecte des compétences ou des connaissances dans le cerveau via le module de langage.
        
        Args:
            text (str): Texte à apprendre (par exemple, un texte sur une nouvelle compétence).
        
        Raises:
            ValueError: Si le texte fourni est vide ou mal formé.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Le texte fourni pour l'injection de connaissances est invalide.")
        
        try:
            self.modules['language'].learn_text(text)
            print("Nouvelle compétence injectée dans le cerveau.")
        except Exception as e:
            print(f"Erreur lors de l'injection de connaissances : {str(e)}")
