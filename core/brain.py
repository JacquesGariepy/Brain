# core/brain.py

import importlib
import os
from typing import Dict
from .interfaces import BrainModule
from modules.attention import AttentionModule
from modules.decision import DecisionModule
from modules.emotion import EmotionModule
from modules.learning import LearningModule
from modules.memory import MemoryModule
from modules.network import Network
from modules.neuron import Neuron
from modules.synapse import Synapse

class Brain:
    def __init__(self):
        self.modules: Dict[str, BrainModule] = {}
        self.load_core_modules()
        self.load_plugins()
        self.attention_module = AttentionModule([])
        self.decision_module = DecisionModule()
        self.emotion_module = EmotionModule()
        self.learning_module = LearningModule(self.network, self.memory_module)
        self.memory_module = MemoryModule()
        self.network = Network()
        self.neurons = []
        self.synapses = []
        self.create_neurons_and_synapses()

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

    def create_neurons_and_synapses(self):
        """
        Crée les neurones et les synapses pour le réseau neuronal.
        """
        for i in range(10):
            neuron = Neuron(neuron_id=i)
            self.neurons.append(neuron)
            self.network.add_neuron(neuron)
        
        for pre_neuron in self.neurons:
            for post_neuron in self.neurons:
                if pre_neuron != post_neuron:
                    synapse = Synapse(pre_neuron, post_neuron)
                    self.synapses.append(synapse)
                    self.network.connect_neurons(pre_neuron, post_neuron)

    def perceive_and_process(self, sensory_input, dt):
        """
        Perçoit et traite les entrées sensorielles.
        
        Args:
            sensory_input (list): Liste des entrées sensorielles.
            dt (float): Pas de temps de simulation.
        """
        self.modules['perception'].encode_sensory_input(sensory_input)
        self.network.update(dt)
        self.emotion_module.update_emotions(sensory_input, self.memory_module.retrieve_short_term(), 0, dt)
        self.attention_module.update_attention({neuron.neuron_id: 1.0 for neuron in self.neurons})

    def execute_decision(self, dt):
        """
        Exécute une décision basée sur l'accumulation d'évidence.
        
        Args:
            dt (float): Pas de temps de simulation.
        """
        evidence = 0.5  # Exemple d'évidence
        self.decision_module.update_decision(evidence, self.emotion_module.emotional_states["fear"], dt)
        if self.decision_module.choice_made:
            print(f"Décision prise : {self.decision_module.decision}")
            self.decision_module.reset()

    def learn(self, inputs, targets):
        """
        Apprend à partir des entrées et des cibles.
        
        Args:
            inputs (array-like): Entrées du réseau.
            targets (array-like): Sorties attendues.
        """
        self.learning_module.supervised_learning(inputs, targets)

    def communicate(self, prompt):
        """
        Communique en générant une phrase à partir d'un prompt.
        
        Args:
            prompt (str): Prompt initial pour générer du texte.
            
        Returns:
            str: Phrase générée.
        """
        return self.modules['language'].generate_sentence(prompt)

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

    def save_state(self):
        """Sauvegarde l'état du cerveau."""
        self.memory_module.save_long_term_memory()

    def load_state(self):
        """Charge l'état du cerveau."""
        self.memory_module.load_long_term_memory()
