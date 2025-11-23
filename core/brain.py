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
from utils.logging import brain_logger
from utils.exceptions import BrainException


class Brain:
    """
    Classe principale du système Brain - Réseau neuronal bio-inspiré.
    
    Attributes:
        modules (Dict[str, BrainModule]): Modules cognitifs du cerveau.
        network (Network): Réseau neuronal.
        neurons (list): Liste des neurones.
        synapses (list): Liste des synapses.
    """
    
    def __init__(self, num_neurons=10):
        """
        Initialise le cerveau avec tous ses modules.
        
        Args:
            num_neurons (int): Nombre de neurones à créer.
        """
        brain_logger.info("Initialisation du cerveau Brain")
        
        # Initialiser les structures de base en premier
        self.modules: Dict[str, BrainModule] = {}
        self.network = Network()
        self.neurons = []
        self.synapses = []
        self.num_neurons = num_neurons
        
        # Initialiser les modules de base
        self.memory_module = MemoryModule()
        self.emotion_module = EmotionModule()
        self.decision_module = DecisionModule()
        self.attention_module = AttentionModule([])
        
        # Charger les modules core
        self.load_core_modules()
        
        # Initialiser le module d'apprentissage (nécessite network et memory)
        self.learning_module = LearningModule(self.network, self.memory_module)
        
        # Créer les neurones et synapses
        self.create_neurons_and_synapses()
        
        # Mettre à jour le module d'attention avec les neurones créés
        self.attention_module = AttentionModule(self.neurons)
        
        # Charger les plugins
        self.load_plugins()
        
        brain_logger.info(f"Cerveau initialisé avec {len(self.neurons)} neurones et {len(self.synapses)} synapses")

    def load_core_modules(self):
        """Charge les modules core du cerveau."""
        try:
            from .perception import PerceptionModule
            from .language import LanguageModule
            from .reasoning import ReasoningModule

            self.modules['perception'] = PerceptionModule()
            self.modules['language'] = LanguageModule()
            self.modules['reasoning'] = ReasoningModule()
            
            brain_logger.info("Modules core chargés avec succès")
        except Exception as e:
            brain_logger.error(f"Erreur lors du chargement des modules core: {str(e)}")
            raise BrainException(f"Impossible de charger les modules core: {str(e)}")

    def load_plugins(self):
        """Charge les plugins depuis le dossier plugins."""
        try:
            plugin_folder = 'plugins'
            if not os.path.exists(plugin_folder):
                brain_logger.warning(f"Dossier de plugins '{plugin_folder}' introuvable")
                return
            
            for filename in os.listdir(plugin_folder):
                if filename.endswith('.py') and not filename.startswith('__') and filename != 'plugin_interface.py':
                    try:
                        module_name = filename[:-3]
                        module = importlib.import_module(f'plugins.{module_name}')
                        if hasattr(module, 'Plugin'):
                            plugin_class = getattr(module, 'Plugin')
                            plugin_instance = plugin_class()
                            self.modules[module_name] = plugin_instance
                            brain_logger.info(f'Plugin chargé : {module_name}')
                    except Exception as e:
                        brain_logger.warning(f"Impossible de charger le plugin {module_name}: {str(e)}")
        except Exception as e:
            brain_logger.error(f"Erreur lors du chargement des plugins: {str(e)}")

    def process(self, data):
        """
        Traite les données à travers tous les modules du cerveau.
        
        Args:
            data: Données à traiter.
            
        Returns:
            Données traitées.
        """
        try:
            brain_logger.debug("Début du traitement des données")
            
            # Flux de traitement de base
            data = self.modules['perception'].process(data)
            data = self.modules['language'].process(data)
            data = self.modules['reasoning'].process(data)
            
            # Traitement avec les plugins
            for module_name, module in self.modules.items():
                if module_name not in ['perception', 'language', 'reasoning']:
                    data = module.process(data)
            
            brain_logger.debug("Traitement des données terminé")
            return data
        except Exception as e:
            brain_logger.error(f"Erreur lors du traitement des données: {str(e)}")
            raise BrainException(f"Erreur de traitement: {str(e)}")

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
            brain_logger.info("Nouvelle compétence injectée dans le cerveau")
        except Exception as e:
            brain_logger.error(f"Erreur lors de l'injection de connaissances : {str(e)}")
            raise

    def create_neurons_and_synapses(self):
        """Crée les neurones et les synapses pour le réseau neuronal."""
        try:
            brain_logger.info(f"Création de {self.num_neurons} neurones")
            
            for i in range(self.num_neurons):
                neuron = Neuron(neuron_id=i)
                self.neurons.append(neuron)
                self.network.add_neuron(neuron)
            
            # Créer les connexions synaptiques
            synapse_count = 0
            for pre_neuron in self.neurons:
                for post_neuron in self.neurons:
                    if pre_neuron != post_neuron:
                        synapse = Synapse(pre_neuron, post_neuron)
                        self.synapses.append(synapse)
                        self.network.connect_neurons(pre_neuron, post_neuron)
                        synapse_count += 1
            
            brain_logger.info(f"{synapse_count} synapses créées")
        except Exception as e:
            brain_logger.error(f"Erreur lors de la création des neurones et synapses: {str(e)}")
            raise BrainException(f"Impossible de créer le réseau neuronal: {str(e)}")

    def perceive_and_process(self, sensory_input, dt):
        """
        Perçoit et traite les entrées sensorielles.
        
        Args:
            sensory_input (list): Liste des entrées sensorielles.
            dt (float): Pas de temps de simulation.
        """
        try:
            self.modules['perception'].encode_sensory_input(sensory_input)
            self.network.update(dt)
            self.emotion_module.update_emotions(sensory_input, self.memory_module.retrieve_short_term(), 0, dt)
            self.attention_module.update_attention({neuron.neuron_id: 1.0 for neuron in self.neurons})
        except Exception as e:
            brain_logger.error(f"Erreur lors de la perception et du traitement: {str(e)}")

    def execute_decision(self, dt):
        """
        Exécute une décision basée sur l'accumulation d'évidence.
        
        Args:
            dt (float): Pas de temps de simulation.
        """
        try:
            evidence = 0.5  # Exemple d'évidence
            self.decision_module.update_decision(evidence, self.emotion_module.emotional_states["fear"], dt)
            if self.decision_module.choice_made:
                brain_logger.info(f"Décision prise : {self.decision_module.decision}")
                self.decision_module.reset()
        except Exception as e:
            brain_logger.error(f"Erreur lors de l'exécution de la décision: {str(e)}")

    def learn(self, inputs, targets):
        """
        Apprend à partir des entrées et des cibles.
        
        Args:
            inputs (array-like): Entrées du réseau.
            targets (array-like): Sorties attendues.
        """
        try:
            self.learning_module.supervised_learning(inputs, targets)
        except Exception as e:
            brain_logger.error(f"Erreur lors de l'apprentissage: {str(e)}")

    def communicate(self, prompt):
        """
        Communique en générant une phrase à partir d'un prompt.
        
        Args:
            prompt (str): Prompt initial pour générer du texte.
            
        Returns:
            str: Phrase générée.
        """
        try:
            return self.modules['language'].generate_sentence(prompt)
        except Exception as e:
            brain_logger.error(f"Erreur lors de la communication: {str(e)}")
            return ""

    def save_state(self):
        """Sauvegarde l'état du cerveau."""
        try:
            self.memory_module.save_long_term_memory()
            brain_logger.info("État du cerveau sauvegardé")
        except Exception as e:
            brain_logger.error(f"Erreur lors de la sauvegarde de l'état: {str(e)}")

    def load_state(self):
        """Charge l'état du cerveau."""
        try:
            self.memory_module.load_long_term_memory()
            brain_logger.info("État du cerveau chargé")
        except Exception as e:
            brain_logger.error(f"Erreur lors du chargement de l'état: {str(e)}")
