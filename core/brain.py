"""
Module Brain principal - Orchestration de tous les modules.

Ce module coordonne tous les composants du cerveau artificiel:
- Réseau neuronal
- Mémoire (court et long terme)
- Apprentissage (supervisé, non supervisé, renforcement)
- Émotions
- Attention
- Décision
- Langage
- Perception

Conforme aux standards NASA/MIT pour le code scientifique.
"""
import importlib
import os
from typing import Dict, List, Optional
import logging
from .interfaces import BrainModule
from modules.attention import AttentionModule
from modules.decision import DecisionModule
from modules.emotion import EmotionModule
from modules.learning import LearningModule
from modules.memory import MemoryModule
from modules.network import Network
from modules.neuron import Neuron
from modules.synapse import Synapse

logger = logging.getLogger(__name__)


class Brain:
    """
    Classe principale du cerveau artificiel.

    Intègre tous les modules et gère leur interaction.
    """

    def __init__(self, num_neurons: int = 10):
        """
        Initialise le cerveau avec tous ses modules.

        Args:
            num_neurons: Nombre de neurones à créer dans le réseau
        """
        logger.info(f"=== Initialisation du Brain avec {num_neurons} neurones ===")

        # 1. Créer les modules de base DANS LE BON ORDRE
        self.network = Network()
        self.memory_module = MemoryModule()

        # 2. Créer les modules qui en dépendent
        self.learning_module = LearningModule(self.network, self.memory_module)
        self.emotion_module = EmotionModule()
        self.decision_module = DecisionModule()

        # 3. Créer les neurones et synapses
        self.neurons: List[Neuron] = []
        self.synapses: List[Synapse] = []
        self.create_neurons_and_synapses(num_neurons)

        # 4. Créer le module d'attention (nécessite la liste de neurones)
        self.attention_module = AttentionModule(self.neurons)

        # 5. Charger les modules core et plugins
        self.modules: Dict[str, BrainModule] = {}
        self.load_core_modules()
        self.load_plugins()

        logger.info("Brain initialisé avec succès")

    def load_core_modules(self):
        """Charge les modules core du cerveau."""
        try:
            from .perception import PerceptionModule
            from .language import LanguageModule
            from .reasoning import ReasoningModule

            self.modules['perception'] = PerceptionModule()
            self.modules['language'] = LanguageModule()
            self.modules['reasoning'] = ReasoningModule()
            logger.info("Modules core chargés: perception, language, reasoning")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modules core: {e}")

    def load_plugins(self):
        """Charge les plugins depuis le dossier plugins/."""
        plugin_folder = 'plugins'
        if not os.path.exists(plugin_folder):
            logger.warning(f"Dossier {plugin_folder} introuvable")
            return

        for filename in os.listdir(plugin_folder):
            if filename.endswith('.py') and not filename.startswith('__'):
                try:
                    module_name = filename[:-3]
                    module = importlib.import_module(f'plugins.{module_name}')
                    if hasattr(module, 'Plugin'):
                        plugin_class = getattr(module, 'Plugin')
                        plugin_instance = plugin_class()
                        self.modules[module_name] = plugin_instance
                        logger.info(f'Plugin chargé : {module_name}')
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du plugin {filename}: {e}")

    def create_neurons_and_synapses(self, num_neurons: int):
        """
        Crée les neurones et les synapses pour le réseau neuronal.

        Args:
            num_neurons: Nombre de neurones à créer
        """
        logger.info(f"Création de {num_neurons} neurones...")

        # Créer les neurones
        for i in range(num_neurons):
            neuron = Neuron(neuron_id=i)
            self.neurons.append(neuron)
            self.network.add_neuron(neuron)

        # Connecter les neurones (réseau densément connecté)
        logger.info("Connexion des neurones...")
        for pre_neuron in self.neurons:
            for post_neuron in self.neurons:
                if pre_neuron != post_neuron:
                    synapse = Synapse(pre_neuron, post_neuron)
                    self.synapses.append(synapse)

        logger.info(f"{len(self.neurons)} neurones et {len(self.synapses)} synapses créés")

    def process(self, data):
        """
        Flux de traitement de base à travers tous les modules.

        Args:
            data: Données d'entrée à traiter

        Returns:
            Données traitées par tous les modules
        """
        # Flux de traitement core
        if 'perception' in self.modules:
            data = self.modules['perception'].process(data)
        if 'language' in self.modules:
            data = self.modules['language'].process(data)
        if 'reasoning' in self.modules:
            data = self.modules['reasoning'].process(data)

        # Traitement avec les plugins
        for module_name, module in self.modules.items():
            if module_name not in ['perception', 'language', 'reasoning']:
                data = module.process(data)

        return data

    def perceive_and_process(self, sensory_input: List[float], dt: float):
        """
        Perçoit et traite les entrées sensorielles.

        Args:
            sensory_input: Liste des entrées sensorielles
            dt: Pas de temps de simulation
        """
        # Stocker en mémoire court terme
        self.memory_module.store_short_term(sensory_input)

        # Mettre à jour le réseau neuronal
        self.network.update(dt)

        # Mettre à jour les émotions
        self.emotion_module.update_emotions(
            sensory_input,
            self.memory_module.retrieve_short_term(),
            0,
            dt
        )

        # Mettre à jour l'attention
        relevance = {neuron.neuron_id: 1.0 for neuron in self.neurons}
        self.attention_module.update_attention(relevance)

        # Appliquer l'influence émotionnelle
        self.emotion_module.influence_on_neurons(self.neurons)

    def execute_decision(self, dt: float, evidence: float = 0.5):
        """
        Exécute une décision basée sur l'accumulation d'évidence.

        Args:
            dt: Pas de temps de simulation
            evidence: Évidence pour la décision
        """
        emotion_influence = self.emotion_module.emotional_states.get("fear", 0)
        self.decision_module.update_decision(evidence, emotion_influence, dt)

        if self.decision_module.choice_made:
            logger.info(f"Décision prise : {self.decision_module.decision}")
            self.decision_module.reset()

    def learn(self, inputs: List[float], targets: List[float], learning_type: str = 'supervised'):
        """
        Apprend à partir des entrées et des cibles.

        Args:
            inputs: Entrées du réseau
            targets: Sorties attendues
            learning_type: Type d'apprentissage ('supervised', 'unsupervised', 'reinforcement')

        Returns:
            Pour supervised: erreur moyenne (float)
            Pour unsupervised: labels des clusters (np.ndarray)
            Pour reinforcement: None
        """
        if learning_type == 'supervised':
            error = self.learning_module.supervised_learning(inputs, targets)
            logger.info(f"Apprentissage supervisé effectué, erreur={error:.4f}")
            return error
        elif learning_type == 'unsupervised':
            clusters = self.learning_module.unsupervised_learning(inputs)
            logger.info(f"Apprentissage non supervisé effectué, {len(set(clusters))} clusters")
            return clusters
        elif learning_type == 'reinforcement':
            # Pour le renforcement, targets contient la récompense
            reward = targets[0] if len(targets) > 0 else 0.0
            self.learning_module.reinforcement_learning(reward)
            logger.info(f"Apprentissage par renforcement effectué, reward={reward:.4f}")
            return None

    def save_state(self):
        """Sauvegarde l'état du cerveau."""
        self.memory_module.save_long_term_memory()
        logger.info("État du cerveau sauvegardé")

    def load_state(self):
        """Charge l'état du cerveau."""
        self.memory_module.load_long_term_memory()
        logger.info("État du cerveau chargé")

    def get_status(self) -> dict:
        """
        Retourne le statut complet du cerveau.

        Returns:
            Dictionnaire avec toutes les métriques
        """
        network_activity = self.network.get_activity()

        return {
            'network': network_activity,
            'emotions': self.emotion_module.emotional_states.copy(),
            'decision': self.decision_module.get_state(),
            'memory_short_term': len(self.memory_module.retrieve_short_term()),
            'modules_loaded': list(self.modules.keys())
        }
