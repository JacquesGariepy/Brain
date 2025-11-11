"""
Module d'apprentissage pour le réseau neuronal.

Ce module implémente trois types d'apprentissage:
1. Apprentissage supervisé avec rétropropagation
2. Apprentissage non supervisé par clustering
3. Apprentissage par renforcement

Conforme aux standards NASA/MIT pour le code scientifique.
"""
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class LearningModule:
    """
    Module d'apprentissage supervisé, non supervisé et par renforcement pour le réseau neuronal.

    Attributes:
        network: Réseau neuronal sur lequel effectuer l'apprentissage
        memory: Module de mémoire pour stocker les résultats d'apprentissage
        learning_rate: Taux d'apprentissage par défaut

    Methods:
        supervised_learning: Apprentissage avec des exemples étiquetés.
        unsupervised_learning: Apprentissage basé sur le regroupement (clustering).
        reinforcement_learning: Apprentissage basé sur les récompenses.
    """

    def __init__(self, network, memory_module, learning_rate: float = 0.01):
        """
        Initialise le module d'apprentissage.

        Args:
            network: Réseau neuronal
            memory_module: Module de mémoire
            learning_rate: Taux d'apprentissage initial (default: 0.01)
        """
        self.network = network
        self.memory = memory_module
        self.learning_rate = learning_rate
        self.training_history = []
        logger.info(f"LearningModule initialisé avec learning_rate={learning_rate}")

    def supervised_learning(self, inputs: np.ndarray, targets: np.ndarray,
                          learning_rate: Optional[float] = None) -> float:
        """
        Effectue un apprentissage supervisé en ajustant les poids synaptiques en fonction des erreurs.

        Args:
            inputs: Entrées du réseau (array-like)
            targets: Sorties attendues (array-like)
            learning_rate: Taux d'apprentissage (utilise self.learning_rate si None)

        Returns:
            float: Erreur moyenne après l'apprentissage
        """
        if learning_rate is None:
            learning_rate = self.learning_rate

        inputs = np.asarray(inputs)
        targets = np.asarray(targets)

        # Propagation avant
        outputs = self.forward_pass(inputs)

        # Calcul de l'erreur
        errors = targets - outputs
        mean_error = np.mean(np.abs(errors))

        # Rétropropagation
        self.backward_pass(errors, learning_rate)

        # Enregistrement dans l'historique
        self.training_history.append({
            'mean_error': float(mean_error),
            'learning_rate': learning_rate
        })

        logger.debug(f"Supervised learning: mean_error={mean_error:.4f}")
        return mean_error

    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """
        Propagation avant des entrées à travers le réseau.

        Args:
            inputs: Entrées du réseau

        Returns:
            np.array: Sorties calculées (activations des neurones)
        """
        inputs = np.asarray(inputs)
        outputs = []

        # Réinitialiser tous les neurones
        for neuron in self.network.neurons:
            neuron.reset()

        # Appliquer les entrées aux neurones correspondants
        num_inputs = min(len(inputs), len(self.network.neurons))
        for i in range(num_inputs):
            neuron = self.network.neurons[i]
            # Stimuler le neurone avec l'entrée
            neuron.v_m = neuron.v_rest + inputs[i] * 10.0  # Mise à l'échelle
            # Vérifier si le neurone spike
            if neuron.v_m >= neuron.v_threshold:
                neuron.spike = True
                neuron.last_spike_time = 0.0
                outputs.append(1.0)
            else:
                neuron.spike = False
                outputs.append(0.0)

        # Compléter avec des zéros si nécessaire
        while len(outputs) < len(self.network.neurons):
            outputs.append(0.0)

        return np.array(outputs)

    def backward_pass(self, errors: np.ndarray, learning_rate: float):
        """
        Rétropropagation de l'erreur pour ajuster les poids synaptiques.

        Args:
            errors: Erreurs observées entre les sorties réelles et attendues
            learning_rate: Taux d'apprentissage
        """
        errors = np.asarray(errors)

        # Ajuster les poids synaptiques en fonction des erreurs
        for synapse in self.network.synapses:
            post_idx = synapse.post_neuron.neuron_id

            # Vérifier que l'index est valide
            if post_idx < len(errors):
                # Calcul du gradient
                delta_w = learning_rate * errors[post_idx] * synapse.pre_neuron.v_m

                # Mise à jour du poids
                synapse.weight += delta_w
                synapse.weight = np.clip(synapse.weight, 0.0, 1.0)

    def unsupervised_learning(self, inputs: np.ndarray, num_clusters: int = 3) -> np.ndarray:
        """
        Effectue un apprentissage non supervisé basé sur le regroupement des neurones en clusters.

        Args:
            inputs: Données d'entrée pour l'apprentissage non supervisé
            num_clusters: Nombre de clusters à utiliser pour l'algorithme de k-moyennes

        Returns:
            np.ndarray: Labels des clusters pour chaque neurone
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.error("scikit-learn n'est pas installé. Impossible d'effectuer l'apprentissage non supervisé.")
            return np.zeros(len(self.network.neurons))

        inputs = np.asarray(inputs).reshape(-1, 1)

        # Clustering avec K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(inputs)
        clusters = kmeans.predict(inputs)

        # Ajuster les poids synaptiques en fonction des clusters
        for synapse in self.network.synapses:
            pre_id = synapse.pre_neuron.neuron_id
            post_id = synapse.post_neuron.neuron_id

            # Vérifier que les indices sont valides
            if pre_id < len(clusters) and post_id < len(clusters):
                if clusters[pre_id] == clusters[post_id]:
                    # Renforcer les connexions intra-cluster
                    synapse.weight += 0.01
                else:
                    # Affaiblir les connexions inter-cluster
                    synapse.weight -= 0.01

                # Maintenir les poids dans [0, 1]
                synapse.weight = np.clip(synapse.weight, 0.0, 1.0)

        logger.info(f"Unsupervised learning: {num_clusters} clusters créés")
        return clusters

    def reinforcement_learning(self, reward: float, gamma: float = 0.9):
        """
        Effectue un apprentissage par renforcement basé sur les récompenses reçues.

        Args:
            reward: Récompense reçue pour renforcer ou punir un comportement
            gamma: Facteur d'atténuation pour l'apprentissage par renforcement (0-1)
        """
        # Calculer le signal d'erreur de prédiction de récompense (TD error)
        delta = reward  # Simplifié: dans une implémentation complète, ce serait reward + gamma * V(s') - V(s)

        # Mettre à jour les poids synaptiques
        for synapse in self.network.synapses:
            # Appliquer la règle de mise à jour par renforcement
            # Les synapses actives récemment sont renforcées/affaiblies par le reward
            if synapse.pre_neuron.spike or synapse.post_neuron.spike:
                synapse.weight += self.learning_rate * delta
                synapse.weight = np.clip(synapse.weight, 0.0, 1.0)

        logger.debug(f"Reinforcement learning: reward={reward:.4f}, delta={delta:.4f}")

    def get_training_history(self) -> List[dict]:
        """
        Retourne l'historique d'apprentissage.

        Returns:
            Liste des enregistrements d'apprentissage
        """
        return self.training_history

    def reset_training_history(self):
        """Réinitialise l'historique d'apprentissage."""
        self.training_history = []
        logger.info("Historique d'apprentissage réinitialisé")
