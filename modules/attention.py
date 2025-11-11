"""
Module d'attention avec modèle computationnel réel.

Implémente:
- Saliency map (carte de saillance)
- Winner-take-all competition
- Dynamiques temporelles
- Attention bottom-up (stimulus-driven) et top-down (goal-driven)

Conforme aux modèles d'attention visuelle (Itti & Koch, 2000)
et d'attention cognitive (Desimone & Duncan, 1995).
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AttentionModule:
    """
    Module d'attention qui ajuste le facteur alpha des neurones en fonction
    de la saillance des stimuli et des objectifs cognitifs.

    Attributes:
        neurons (list): Liste des neurones impliqués dans le réseau.
        attention_map (np.ndarray): Carte d'attention actuelle (0-1 pour chaque neurone).
        tau_attention (float): Constante de temps pour la dynamique d'attention (ms).
        competition_strength (float): Force de la compétition winner-take-all (0-1).
        top_down_bias (dict): Biais top-down pour des neurones spécifiques.
    """

    def __init__(self, neurons, tau_attention: float = 50.0, competition_strength: float = 0.5):
        """
        Initialise le module d'attention.

        Args:
            neurons: Liste des neurones du réseau
            tau_attention: Constante de temps pour la dynamique d'attention (ms)
            competition_strength: Force de la compétition (0=pas de compétition, 1=winner-take-all strict)
        """
        self.neurons = neurons
        self.tau_attention = tau_attention
        self.competition_strength = competition_strength

        # Initialiser la carte d'attention (valeurs entre 0 et 1)
        self.attention_map = np.ones(len(neurons))

        # Biais top-down (objectifs cognitifs)
        self.top_down_bias = {}

        logger.info(f"AttentionModule initialisé: {len(neurons)} neurones, tau={tau_attention}ms")

    def compute_saliency(self, activation_levels: np.ndarray) -> np.ndarray:
        """
        Calcule la carte de saillance basée sur les niveaux d'activation.

        La saillance mesure à quel point un stimulus "se démarque" de son contexte.
        Utilise un modèle de center-surround (différence avec la moyenne locale).

        Args:
            activation_levels: Niveaux d'activation des neurones (potentiels ou taux de spike)

        Returns:
            Carte de saillance (valeurs entre 0 et 1)
        """
        if len(activation_levels) == 0:
            return np.zeros(len(self.neurons))

        activation_levels = np.asarray(activation_levels)

        # Normaliser les activations
        if np.max(np.abs(activation_levels)) > 0:
            normalized = (activation_levels - np.mean(activation_levels)) / (np.std(activation_levels) + 1e-9)
        else:
            normalized = activation_levels

        # Center-surround: différence avec la moyenne
        mean_activation = np.mean(activation_levels)
        saliency = np.abs(activation_levels - mean_activation)

        # Normaliser entre 0 et 1
        if np.max(saliency) > 0:
            saliency = saliency / np.max(saliency)

        return saliency

    def apply_competition(self, saliency_map: np.ndarray) -> np.ndarray:
        """
        Applique une compétition winner-take-all sur la carte de saillance.

        Les neurones avec forte saillance inhibent ceux avec faible saillance.

        Args:
            saliency_map: Carte de saillance d'entrée

        Returns:
            Carte de saillance après compétition
        """
        if self.competition_strength == 0:
            return saliency_map

        # Winner-take-all avec softmax
        # Plus competition_strength est élevé, plus le winner gagne
        sharpness = 10.0 * self.competition_strength
        exp_saliency = np.exp(sharpness * saliency_map)
        competitive_saliency = exp_saliency / (np.sum(exp_saliency) + 1e-9)

        # Normaliser pour avoir des valeurs entre 0 et 1
        if np.max(competitive_saliency) > 0:
            competitive_saliency = competitive_saliency / np.max(competitive_saliency)

        return competitive_saliency

    def apply_top_down_bias(self, bottom_up_attention: np.ndarray) -> np.ndarray:
        """
        Applique les biais top-down (dirigés par les objectifs) à l'attention bottom-up.

        Args:
            bottom_up_attention: Attention stimulus-driven

        Returns:
            Attention combinée (bottom-up + top-down)
        """
        combined_attention = bottom_up_attention.copy()

        # Appliquer les biais top-down pour des neurones spécifiques
        for neuron_id, bias_value in self.top_down_bias.items():
            if neuron_id < len(combined_attention):
                # Combiner multiplicativement (0.5 bottom-up + 0.5 top-down)
                combined_attention[neuron_id] = 0.5 * combined_attention[neuron_id] + 0.5 * bias_value

        return combined_attention

    def update_attention_dynamics(self, target_attention: np.ndarray, dt: float):
        """
        Met à jour la dynamique temporelle de l'attention avec un modèle différentiel.

        L'attention change graduellement vers la cible, pas instantanément.

        Args:
            target_attention: Attention cible
            dt: Pas de temps (ms)
        """
        # Équation différentielle: dA/dt = (-A + target) / tau
        dA = dt * (target_attention - self.attention_map) / self.tau_attention
        self.attention_map += dA

        # Clipper entre 0 et 1
        self.attention_map = np.clip(self.attention_map, 0.0, 1.0)

    def update_attention(self, relevance_signal=None, dt: float = 1.0):
        """
        Met à jour les facteurs d'attention des neurones.

        Cette version utilise un modèle complet:
        1. Calcule la saillance basée sur l'activité neuronale
        2. Applique la compétition winner-take-all
        3. Intègre les biais top-down
        4. Met à jour avec dynamique temporelle
        5. Applique aux neurones

        Args:
            relevance_signal (dict, optional): Signaux de pertinence externe (overrides).
            dt (float): Pas de temps pour la dynamique temporelle (ms).
        """
        # 1. Extraire les niveaux d'activation des neurones
        activation_levels = np.array([neuron.v_m for neuron in self.neurons])

        # 2. Calculer la saillance (bottom-up)
        saliency_map = self.compute_saliency(activation_levels)

        # 3. Appliquer la compétition
        competitive_attention = self.apply_competition(saliency_map)

        # 4. Appliquer les biais top-down
        combined_attention = self.apply_top_down_bias(competitive_attention)

        # 5. Override avec relevance_signal si fourni
        if relevance_signal is not None:
            for neuron_id, relevance in relevance_signal.items():
                if neuron_id < len(combined_attention):
                    combined_attention[neuron_id] = relevance

        # 6. Mettre à jour la dynamique temporelle
        self.update_attention_dynamics(combined_attention, dt)

        # 7. Appliquer aux neurones (alpha = 1.0 + attention_boost)
        for i, neuron in enumerate(self.neurons):
            # Alpha modulé entre 0.5 (pas d'attention) et 2.0 (attention maximale)
            neuron.alpha = 0.5 + 1.5 * self.attention_map[i]

        # Logging détaillé
        num_attended = np.sum(self.attention_map > 0.7)
        logger.debug(f"Attention: {num_attended}/{len(self.neurons)} neurones fortement attendus")

    def set_top_down_bias(self, neuron_id: int, bias_value: float):
        """
        Définit un biais top-down pour un neurone spécifique.

        Args:
            neuron_id: ID du neurone
            bias_value: Valeur du biais (0-1)
        """
        self.top_down_bias[neuron_id] = np.clip(bias_value, 0.0, 1.0)
        logger.debug(f"Biais top-down défini: neurone {neuron_id} -> {bias_value:.2f}")

    def clear_top_down_bias(self):
        """Efface tous les biais top-down."""
        self.top_down_bias.clear()
        logger.debug("Biais top-down effacés")

    def get_attention_state(self) -> dict:
        """
        Retourne l'état actuel de l'attention.

        Returns:
            Dictionnaire avec les métriques d'attention
        """
        return {
            'attention_map': self.attention_map.copy(),
            'mean_attention': np.mean(self.attention_map),
            'max_attention': np.max(self.attention_map),
            'num_top_down_biases': len(self.top_down_bias),
            'attended_neurons': np.where(self.attention_map > 0.7)[0].tolist()
        }

    def set_competition_strength(self, strength: float):
        """
        Modifie la force de la compétition winner-take-all.

        Args:
            strength: Force de compétition (0-1)
        """
        self.competition_strength = np.clip(strength, 0.0, 1.0)
        logger.info(f"Force de compétition modifiée: {self.competition_strength:.2f}")
