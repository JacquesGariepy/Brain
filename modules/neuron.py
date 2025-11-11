"""
Modèle de neurone basé sur Leaky Integrate-and-Fire (LIF).

Implémente un neurone biologique réaliste avec:
- Dynamique LIF avec constante de temps membranaire
- Gestion des synapses entrantes et sortantes
- Horodatage des spikes pour STDP
- Modulation par attention et émotions

Conforme aux standards NASA/MIT pour le code scientifique.
"""
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class Neuron:
    """
    Modèle de neurone basé sur Leaky Integrate-and-Fire (LIF) avec gestion des synapses entrantes et sortantes,
    et horodatage des spikes pour la STDP.

    Attributes:
        neuron_id (int): Identifiant unique du neurone.
        tau_m (float): Constante de temps membranaire (ms).
        v_rest (float): Potentiel de repos du neurone (mV).
        v_threshold (float): Seuil de déclenchement du spike (mV).
        v_reset (float): Potentiel membranaire après un spike (mV).
        r_m (float): Résistance membranaire (MΩ).
        alpha (float): Facteur d'attention modulant la réponse neuronale.
        emotion_influence (float): Influence des émotions sur l'excitabilité neuronale.
        incoming_synapses (list): Liste des synapses entrantes.
        outgoing_synapses (list): Liste des synapses sortantes.
        last_spike_time (float): Temps du dernier spike du neurone.
    """

    def __init__(self, neuron_id: int, tau_m: float = 20.0, v_rest: float = -65.0,
                 v_threshold: float = -50.0, v_reset: float = -65.0, r_m: float = 1.0):
        """
        Initialise un neurone LIF.

        Args:
            neuron_id: Identifiant unique du neurone
            tau_m: Constante de temps membranaire (ms)
            v_rest: Potentiel de repos (mV)
            v_threshold: Seuil de spike (mV)
            v_reset: Potentiel après spike (mV)
            r_m: Résistance membranaire (MΩ)
        """
        self.neuron_id = neuron_id
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.r_m = r_m
        self.v_m = v_rest  # Potentiel membranaire initial
        self.spike = False
        self.alpha = 1.0  # Facteur d'attention initial
        self.emotion_influence = 0.0  # Influence émotionnelle sur le neurone
        self.incoming_synapses: List = []
        self.outgoing_synapses: List = []
        self.last_spike_time: Optional[float] = None  # Temps du dernier spike
        self.current_time = 0.0  # Temps courant de la simulation
        self.input_current = 0.0  # Courant d'entrée total
        logger.debug(f"Neurone {neuron_id} créé")

    def add_incoming_synapse(self, synapse):
        """Ajoute une synapse entrante."""
        self.incoming_synapses.append(synapse)
        logger.debug(f"Synapse entrante ajoutée au neurone {self.neuron_id}")

    def add_outgoing_synapse(self, synapse):
        """Ajoute une synapse sortante."""
        self.outgoing_synapses.append(synapse)
        logger.debug(f"Synapse sortante ajoutée au neurone {self.neuron_id}")

    def receive_current(self, syn_current: float):
        """
        Reçoit le courant total des synapses entrantes.

        Args:
            syn_current: Courant synaptique total
        """
        self.input_current += syn_current

    def update(self, dt: float):
        """
        Met à jour le potentiel membranaire du neurone en fonction du courant synaptique total.

        Args:
            dt: Pas de temps de simulation (ms)
        """
        self.current_time += dt
        total_synaptic_current = sum(
            synapse.get_current(self.current_time) for synapse in self.incoming_synapses
        )
        # Inclure le courant émotionnel et le facteur d'attention
        total_current = total_synaptic_current + self.emotion_influence

        # Équation différentielle LIF
        dv = dt * ((- (self.v_m - self.v_rest) + self.r_m * self.alpha * total_current) / self.tau_m)
        self.v_m += dv

        # Vérifier si le neurone dépasse le seuil de déclenchement
        if self.v_m >= self.v_threshold:
            self.v_m = self.v_reset
            self.spike = True
            self.last_spike_time = self.current_time
            logger.debug(f"Neurone {self.neuron_id} a spiké au temps {self.current_time:.2f}")
        else:
            self.spike = False

    def reset(self):
        """Réinitialise le potentiel du neurone après un spike."""
        self.v_m = self.v_rest
        self.spike = False
        self.last_spike_time = None
        self.input_current = 0.0
        logger.debug(f"Neurone {self.neuron_id} réinitialisé")

    def reset_current(self):
        """Réinitialise le courant d'entrée pour le prochain pas de temps."""
        self.input_current = 0.0

    def update_potential(self, input_value: float, dt: float):
        """
        Met à jour le potentiel membranaire avec une valeur d'entrée directe.

        Cette méthode est utilisée pour l'apprentissage supervisé où on applique
        directement des stimuli aux neurones.

        Args:
            input_value: Valeur d'entrée à appliquer
            dt: Pas de temps de simulation
        """
        # Appliquer l'entrée comme un courant
        current = input_value * 10.0  # Mise à l'échelle

        # Mise à jour LIF avec le courant
        dv = dt * ((- (self.v_m - self.v_rest) + self.r_m * current) / self.tau_m)
        self.v_m += dv

        # Vérifier le seuil
        if self.v_m >= self.v_threshold:
            self.v_m = self.v_reset
            self.spike = True
            self.last_spike_time = self.current_time
        else:
            self.spike = False

    def set_attention(self, alpha: float):
        """
        Définit le facteur d'attention du neurone.

        Args:
            alpha: Facteur d'attention (typiquement entre 0.5 et 2.0)
        """
        self.alpha = max(0.0, alpha)
        logger.debug(f"Neurone {self.neuron_id}: attention={self.alpha:.2f}")

    def set_emotion_influence(self, influence: float):
        """
        Définit l'influence émotionnelle sur le neurone.

        Args:
            influence: Influence émotionnelle (peut être positive ou négative)
        """
        self.emotion_influence = influence
        logger.debug(f"Neurone {self.neuron_id}: emotion_influence={self.emotion_influence:.2f}")

    def get_state(self) -> dict:
        """
        Retourne l'état actuel du neurone.

        Returns:
            Dictionnaire contenant l'état du neurone
        """
        return {
            'neuron_id': self.neuron_id,
            'v_m': self.v_m,
            'spike': self.spike,
            'alpha': self.alpha,
            'emotion_influence': self.emotion_influence,
            'last_spike_time': self.last_spike_time,
            'current_time': self.current_time
        }
