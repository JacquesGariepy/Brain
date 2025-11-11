"""
Modèle de réseau neuronal regroupant neurones et synapses.

Gère la dynamique globale du réseau incluant:
- Ajout et connexion de neurones
- Propagation des spikes
- Mise à jour temporelle du réseau

Conforme aux standards NASA/MIT pour le code scientifique.
"""
from typing import List, Optional
import logging
from modules.synapse import Synapse

logger = logging.getLogger(__name__)


class Network:
    """
    Modèle du réseau neuronal, regroupant les neurones et les synapses.

    Attributes:
        neurons: Liste des neurones du réseau
        synapses: Liste des synapses du réseau
        current_time: Temps courant de la simulation (ms)

    Methods:
        add_neuron: Ajoute un neurone au réseau.
        connect_neurons: Crée une synapse entre deux neurones.
        update: Met à jour le réseau (neurones et synapses).
    """

    def __init__(self):
        """Initialise un réseau neuronal vide."""
        self.neurons: List = []
        self.synapses: List = []
        self.current_time = 0.0  # Temps courant de la simulation
        logger.info("Network créé")

    def add_neuron(self, neuron):
        """
        Ajoute un neurone au réseau.

        Args:
            neuron: Neurone à ajouter
        """
        self.neurons.append(neuron)
        logger.debug(f"Neurone {neuron.neuron_id} ajouté au réseau (total: {len(self.neurons)})")

    def connect_neurons(self, pre_neuron, post_neuron,
                       weight: float = 0.5, delay: float = 1.0, config: Optional[dict] = None):
        """
        Crée une synapse entre deux neurones.

        Args:
            pre_neuron: Neurone pré-synaptique
            post_neuron: Neurone post-synaptique
            weight: Poids synaptique initial (default: 0.5)
            delay: Délai de transmission (ms, default: 1.0)
            config: Configuration optionnelle de la synapse
        """
        synapse = Synapse(pre_neuron, post_neuron, weight, delay, config)
        self.synapses.append(synapse)
        pre_neuron.add_outgoing_synapse(synapse)
        post_neuron.add_incoming_synapse(synapse)
        logger.debug(f"Connexion créée: Neurone {pre_neuron.neuron_id} -> Neurone {post_neuron.neuron_id}")

    def update(self, dt: float):
        """
        Met à jour le réseau (neurones et synapses).

        Args:
            dt: Pas de temps de simulation (ms)
        """
        self.current_time += dt

        # Réinitialiser les courants des neurones
        for neuron in self.neurons:
            neuron.reset_current()

        # Mettre à jour les neurones
        for neuron in self.neurons:
            neuron.update(dt)

        # Transmettre les spikes des neurones qui ont spiké
        for neuron in self.neurons:
            if neuron.spike:
                # Transmettre le spike via les synapses sortantes
                for synapse in neuron.outgoing_synapses:
                    synapse.transmit_spike(self.current_time)

        # Mettre à jour les synapses
        for synapse in self.synapses:
            # Mettre à jour la modulation astrocytaire
            synapse.update_astrocyte_modulation()
            # Obtenir le courant synaptique
            syn_current = synapse.get_current(self.current_time)
            # Le neurone post-synaptique reçoit le courant
            synapse.post_neuron.receive_current(syn_current)
            # Si le neurone post-synaptique a spiké, on met à jour la STDP
            if synapse.post_neuron.spike:
                synapse.receive_spike(self.current_time)

    def get_activity(self) -> dict:
        """
        Retourne l'activité actuelle du réseau.

        Returns:
            Dictionnaire avec les statistiques d'activité
        """
        num_spikes = sum(1 for neuron in self.neurons if neuron.spike)
        avg_potential = sum(neuron.v_m for neuron in self.neurons) / len(self.neurons) if self.neurons else 0.0

        return {
            'time': self.current_time,
            'num_neurons': len(self.neurons),
            'num_synapses': len(self.synapses),
            'num_spikes': num_spikes,
            'avg_potential': avg_potential,
            'firing_rate': num_spikes / len(self.neurons) if self.neurons else 0.0
        }

    def reset(self):
        """Réinitialise le réseau neuronal."""
        self.current_time = 0.0
        for neuron in self.neurons:
            neuron.reset()
        logger.info("Network réinitialisé")

    def get_weights(self) -> List[float]:
        """
        Retourne tous les poids synaptiques du réseau.

        Returns:
            Liste des poids synaptiques
        """
        return [synapse.weight for synapse in self.synapses]

    def set_weights(self, weights: List[float]):
        """
        Définit les poids synaptiques du réseau.

        Args:
            weights: Liste des nouveaux poids

        Raises:
            ValueError: Si le nombre de poids ne correspond pas
        """
        if len(weights) != len(self.synapses):
            raise ValueError(f"Nombre de poids ({len(weights)}) != nombre de synapses ({len(self.synapses)})")

        for synapse, weight in zip(self.synapses, weights):
            synapse.weight = max(0.0, min(1.0, weight))
        logger.info(f"Poids synaptiques mis à jour ({len(weights)} synapses)")
