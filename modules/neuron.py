import numpy as np

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

    def __init__(self, neuron_id, tau_m=20.0, v_rest=-65.0, v_threshold=-50.0, v_reset=-65.0, r_m=1.0):
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
        self.incoming_synapses = []
        self.outgoing_synapses = []
        self.last_spike_time = None  # Temps du dernier spike
        self.current_time = 0.0  # Temps courant de la simulation

    def add_incoming_synapse(self, synapse):
        """Ajoute une synapse entrante."""
        self.incoming_synapses.append(synapse)

    def add_outgoing_synapse(self, synapse):
        """Ajoute une synapse sortante."""
        self.outgoing_synapses.append(synapse)

    def receive_current(self, syn_current):
        """Reçoit le courant total des synapses entrantes."""
        self.input_current = syn_current

    def update(self, dt):
        """
        Met à jour le potentiel membranaire du neurone en fonction du courant synaptique total.

        Args:
            dt (float): Pas de temps de simulation.
        """
        self.current_time += dt
        total_synaptic_current = sum(
            synapse.get_current(self.current_time) for synapse in self.incoming_synapses
        )
        # Inclure le courant émotionnel et le facteur d'attention
        total_current = total_synaptic_current + self.emotion_influence
        dv = dt * ((- (self.v_m - self.v_rest) + self.r_m * self.alpha * total_current) / self.tau_m)
        self.v_m += dv

        # Vérifier si le neurone dépasse le seuil de déclenchement
        if self.v_m >= self.v_threshold:
            self.v_m = self.v_reset
            self.spike = True
            self.last_spike_time = self.current_time
        else:
            self.spike = False

    def reset(self):
        """Réinitialise le potentiel du neurone après un spike."""
        self.v_m = self.v_rest
        self.spike = False
        self.last_spike_time = None
        self.input_current = 0.0
