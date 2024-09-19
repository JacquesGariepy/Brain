from collections import deque
import numpy as np

class Synapse:
    """
    Modèle de synapse avec plasticité dépendante du temps des spikes (STDP).
    
    Attributes:
        pre_neuron (Neuron): Neurone pré-synaptique.
        post_neuron (Neuron): Neurone post-synaptique.
        weight (float): Poids synaptique.
        delay (float): Délai de transmission de l'information entre les neurones.
    """
    
    def __init__(self, pre_neuron, post_neuron, weight=0.5, delay=1.0):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.delay = delay
        self.spike_times = deque()  # File pour gérer les spikes
    
    def transmit_spike(self):
        """Transmet un spike de manière différée."""
        self.spike_times.append(self.delay)

    def get_current(self):
        """Calcule le courant synaptique basé sur le poids synaptique."""
        return self.weight * len(self.spike_times)

    def update_weight_stdp(self, delta_t):
        """
        Met à jour le poids synaptique en fonction de la STDP.
        
        Args:
            delta_t (float): Différence de temps entre les spikes des neurones pré et post-synaptiques.
        """
        tau_plus, tau_minus = 20.0, 20.0
        A_plus, A_minus = 0.01, 0.012

        if delta_t > 0:
            delta_w = A_plus * np.exp(-delta_t / tau_plus)
        else:
            delta_w = -A_minus * np.exp(delta_t / tau_minus)

        self.weight += delta_w
        self.weight = np.clip(self.weight, 0.0, 1.0)
