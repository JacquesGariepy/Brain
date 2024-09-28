import numpy as np
from collections import deque

class Synapse:
    """
    Modèle de synapse avec plasticité dynamique avancée, incluant la plasticité à court terme,
    la STDP, la plasticité homéostatique, et la modulation astrocytaire.
    
    Attributes:
        pre_neuron (Neuron): Neurone pré-synaptique.
        post_neuron (Neuron): Neurone post-synaptique.
        weight (float): Poids synaptique.
        delay (float): Délai de transmission de l'information entre les neurones.
        x (float): Variable d'efficacité synaptique pour la plasticité à court terme.
        u (float): Facteur d'utilisation des ressources synaptiques pour la plasticité à court terme.
        last_pre_spike_time (float): Temps du dernier spike du neurone pré-synaptique.
        last_post_spike_time (float): Temps du dernier spike du neurone post-synaptique.
        astro_ca (float): Concentration de calcium de l'astrocyte pour la modulation.
    """
    
    def __init__(self, pre_neuron, post_neuron, weight=0.5, delay=1.0, config=None):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.delay = delay
        self.spike_times = deque()
        
        # Variables pour la plasticité à court terme
        self.x = 1.0  # Efficacité synaptique initiale
        self.u = 0.2  # Facteur d'utilisation initial
        self.U = 0.2  # Paramètre U pour la plasticité à court terme
        self.A = 1.0  # Facteur de mise à l'échelle de l'efficacité synaptique
        self.tau_p = 200.0  # Constante de temps pour la plasticité à court terme
        
        # Paramètres STDP
        self.A_plus = 0.01
        self.A_minus = 0.012
        self.tau_plus = 20.0
        self.tau_minus = 20.0
        self.last_pre_spike_time = None
        self.last_post_spike_time = None
        
        # Plasticité homéostatique
        self.target_rate = 0.1  # Taux de firing cible
        self.alpha = 0.001  # Taux d'apprentissage pour la plasticité homéostatique
        
        # Modulation astrocytaire
        self.astro_ca = 0.0  # Concentration de calcium de l'astrocyte
        self.tau_astro = 1000.0  # Constante de temps pour la dynamique de l'astrocyte
        
        # Configuration personnalisée si fournie
        if config:
            self.U = config.get('U', self.U)
            self.A = config.get('A', self.A)
            self.tau_p = config.get('tau_p', self.tau_p)
            self.A_plus = config.get('A_plus', self.A_plus)
            self.A_minus = config.get('A_minus', self.A_minus)
            self.tau_plus = config.get('tau_plus', self.tau_plus)
            self.tau_minus = config.get('tau_minus', self.tau_minus)
            self.target_rate = config.get('target_rate', self.target_rate)
            self.alpha = config.get('alpha', self.alpha)
            self.tau_astro = config.get('tau_astro', self.tau_astro)
        
    def transmit_spike(self, current_time):
        """Transmet un spike après le délai spécifié."""
        self.spike_times.append(current_time + self.delay)
        # Met à jour le temps du dernier spike pré-synaptique
        self.last_pre_spike_time = current_time
        
        # Met à jour les variables de plasticité à court terme
        self.update_short_term_plasticity()
    
    def receive_spike(self, current_time):
        """Gère la réception d'un spike par le neurone post-synaptique."""
        self.last_post_spike_time = current_time
        # Met à jour le poids synaptique avec la STDP
        self.update_weight_stdp()
        # Met à jour la plasticité homéostatique
        self.update_homeostatic_plasticity()
        # Met à jour la modulation astrocytaire
        self.update_astrocyte_modulation()
    
    def update_short_term_plasticity(self):
        """Met à jour l'efficacité synaptique et le facteur d'utilisation."""
        dt = self.delay  # Supposons que le spike vient de se produire
        # Met à jour u
        du = (self.U - self.u) / self.tau_p + self.U * (1 - self.u)
        self.u += du * dt
        # Met à jour x
        dx = (1 - self.x) / self.tau_p - self.u * self.x
        self.x += dx * dt
        # Assure que les variables sont dans [0,1]
        self.u = np.clip(self.u, 0.0, 1.0)
        self.x = np.clip(self.x, 0.0, 1.0)
        
    def get_current(self, current_time):
        """Calcule le courant synaptique basé sur le poids et les temps de spike."""
        # Supprime les spikes expirés
        while self.spike_times and self.spike_times[0] <= current_time:
            self.spike_times.popleft()
            # L'effet de la plasticité à court terme est déjà pris en compte
        
        # Calcule le courant synaptique
        current = self.weight * self.A * self.x * self.u * len(self.spike_times)
        # Applique la modulation astrocytaire
        current *= (1 + 0.1 * self.astro_ca)
        return current
    
    def update_weight_stdp(self):
        """Met à jour le poids synaptique en utilisant la STDP."""
        if self.last_pre_spike_time is not None and self.last_post_spike_time is not None:
            delta_t = self.last_post_spike_time - self.last_pre_spike_time
            if delta_t > 0:
                delta_w = self.A_plus * np.exp(-delta_t / self.tau_plus)
            else:
                delta_w = -self.A_minus * np.exp(delta_t / self.tau_minus)
            self.weight += delta_w
            self.weight = np.clip(self.weight, 0.0, 1.0)
    
    def update_homeostatic_plasticity(self):
        """Ajuste le poids synaptique pour maintenir des taux de firing stables."""
        # Pour simplifier, supposons que le taux est proportionnel à l'activité récente
        rate = 1.0 / (self.last_post_spike_time - self.last_pre_spike_time + 1e-9)
        delta_w = self.alpha * (self.target_rate - rate)
        self.weight += delta_w
        self.weight = np.clip(self.weight, 0.0, 1.0)
    
    def update_astrocyte_modulation(self):
        """Met à jour la concentration de calcium de l'astrocyte."""
        # Modèle simple : la concentration de calcium augmente avec l'activité
        dca = (-self.astro_ca + 1.0) / self.tau_astro
        self.astro_ca += dca
        self.astro_ca = np.clip(self.astro_ca, 0.0, 1.0)
