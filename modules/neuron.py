import numpy as np

class Neuron:
    """
    Modèle de neurone basé sur Leaky Integrate-and-Fire (LIF).
    
    Attributes:
        neuron_id (int): Identifiant unique du neurone.
        tau_m (float): Constante de temps membranaire (ms).
        v_rest (float): Potentiel de repos du neurone (mV).
        v_threshold (float): Seuil de déclenchement du spike (mV).
        v_reset (float): Potentiel membranaire après un spike (mV).
        r_m (float): Résistance membranaire (MΩ).
        alpha (float): Facteur d'attention modulant la réponse neuronale.
        emotion_influence (float): Influence des émotions sur l'excitabilité neuronale.
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

    def update_potential(self, input_current, dt):
        """
        Met à jour le potentiel membranaire du neurone en fonction du courant synaptique.
        
        Args:
            input_current (float): Courant d'entrée appliqué au neurone.
            dt (float): Pas de temps de simulation.
        """
        total_current = input_current + self.emotion_influence
        dv = dt * ((- (self.v_m - self.v_rest) + self.r_m * self.alpha * total_current) / self.tau_m)
        self.v_m += dv
        
        # Vérifier si le neurone dépasse le seuil de déclenchement
        if self.v_m >= self.v_threshold:
            self.v_m = self.v_reset
            self.spike = True
        else:
            self.spike = False

    def reset(self):
        """Réinitialise le potentiel du neurone après un spike."""
        self.v_m = self.v_rest
        self.spike = False
