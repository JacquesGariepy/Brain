class AttentionModule:
    """
    Module d'attention qui ajuste le facteur alpha des neurones en fonction de la pertinence des stimuli.
    
    Attributes:
        neurons (list): Liste des neurones impliqués dans le réseau.
    """
    
    def __init__(self, neurons):
        self.neurons = neurons

    def update_attention(self, relevance_signal):
        """
        Met à jour les facteurs d'attention des neurones.
        
        Args:
            relevance_signal (dict): Dictionnaire contenant les signaux de pertinence pour chaque neurone.
                                     La clé est l'identifiant du neurone et la valeur est la pertinence (0-1).
        """
        for neuron in self.neurons:
            neuron.alpha = 1.0 + relevance_signal.get(neuron.neuron_id, 0.0)
