class PerceptionModule:
    """
    Module de perception qui gère l'entrée sensorielle et la codification pour le réseau neuronal.
    
    Attributes:
        network (Network): Réseau neuronal auquel les entrées sont transmises.
        sensory_neurons (list): Liste des neurones sensoriels.
    """
    
    def __init__(self, network):
        self.network = network
        self.sensory_neurons = []

    def add_sensory_neurons(self, neurons):
        """
        Ajoute des neurones sensoriels au module de perception.
        
        Args:
            neurons (list): Liste des neurones sensoriels à ajouter.
        """
        self.sensory_neurons.extend(neurons)

    def encode_sensory_input(self, sensory_input):
        """
        Encode les entrées sensorielles en courants neuronaux pour les neurones sensoriels.
        
        Args:
            sensory_input (list): Liste des entrées sensorielles à encoder.
        """
        for neuron, value in zip(self.sensory_neurons, sensory_input):
            neuron.v_m += value  # Mise à jour du potentiel membranaire des neurones sensoriels
