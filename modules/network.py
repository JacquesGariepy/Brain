class Network:
    """
    Modèle du réseau neuronal, regroupant les neurones et les synapses.
    
    Methods:
        add_neuron: Ajoute un neurone au réseau.
        connect_neurons: Crée une synapse entre deux neurones.
        update: Met à jour le réseau (neurones et synapses).
    """
    
    def __init__(self):
        self.neurons = []
        self.synapses = []

    def add_neuron(self, neuron):
        """Ajoute un neurone au réseau."""
        self.neurons.append(neuron)

    def connect_neurons(self, pre_neuron, post_neuron, weight=0.5, delay=1.0):
        """Crée une synapse entre deux neurones."""
        synapse = Synapse(pre_neuron, post_neuron, weight, delay)
        self.synapses.append(synapse)
        pre_neuron.add_outgoing_synapse(synapse)
        post_neuron.add
