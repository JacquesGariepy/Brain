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
        self.current_time = 0.0  # Temps courant de la simulation
    
    def add_neuron(self, neuron):
        """Ajoute un neurone au réseau."""
        self.neurons.append(neuron)
    
    def connect_neurons(self, pre_neuron, post_neuron, weight=0.5, delay=1.0):
        """Crée une synapse entre deux neurones."""
        synapse = Synapse(pre_neuron, post_neuron, weight, delay)
        self.synapses.append(synapse)
        pre_neuron.add_outgoing_synapse(synapse)
        post_neuron.add_incoming_synapse(synapse)
    
    def update(self, dt):
        """Met à jour le réseau (neurones et synapses)."""
        self.current_time += dt
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
        
        # Réinitialiser le courant des neurones pour le prochain pas de temps
        for neuron in self.neurons:
            neuron.reset_current()
