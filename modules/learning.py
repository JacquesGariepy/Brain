import numpy as np

class LearningModule:
    """
    Module d'apprentissage supervisé, non supervisé et par renforcement pour le réseau neuronal.
    
    Methods:
        supervised_learning: Apprentissage avec des exemples étiquetés.
        unsupervised_learning: Apprentissage basé sur le regroupement (clustering).
        reinforcement_learning: Apprentissage basé sur les récompenses.
    """
    
    def __init__(self, network, memory_module):
        self.network = network
        self.memory = memory_module

    def supervised_learning(self, inputs, targets, learning_rate=0.01):
        """
        Effectue un apprentissage supervisé en ajustant les poids synaptiques en fonction des erreurs.
        
        Args:
            inputs (array-like): Entrées du réseau.
            targets (array-like): Sorties attendues.
            learning_rate (float): Taux d'apprentissage.
        """
        outputs = self.forward_pass(inputs)
        errors = targets - outputs
        self.backward_pass(errors, learning_rate)

    def forward_pass(self, inputs):
        """
        Propagation avant des entrées à travers le réseau.
        
        Args:
            inputs (array-like): Entrées du réseau.
            
        Returns:
            np.array: Sorties calculées.
        """
        outputs = []
        for neuron in self.network.neurons:
            neuron.reset()
        for neuron, input_value in zip(self.network.neurons, inputs):
            neuron.update(dt=1.0)
            outputs.append(1.0 if neuron.spike else 0.0)
        return np.array(outputs)

    def backward_pass(self, errors, learning_rate):
        """
        Rétropropagation de l'erreur pour ajuster les poids synaptiques.
        
        Args:
            errors (array-like): Erreurs observées entre les sorties réelles et attendues.
            learning_rate (float): Taux d'apprentissage.
        """
        for synapse in self.network.synapses:
            if synapse.post_neuron.neuron_id < len(errors):
                delta_w = learning_rate * errors[synapse.post_neuron.neuron_id] * synapse.pre_neuron.v_m
                synapse.weight += delta_w
                synapse.weight = np.clip(synapse.weight, 0.0, 1.0)

    def unsupervised_learning(self, inputs, num_clusters=3):
        """
        Effectue un apprentissage non supervisé basé sur le regroupement des neurones en clusters.
        
        Args:
            inputs (array-like): Données d'entrée pour l'apprentissage non supervisé.
            num_clusters (int): Nombre de clusters à utiliser pour l'algorithme de k-moyennes.
        """
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(inputs.reshape(-1, 1))
        clusters = kmeans.predict(inputs.reshape(-1, 1))
        
        for i, neuron in enumerate(self.network.neurons):
            for synapse in neuron.outgoing_synapses:
                if clusters[synapse.pre_neuron.neuron_id] == clusters[synapse.post_neuron.neuron_id]:
                    synapse.weight += 0.01  # Renforcer les connexions intra-cluster
                else:
                    synapse.weight -= 0.01  # Affaiblir les connexions inter-cluster
                synapse.weight = np.clip(synapse.weight, 0.0, 1.0)

    def reinforcement_learning(self, reward):
        """
        Effectue un apprentissage par renforcement basé sur les récompenses reçues.
        
        Args:
            reward (float): Récompense reçue pour renforcer ou punir un comportement.
        """
        delta = reward
        for synapse in self.network.synapses:
            # Renforce les synapses en fonction de la récompense
            synapse.weight += 0.01 * delta
            synapse.weight = np.clip(synapse.weight, 0.0, 1.0)
