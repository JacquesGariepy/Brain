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
            neuron.update_potential(input_value, dt=1.0)
            outputs.append(neuron.spike)
        return np.array(outputs)

    def backward_pass(self, errors, learning_rate):
        """
        Rétropropagation de l'erreur pour ajuster les poids synaptiques.
        
        Args:
            errors (array-like): Erreurs observées entre les sorties réelles et attendues.
            learning_rate (float): Taux d'apprentissage.
        """
        for synapse in self.network.synapses:
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

      import numpy as np

class DecisionModule:
    """
    Module de prise de décision basé sur l'accumulation d'évidence jusqu'à un seuil.
    
    Attributes:
        D_t (float): Variable d'accumulation d'évidence.
        threshold (float): Seuil pour prendre une décision.
        choice_made (bool): Indique si une décision a été prise.
        decision (str): Décision finale (positive ou négative).
    """
    
    def __init__(self, threshold=1.0, bias=0.0):
        self.D_t = 0.0  # Variable d'accumulation d'évidence
        self.threshold = threshold
        self.bias = bias
        self.choice_made = False
        self.decision = None

    def update_decision(self, evidence, emotion_influence, dt):
        """
        Met à jour la variable d'accumulation d'évidence et prend une décision si le seuil est atteint.
        
        Args:
            evidence (float): Évidence accumulée pour la décision.
            emotion_influence (float): Influence des émotions sur la décision.
            dt (float): Pas de temps de simulation.
        """
        noise = np.random.normal(0, 0.1)
        dD = dt * (evidence + self.bias + emotion_influence + noise)
        self.D_t += dD
        
        # Vérifier si le seuil de décision est atteint
    def reinforcement_learning(self, reward):
        """
        Effectue un apprentissage par renforcement basé sur les récompenses reçues.
        
        Args:
            reward (float): Récompense reçue pour renforcer ou punir un comportement.
        """
        delta = reward
        for synapse in self.network.synapses:
            synapse.update_weight_rl(delta)

