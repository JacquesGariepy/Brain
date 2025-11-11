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

    def supervised_learning(self, inputs, targets, learning_rate=0.1):
        """
        Effectue un apprentissage supervisé en ajustant les poids synaptiques en fonction des erreurs.

        Args:
            inputs (array-like): Entrées du réseau (peut être un batch ou un seul échantillon).
            targets (array-like): Sorties attendues.
            learning_rate (float): Taux d'apprentissage (augmenté à 0.1 pour des changements visibles).
        """
        inputs = np.array(inputs)
        targets = np.array(targets)

        # Handle both single samples and batches
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)

        # Train on each sample in the batch
        for input_sample, target_sample in zip(inputs, targets):
            # Forward pass - handles input padding internally
            outputs = self.forward_pass(input_sample)

            # Ensure target and output shapes match
            # outputs will always have length = number of neurons
            num_neurons = len(self.network.neurons)
            if len(target_sample) < num_neurons:
                # Pad target with zeros to match output size
                target_padded = np.zeros(num_neurons)
                target_padded[:len(target_sample)] = target_sample
                errors = target_padded - outputs
            else:
                # Truncate target to match output size
                target_truncated = target_sample[:num_neurons]
                errors = target_truncated - outputs

            self.backward_pass(errors, learning_rate)

    def forward_pass(self, inputs):
        """
        Propagation avant des entrées à travers le réseau.

        Args:
            inputs (array-like): Entrées du réseau.

        Returns:
            np.array: Sorties calculées (activations continues entre 0 et 1).
        """
        outputs = []
        # Reset all neurons
        for neuron in self.network.neurons:
            neuron.reset()

        # Process inputs - if we have fewer inputs than neurons, pad with zeros
        num_neurons = len(self.network.neurons)
        if len(inputs) < num_neurons:
            inputs_padded = np.zeros(num_neurons)
            inputs_padded[:len(inputs)] = inputs
            inputs = inputs_padded

        # Update all neurons with amplified inputs and collect normalized outputs
        for neuron, input_value in zip(self.network.neurons, inputs):
            # Amplifier le courant d'entrée pour permettre aux neurones de spiker
            amplified_input = input_value * 600.0  # Amplification forte pour le modèle LIF
            neuron.update_potential(amplified_input, dt=1.0)

            # Utiliser l'activité normalisée plutôt que juste le spike
            # Cela donne une sortie continue entre 0 et 1
            v_rest = -65.0
            v_threshold = -50.0
            normalized_output = (neuron.v_m - v_rest) / (v_threshold - v_rest)
            normalized_output = np.clip(normalized_output, 0.0, 1.0)

            # Si le neurone a spiké, sortie maximale
            if neuron.spike:
                normalized_output = 1.0

            outputs.append(normalized_output)

        return np.array(outputs, dtype=float)

    def backward_pass(self, errors, learning_rate):
        """
        Rétropropagation de l'erreur pour ajuster les poids synaptiques.

        Args:
            errors (array-like): Erreurs observées entre les sorties réelles et attendues.
            learning_rate (float): Taux d'apprentissage.
        """
        for synapse in self.network.synapses:
            # Find the index of the post_neuron in the batch
            try:
                neuron_index = self.network.neurons.index(synapse.post_neuron)
                pre_neuron_index = self.network.neurons.index(synapse.pre_neuron)

                # Utiliser l'activité normalisée du neurone pré-synaptique
                # Normaliser v_m de [-65, -50] vers [0, 1]
                v_rest = -65.0
                v_threshold = -50.0
                normalized_activity = (synapse.pre_neuron.v_m - v_rest) / (v_threshold - v_rest)
                normalized_activity = np.clip(normalized_activity, 0.0, 1.0)

                # Si le neurone a spiké, utiliser une activité de 1.0
                if synapse.pre_neuron.spike:
                    normalized_activity = 1.0

                # Ajouter un biais pour éviter les activités nulles
                # Cela permet toujours un certain apprentissage même sans spikes
                normalized_activity = max(normalized_activity, 0.1)

                # Calculer le changement de poids
                # Aussi utiliser l'erreur du neurone pré-synaptique pour la propagation
                delta_w = learning_rate * errors[neuron_index] * normalized_activity

            except ValueError:
                # Fallback: if not found, skip update
                continue

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
            synapse.update_weight_rl(delta)
