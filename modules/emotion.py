class EmotionModule:
    """
    Module émotionnel qui gère les états émotionnels et leur influence sur le comportement neuronal.
    
    Attributes:
        emotional_states (dict): Contient les niveaux actuels des différentes émotions.
        tau_E (float): Constante de temps pour la dynamique des émotions.
    """
    
    def __init__(self):
        self.emotional_states = {
            "joy": 0.0,
            "sadness": 0.0,
            "fear": 0.0,
            "anger": 0.0,
            "surprise": 0.0,
            "disgust": 0.0
        }
        self.tau_E = 100.0  # Constante de temps pour les émotions
        self.emotion_influence = {}

    def update_emotions(self, sensory_inputs, memories, reward, dt):
        """
        Met à jour les émotions en fonction des stimuli sensoriels, des souvenirs et des récompenses.
        
        Args:
            sensory_inputs (dict): Entrées sensorielles actuelles.
            memories (list): Souvenirs récents.
            reward (float): Récompense ou punition reçue.
            dt (float): Pas de temps de simulation.
        """
        for emotion in self.emotional_states:
            dE = dt * (-self.emotional_states[emotion] + self.compute_emotion_influence(emotion, sensory_inputs, reward))
            self.emotional_states[emotion] += dE / self.tau_E

    def compute_emotion_influence(self, emotion, sensory_inputs, reward):
        """
        Calcule l'influence des entrées sensorielles et des récompenses sur chaque émotion.

        Utilise un modèle basé sur l'appraisal theory: évalue les caractéristiques
        des stimuli pour déterminer l'émotion appropriée.

        Args:
            emotion (str): Nom de l'émotion.
            sensory_inputs (list or dict): Entrées sensorielles actuelles (floats).
            reward (float): Récompense reçue.

        Returns:
            float: Influence calculée sur l'émotion (0-1).
        """
        # Convertir sensory_inputs en liste si nécessaire
        if isinstance(sensory_inputs, dict):
            input_values = list(sensory_inputs.values())
        elif isinstance(sensory_inputs, (list, tuple)):
            input_values = list(sensory_inputs)
        else:
            input_values = [0.0]

        # Calculer des statistiques sur les entrées sensorielles
        if len(input_values) > 0:
            mean_input = sum(input_values) / len(input_values)
            max_input = max(input_values)
            variance = sum((x - mean_input) ** 2 for x in input_values) / len(input_values)
        else:
            mean_input = 0.0
            max_input = 0.0
            variance = 0.0

        # Appraisal-based emotion computation
        if emotion == "joy":
            # Joy: récompense positive + stimuli agréables (valence positive)
            return max(0.0, min(1.0, reward + 0.3 * mean_input))

        elif emotion == "sadness":
            # Sadness: récompense négative + stimuli faibles
            sadness_level = -min(reward, 0) * 0.5 + (1.0 - mean_input) * 0.3
            return max(0.0, min(1.0, sadness_level))

        elif emotion == "fear":
            # Fear: stimuli intenses et inattendus (haute variance + intensité élevée)
            threat_level = max_input * 0.6 + variance * 0.4
            return max(0.0, min(1.0, threat_level if threat_level > 0.7 else 0.0))

        elif emotion == "anger":
            # Anger: récompense négative avec stimuli intenses (frustration)
            if reward < -0.3:
                frustration = -reward * 0.5 + max_input * 0.3
                return max(0.0, min(1.0, frustration))
            return 0.0

        elif emotion == "surprise":
            # Surprise: forte variance (changements soudains)
            return max(0.0, min(1.0, variance if variance > 0.5 else 0.0))

        elif emotion == "disgust":
            # Disgust: récompense fortement négative
            return max(0.0, min(1.0, -reward if reward < -0.5 else 0.0))

        else:
            return 0.0

    def influence_on_neurons(self, neurons):
        """
        Applique l'influence émotionnelle sur les neurones du réseau.

        Chaque émotion a un effet différent sur l'excitabilité neuronale:
        - Joy: Augmente légèrement l'excitabilité
        - Sadness: Diminue l'excitabilité
        - Fear: Augmente fortement l'excitabilité (vigilance)
        - Anger: Augmente l'excitabilité (arousal)
        - Surprise: Augmente temporairement l'excitabilité
        - Disgust: Effet neutre ou légèrement négatif

        Args:
            neurons (list): Liste des neurones du réseau.
        """
        # Calculer l'influence émotionnelle totale
        total_influence = (
            self.emotional_states["joy"] * 0.05 +          # Léger boost
            self.emotional_states["sadness"] * (-0.08) +   # Réduction
            self.emotional_states["fear"] * 0.15 +         # Fort boost (vigilance)
            self.emotional_states["anger"] * 0.10 +        # Boost modéré (arousal)
            self.emotional_states["surprise"] * 0.12 +     # Boost temporaire
            self.emotional_states["disgust"] * (-0.03)     # Légère réduction
        )

        # Appliquer à tous les neurones
        for neuron in neurons:
            neuron.emotion_influence = total_influence
