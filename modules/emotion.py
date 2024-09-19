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
        
        Args:
            emotion (str): Nom de l'émotion.
            sensory_inputs (dict): Entrées sensorielles actuelles.
            reward (float): Récompense reçue.
            
        Returns:
            float: Influence calculée sur l'émotion.
        """
        if emotion == "joy":
            return max(reward, 0)
        elif emotion == "sadness":
            return -min(reward, 0)
        elif emotion == "fear":
            return 1.0 if "threat" in sensory_inputs else 0
        elif emotion == "anger":
            return 0.5 if "frustration" in sensory_inputs else 0
        else:
            return 0.0

    def influence_on_neurons(self, neurons):
        """
        Applique l'influence émotionnelle sur les neurones du réseau.
        
        Args:
            neurons (list): Liste des neurones du réseau.
        """
        for neuron in neurons:
            neuron.emotion_influence = self.emotional_states["fear"] * 0.1  # Par exemple, la peur peut augmenter l'excitabilité
