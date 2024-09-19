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
                if abs(self.D_t) >= self.threshold:
            self.choice_made = True
            self.decision = "Action positive" if self.D_t > 0 else "Action négative"

    def reset(self):
        """Réinitialise la variable d'accumulation d'évidence après qu'une décision a été prise."""
        self.D_t = 0.0
        self.choice_made = False
        self.decision = None
