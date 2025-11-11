"""
Module de prise de décision basé sur l'accumulation d'évidence.

Implémente un modèle de drift-diffusion pour la prise de décision,
avec influence émotionnelle et bruit stochastique.

Conforme aux standards NASA/MIT pour le code scientifique.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DecisionModule:
    """
    Module de prise de décision basé sur l'accumulation d'évidence jusqu'à un seuil.

    Attributes:
        D_t (float): Variable d'accumulation d'évidence.
        threshold (float): Seuil pour prendre une décision.
        bias (float): Biais dans l'accumulation d'évidence.
        choice_made (bool): Indique si une décision a été prise.
        decision (str): Décision finale (positive ou négative).
    """

    def __init__(self, threshold: float = 1.0, bias: float = 0.0):
        """
        Initialise le module de décision.

        Args:
            threshold: Seuil pour prendre une décision
            bias: Biais dans l'accumulation d'évidence
        """
        self.D_t = 0.0  # Variable d'accumulation d'évidence
        self.threshold = threshold
        self.bias = bias
        self.choice_made = False
        self.decision = None
        logger.info(f"DecisionModule créé avec threshold={threshold}, bias={bias}")

    def update_decision(self, evidence: float, emotion_influence: float, dt: float):
        """
        Met à jour la variable d'accumulation d'évidence et prend une décision si le seuil est atteint.

        Args:
            evidence: Évidence accumulée pour la décision
            emotion_influence: Influence des émotions sur la décision
            dt: Pas de temps de simulation
        """
        # Ajouter du bruit stochastique pour modéliser l'incertitude
        noise = np.random.normal(0, 0.1)

        # Équation de drift-diffusion
        dD = dt * (evidence + self.bias + emotion_influence + noise)
        self.D_t += dD

        # Vérifier si le seuil de décision est atteint
        if abs(self.D_t) >= self.threshold:
            self.choice_made = True
            self.decision = "Action positive" if self.D_t > 0 else "Action négative"
            logger.info(f"Décision prise: {self.decision} (D_t={self.D_t:.2f})")

    def reset(self):
        """Réinitialise la variable d'accumulation d'évidence après qu'une décision a été prise."""
        self.D_t = 0.0
        self.choice_made = False
        self.decision = None
        logger.debug("DecisionModule réinitialisé")

    def get_state(self) -> dict:
        """
        Retourne l'état actuel du module de décision.

        Returns:
            Dictionnaire contenant l'état
        """
        return {
            'D_t': self.D_t,
            'threshold': self.threshold,
            'bias': self.bias,
            'choice_made': self.choice_made,
            'decision': self.decision
        }

    def set_threshold(self, threshold: float):
        """
        Modifie le seuil de décision.

        Args:
            threshold: Nouveau seuil
        """
        self.threshold = max(0.1, threshold)
        logger.info(f"Seuil de décision modifié: {self.threshold}")

    def set_bias(self, bias: float):
        """
        Modifie le biais de décision.

        Args:
            bias: Nouveau biais
        """
        self.bias = bias
        logger.info(f"Biais de décision modifié: {self.bias}")
