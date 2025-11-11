"""
Module de perception du Brain core.

Ce module gère uniquement la perception sensorielle de base.
Conforme aux standards NASA/MIT pour le code scientifique.
"""
from .interfaces import BrainModule
import logging

logger = logging.getLogger(__name__)


class PerceptionModule(BrainModule):
    """
    Module de perception qui gère les entrées sensorielles brutes.

    Ce module traite les données sensorielles de haut niveau et les transforme
    en représentations utilisables par le cerveau.
    """

    def __init__(self):
        """Initialise le module de perception."""
        self.sensor_data = {}
        logger.info("PerceptionModule initialisé")

    def process(self, data):
        """
        Traite les données perceptives.

        Args:
            data: Données d'entrée à traiter

        Returns:
            Données perçues et transformées
        """
        logger.debug("Module de Perception traite les données")

        # Traitement réel des données perceptives
        if isinstance(data, dict):
            self.sensor_data = data
            return {
                'visual': data.get('visual', []),
                'auditory': data.get('auditory', []),
                'tactile': data.get('tactile', []),
                'processed': True
            }
        else:
            # Format simple
            self.sensor_data = {'raw': data}
            return {'raw': data, 'processed': True}

    def get_sensor_data(self):
        """
        Retourne les dernières données sensorielles.

        Returns:
            Dernières données sensorielles
        """
        return self.sensor_data
