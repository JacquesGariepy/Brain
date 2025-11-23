# core/perception.py

from .interfaces import BrainModule
from utils.logging import brain_logger
from utils.exceptions import PerceptionException
import numpy as np


class PerceptionModule(BrainModule):
    """
    Module de perception pour traiter les entrées sensorielles.
    
    Attributes:
        sensory_buffer (list): Buffer pour stocker les entrées sensorielles.
        encoding_weights (dict): Poids d'encodage pour différents types de stimuli.
    """
    
    def __init__(self):
        """Initialise le module de perception."""
        brain_logger.info("Initialisation du module de perception")
        self.sensory_buffer = []
        self.encoding_weights = {
            'visual': 1.0,
            'auditory': 0.8,
            'tactile': 0.6,
            'olfactory': 0.4,
            'gustatory': 0.4
        }
        brain_logger.info("Module de perception initialisé avec succès")
    
    def process(self, data):
        """
        Traite les données sensorielles.
        
        Args:
            data: Données sensorielles à traiter.
            
        Returns:
            Données perçues et encodées.
        """
        try:
            brain_logger.debug(f"Traitement des données de perception: {type(data)}")
            
            # Encoder les données sensorielles
            encoded_data = self.encode_sensory_input(data)
            
            # Stocker dans le buffer
            self.sensory_buffer.append(encoded_data)
            
            # Limiter la taille du buffer
            if len(self.sensory_buffer) > 10:
                self.sensory_buffer.pop(0)
            
            brain_logger.debug("Données de perception traitées avec succès")
            return encoded_data
        except Exception as e:
            brain_logger.error(f"Erreur lors du traitement de la perception: {str(e)}")
            raise PerceptionException(f"Erreur de traitement: {str(e)}")
    
    def encode_sensory_input(self, sensory_input):
        """
        Encode les entrées sensorielles en représentation neuronale.
        
        Args:
            sensory_input: Entrées sensorielles (peut être un dict, list, ou array).
            
        Returns:
            Données encodées.
        """
        try:
            if isinstance(sensory_input, dict):
                # Encoder selon le type de stimulus
                encoded = {}
                for stimulus_type, value in sensory_input.items():
                    weight = self.encoding_weights.get(stimulus_type, 0.5)
                    encoded[stimulus_type] = value * weight
                return encoded
            elif isinstance(sensory_input, (list, np.ndarray)):
                # Encoder comme array
                return np.array(sensory_input) * self.encoding_weights.get('visual', 1.0)
            else:
                # Encoder comme valeur simple
                return sensory_input
        except Exception as e:
            brain_logger.error(f"Erreur lors de l'encodage sensoriel: {str(e)}")
            raise PerceptionException(f"Erreur d'encodage: {str(e)}")
    
    def get_recent_perceptions(self, n=5):
        """
        Récupère les N perceptions les plus récentes.
        
        Args:
            n (int): Nombre de perceptions à récupérer.
            
        Returns:
            list: Liste des perceptions récentes.
        """
        return self.sensory_buffer[-n:] if self.sensory_buffer else []
    
    def clear_buffer(self):
        """Efface le buffer de perception."""
        self.sensory_buffer = []
        brain_logger.info("Buffer de perception effacé")
