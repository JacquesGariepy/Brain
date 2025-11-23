"""
Module de gestion des exceptions personnalisées pour le projet Brain.
"""


class BrainException(Exception):
    """Classe de base pour toutes les exceptions du projet Brain."""
    
    def __init__(self, message, details=None):
        """
        Initialise une exception Brain.
        
        Args:
            message (str): Message d'erreur principal.
            details (dict, optional): Détails supplémentaires sur l'erreur.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} - Détails: {self.details}"
        return self.message


class NeuronException(BrainException):
    """Exception levée lors d'erreurs liées aux neurones."""
    pass


class SynapseException(BrainException):
    """Exception levée lors d'erreurs liées aux synapses."""
    pass


class NetworkException(BrainException):
    """Exception levée lors d'erreurs liées au réseau neuronal."""
    pass


class MemoryException(BrainException):
    """Exception levée lors d'erreurs liées à la mémoire."""
    pass


class LearningException(BrainException):
    """Exception levée lors d'erreurs liées à l'apprentissage."""
    pass


class PerceptionException(BrainException):
    """Exception levée lors d'erreurs liées à la perception."""
    pass


class LanguageException(BrainException):
    """Exception levée lors d'erreurs liées au traitement du langage."""
    pass


class EmotionException(BrainException):
    """Exception levée lors d'erreurs liées aux émotions."""
    pass


class DecisionException(BrainException):
    """Exception levée lors d'erreurs liées à la prise de décision."""
    pass


class AttentionException(BrainException):
    """Exception levée lors d'erreurs liées à l'attention."""
    pass


class PluginException(BrainException):
    """Exception levée lors d'erreurs liées aux plugins."""
    pass


class ConfigurationException(BrainException):
    """Exception levée lors d'erreurs de configuration."""
    pass


class ValidationException(BrainException):
    """Exception levée lors d'erreurs de validation des données."""
    pass
