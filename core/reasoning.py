import numpy as np
from utils.exceptions import BrainException
from utils.logging import brain_logger
from .interfaces import BrainModule


class ReasoningModule(BrainModule):
    """
    Module de raisonnement pour effectuer des inférences logiques et des déductions.
    
    Attributes:
        rules (dict): Ensemble de règles de raisonnement.
        knowledge_base (dict): Base de connaissances pour le raisonnement.
    """
    
    def __init__(self):
        """Initialise le module de raisonnement."""
        brain_logger.info("Initialisation du module de raisonnement")
        self.rules = {}
        self.knowledge_base = {}
        self.inference_history = []
        brain_logger.info("Module de raisonnement initialisé avec succès")

    def process(self, data):
        """
        Traite les données pour effectuer un raisonnement.
        
        Args:
            data: Données à traiter.
            
        Returns:
            Résultat du raisonnement.
        """
        try:
            # Applique les règles de raisonnement aux données
            result = self.apply_rules(data)
            brain_logger.debug(f"Raisonnement appliqué aux données: {result}")
            return result
        except Exception as e:
            brain_logger.error(f"Erreur lors du traitement des données: {str(e)}")
            return data

    def add_rule(self, rule_name, condition, action):
        """
        Ajoute une règle de raisonnement.
        
        Args:
            rule_name (str): Nom de la règle.
            condition (callable): Fonction de condition.
            action (callable): Fonction d'action à exécuter si la condition est vraie.
        """
        self.rules[rule_name] = {
            'condition': condition,
            'action': action
        }
        brain_logger.info(f"Règle ajoutée: {rule_name}")

    def apply_rules(self, data):
        """
        Applique toutes les règles de raisonnement aux données.
        
        Args:
            data: Données à traiter.
            
        Returns:
            Données traitées après application des règles.
        """
        result = data
        for rule_name, rule in self.rules.items():
            try:
                if rule['condition'](result):
                    result = rule['action'](result)
                    self.inference_history.append({
                        'rule': rule_name,
                        'input': data,
                        'output': result
                    })
                    brain_logger.debug(f"Règle appliquée: {rule_name}")
            except Exception as e:
                brain_logger.warning(f"Erreur lors de l'application de la règle {rule_name}: {str(e)}")
        return result

    def add_knowledge(self, key, value):
        """
        Ajoute une connaissance à la base de connaissances.
        
        Args:
            key (str): Clé de la connaissance.
            value: Valeur de la connaissance.
        """
        self.knowledge_base[key] = value
        brain_logger.info(f"Connaissance ajoutée: {key}")

    def query_knowledge(self, key):
        """
        Interroge la base de connaissances.
        
        Args:
            key (str): Clé à rechercher.
            
        Returns:
            Valeur associée à la clé ou None.
        """
        result = self.knowledge_base.get(key, None)
        brain_logger.debug(f"Requête de connaissance pour '{key}': {result}")
        return result

    def deduce(self, premises):
        """
        Effectue une déduction basée sur des prémisses.
        
        Args:
            premises (list): Liste des prémisses.
            
        Returns:
            Conclusion déduite.
        """
        try:
            # Logique de déduction simple
            # Ceci est un exemple basique, à étendre selon les besoins
            if all(premises):
                conclusion = True
            else:
                conclusion = False
            
            self.inference_history.append({
                'type': 'deduction',
                'premises': premises,
                'conclusion': conclusion
            })
            
            brain_logger.debug(f"Déduction effectuée: {premises} -> {conclusion}")
            return conclusion
        except Exception as e:
            brain_logger.error(f"Erreur lors de la déduction: {str(e)}")
            raise BrainException(f"Erreur de déduction: {str(e)}")

    def get_inference_history(self):
        """
        Retourne l'historique des inférences.
        
        Returns:
            list: Historique des inférences.
        """
        return self.inference_history

    def clear_history(self):
        """Efface l'historique des inférences."""
        self.inference_history = []
        brain_logger.info("Historique des inférences effacé")
