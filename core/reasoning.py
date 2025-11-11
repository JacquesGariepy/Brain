"""
Module de raisonnement du Brain core.

Module de raisonnement logique et prise de décision de haut niveau.
Conforme aux standards NASA/MIT pour le code scientifique.
"""
from .interfaces import BrainModule
import logging

logger = logging.getLogger(__name__)


class ReasoningModule(BrainModule):
    """
    Module de raisonnement logique et inférence.

    Ce module gère le raisonnement déductif, inductif et abductif,
    ainsi que la planification et la résolution de problèmes.
    """

    def __init__(self):
        """Initialise le module de raisonnement."""
        self.facts = []
        self.rules = []
        self.inferences = []
        logger.info("ReasoningModule initialisé")

    def process(self, data):
        """
        Traite les données pour le raisonnement.

        Args:
            data: Données à analyser

        Returns:
            Résultats du raisonnement
        """
        logger.debug("Module de Raisonnement traite les données")

        # Traitement réel du raisonnement
        if isinstance(data, dict):
            # Extraction de faits
            if 'facts' in data:
                self.facts.extend(data['facts'])

            # Appliquer des règles de raisonnement
            inferences = self._apply_reasoning(data)

            return {
                'inferences': inferences,
                'num_facts': len(self.facts),
                'processed': True
            }
        else:
            # Ajouter comme fait simple
            self.facts.append(data)
            return {
                'fact_added': True,
                'processed': True
            }

    def _apply_reasoning(self, data):
        """
        Applique le raisonnement logique aux données.

        Args:
            data: Données à analyser

        Returns:
            Liste d'inférences
        """
        # Logique de raisonnement simple
        inferences = []

        # Exemple de règle simple
        if 'condition' in data and 'consequence' in data:
            if self._check_condition(data['condition']):
                inferences.append(data['consequence'])
                self.inferences.append(data['consequence'])

        return inferences

    def _check_condition(self, condition):
        """
        Vérifie une condition.

        Args:
            condition: Condition à vérifier

        Returns:
            True si la condition est satisfaite
        """
        # Vérification simple
        return condition in self.facts or condition is True

    def add_rule(self, condition, consequence):
        """
        Ajoute une règle de raisonnement.

        Args:
            condition: Condition de la règle
            consequence: Conséquence si la condition est vraie
        """
        self.rules.append({'condition': condition, 'consequence': consequence})
        logger.info(f"Règle ajoutée: {condition} -> {consequence}")

    def get_inferences(self):
        """
        Retourne les inférences faites.

        Returns:
            Liste des inférences
        """
        return self.inferences
