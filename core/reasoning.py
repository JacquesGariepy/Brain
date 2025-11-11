"""
Module de raisonnement avec moteur d'inférence réel.

Implémente:
- Forward chaining (chaînage avant)
- Backward chaining (chaînage arrière)
- Unification pour pattern matching
- Base de règles et de faits
- Résolution de requêtes logiques
- Inférence déductive

Basé sur les systèmes experts classiques (CLIPS, Prolog) mais simplifié.
"""
from .interfaces import BrainModule
import logging
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


class Rule:
    """
    Représente une règle logique IF-THEN.

    Attributes:
        name (str): Nom de la règle
        conditions (list): Liste de conditions (antécédents)
        conclusions (list): Liste de conclusions (conséquents)
        confidence (float): Facteur de confiance (0-1)
    """

    def __init__(self, name: str, conditions: list, conclusions: list, confidence: float = 1.0):
        """
        Initialise une règle.

        Args:
            name: Nom de la règle
            conditions: Liste de conditions qui doivent être satisfaites
            conclusions: Liste de conclusions à tirer si conditions satisfaites
            confidence: Facteur de confiance (0-1)
        """
        self.name = name
        self.conditions = conditions
        self.conclusions = conclusions
        self.confidence = confidence

    def __repr__(self):
        return f"Rule({self.name}: IF {self.conditions} THEN {self.conclusions})"


class ReasoningModule(BrainModule):
    """
    Module de raisonnement logique avec moteur d'inférence.

    Implémente un moteur d'inférence capable de chaînage avant et arrière,
    avec unification pour le pattern matching.

    Attributes:
        facts (set): Base de faits connus
        rules (list): Base de règles
        inferences (list): Inférences faites
        inference_log (list): Historique du raisonnement
    """

    def __init__(self):
        """Initialise le module de raisonnement."""
        self.facts: Set[str] = set()
        self.rules: List[Rule] = []
        self.inferences: List[str] = []
        self.inference_log: List[Dict] = []

        # Caches pour optimisation
        self.fact_index = defaultdict(set)  # Index des faits par prédicat

        logger.info("ReasoningModule initialisé avec moteur d'inférence")

    def add_fact(self, fact: str):
        """
        Ajoute un fait à la base de connaissances.

        Args:
            fact: Fait à ajouter (string)
        """
        self.facts.add(fact)

        # Indexer le fait par son prédicat (premier mot)
        predicate = fact.split()[0] if ' ' in fact else fact
        self.fact_index[predicate].add(fact)

        logger.debug(f"Fait ajouté: {fact}")

    def add_rule(self, name: str, conditions: list, conclusions: list, confidence: float = 1.0):
        """
        Ajoute une règle à la base de règles.

        Args:
            name: Nom de la règle
            conditions: Liste de conditions
            conclusions: Liste de conclusions
            confidence: Facteur de confiance
        """
        rule = Rule(name, conditions, conclusions, confidence)
        self.rules.append(rule)
        logger.info(f"Règle ajoutée: {rule}")

    def unify(self, pattern: str, fact: str, bindings: Optional[Dict] = None) -> Optional[Dict]:
        """
        Unifie un pattern avec un fait, gérant les variables.

        Les variables commencent par '?'.
        Exemple: unify("parent ?x ?y", "parent john mary") -> {?x: john, ?y: mary}

        Args:
            pattern: Pattern à matcher (peut contenir des variables)
            fact: Fait à unifier
            bindings: Bindings existants (pour récursion)

        Returns:
            Dictionnaire de bindings si unification réussie, None sinon
        """
        if bindings is None:
            bindings = {}

        pattern_tokens = pattern.split()
        fact_tokens = fact.split()

        # Vérifier longueur
        if len(pattern_tokens) != len(fact_tokens):
            return None

        # Unifier chaque token
        for p_token, f_token in zip(pattern_tokens, fact_tokens):
            if p_token.startswith('?'):
                # C'est une variable
                if p_token in bindings:
                    # Variable déjà liée, vérifier cohérence
                    if bindings[p_token] != f_token:
                        return None
                else:
                    # Lier la variable
                    bindings[p_token] = f_token
            else:
                # Constante, doit matcher exactement
                if p_token != f_token:
                    return None

        return bindings

    def apply_bindings(self, pattern: str, bindings: Dict) -> str:
        """
        Applique les bindings de variables à un pattern.

        Args:
            pattern: Pattern avec variables
            bindings: Dictionnaire de bindings

        Returns:
            Pattern avec variables substituées
        """
        result = pattern
        for var, value in bindings.items():
            result = result.replace(var, value)
        return result

    def evaluate_conditions(self, conditions: list) -> List[Dict]:
        """
        Évalue une liste de conditions et retourne tous les bindings possibles.

        Args:
            conditions: Liste de conditions (patterns)

        Returns:
            Liste de dictionnaires de bindings qui satisfont toutes les conditions
        """
        if not conditions:
            return [{}]

        # Commencer avec la première condition
        first_condition = conditions[0]
        remaining_conditions = conditions[1:]

        all_bindings = []

        # Trouver tous les faits qui matchent la première condition
        for fact in self.facts:
            bindings = self.unify(first_condition, fact)

            if bindings is not None:
                # Vérifier les conditions restantes avec ces bindings
                if remaining_conditions:
                    # Appliquer les bindings aux conditions restantes
                    instantiated_conditions = [
                        self.apply_bindings(cond, bindings)
                        for cond in remaining_conditions
                    ]

                    # Récursion pour vérifier le reste
                    further_bindings = self.evaluate_conditions(instantiated_conditions)

                    for fb in further_bindings:
                        # Fusionner les bindings
                        merged = {**bindings, **fb}
                        all_bindings.append(merged)
                else:
                    # Pas de conditions restantes, on a un match
                    all_bindings.append(bindings)

        return all_bindings

    def forward_chaining(self, max_iterations: int = 100) -> List[str]:
        """
        Effectue un chaînage avant (forward chaining) pour inférer nouveaux faits.

        Applique itérativement les règles jusqu'à ce qu'aucun nouveau fait
        ne puisse être inféré.

        Args:
            max_iterations: Nombre max d'itérations pour éviter boucles infinies

        Returns:
            Liste des nouveaux faits inférés
        """
        new_facts = []
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            facts_added_this_iteration = 0

            # Pour chaque règle
            for rule in self.rules:
                # Évaluer les conditions
                all_bindings = self.evaluate_conditions(rule.conditions)

                # Pour chaque ensemble de bindings valides
                for bindings in all_bindings:
                    # Appliquer les bindings aux conclusions
                    for conclusion in rule.conclusions:
                        inferred_fact = self.apply_bindings(conclusion, bindings)

                        # Ajouter le fait s'il est nouveau
                        if inferred_fact not in self.facts:
                            self.add_fact(inferred_fact)
                            new_facts.append(inferred_fact)
                            self.inferences.append(inferred_fact)
                            facts_added_this_iteration += 1

                            # Logger l'inférence
                            self.inference_log.append({
                                'rule': rule.name,
                                'conditions': rule.conditions,
                                'bindings': bindings,
                                'inference': inferred_fact,
                                'confidence': rule.confidence
                            })

                            logger.debug(f"Inférence: {inferred_fact} (règle: {rule.name})")

            # Si aucun nouveau fait, on a atteint un point fixe
            if facts_added_this_iteration == 0:
                break

        logger.info(f"Forward chaining: {len(new_facts)} nouveaux faits inférés en {iteration} itérations")
        return new_facts

    def backward_chaining(self, goal: str, visited: Optional[Set] = None) -> bool:
        """
        Effectue un chaînage arrière (backward chaining) pour prouver un goal.

        Essaie de prouver le goal en travaillant à rebours depuis le goal
        vers les faits connus.

        Args:
            goal: Goal à prouver
            visited: Ensemble de goals déjà visités (éviter cycles)

        Returns:
            True si le goal peut être prouvé, False sinon
        """
        if visited is None:
            visited = set()

        # Éviter les cycles
        if goal in visited:
            return False

        visited.add(goal)

        # Vérifier si le goal est déjà un fait connu
        for fact in self.facts:
            if self.unify(goal, fact) is not None:
                logger.debug(f"Goal prouvé directement: {goal}")
                return True

        # Essayer de prouver le goal avec les règles
        for rule in self.rules:
            # Vérifier si une conclusion de la règle peut unifier avec le goal
            for conclusion in rule.conclusions:
                bindings = self.unify(conclusion, goal)

                if bindings is not None:
                    # Instancier les conditions avec les bindings
                    instantiated_conditions = [
                        self.apply_bindings(cond, bindings)
                        for cond in rule.conditions
                    ]

                    # Essayer de prouver toutes les conditions
                    all_conditions_provable = True

                    for condition in instantiated_conditions:
                        if not self.backward_chaining(condition, visited.copy()):
                            all_conditions_provable = False
                            break

                    if all_conditions_provable:
                        # Ajouter le goal comme fait inféré
                        self.add_fact(goal)
                        self.inferences.append(goal)

                        self.inference_log.append({
                            'rule': rule.name,
                            'goal': goal,
                            'method': 'backward_chaining',
                            'confidence': rule.confidence
                        })

                        logger.debug(f"Goal prouvé par backward chaining: {goal}")
                        return True

        return False

    def query(self, query_pattern: str) -> List[Dict]:
        """
        Exécute une requête sur la base de faits.

        Args:
            query_pattern: Pattern de requête (peut contenir des variables)

        Returns:
            Liste de dictionnaires de bindings pour tous les matchs
        """
        results = []

        for fact in self.facts:
            bindings = self.unify(query_pattern, fact)
            if bindings is not None:
                results.append(bindings)

        logger.debug(f"Requête '{query_pattern}': {len(results)} résultats")
        return results

    def explain(self, fact: str) -> List[Dict]:
        """
        Explique comment un fait a été inféré.

        Args:
            fact: Fait à expliquer

        Returns:
            Liste des entrées du log d'inférence concernant ce fait
        """
        explanations = [
            entry for entry in self.inference_log
            if entry.get('inference') == fact or entry.get('goal') == fact
        ]

        return explanations

    def clear_facts(self):
        """Efface tous les faits (mais garde les règles)."""
        self.facts.clear()
        self.fact_index.clear()
        logger.info("Faits effacés")

    def clear_inferences(self):
        """Efface les inférences et le log."""
        self.inferences.clear()
        self.inference_log.clear()
        logger.info("Inférences effacées")

    def process(self, data):
        """
        Traite les données pour le raisonnement.

        Peut accepter:
        - Faits individuels (strings)
        - Dictionnaires avec 'fact' ou 'query'
        - Dictionnaires avec 'goal' pour backward chaining

        Args:
            data: Données à analyser

        Returns:
            Résultats du raisonnement
        """
        logger.debug("Module de Raisonnement traite les données")

        # Traitement du raisonnement
        if isinstance(data, dict):
            # Ajouter un fait
            if 'fact' in data:
                fact = data['fact']
                self.add_fact(fact)

                # Automatiquement faire forward chaining
                new_inferences = self.forward_chaining()

                return {
                    'fact_added': fact,
                    'new_inferences': new_inferences,
                    'num_inferences': len(new_inferences),
                    'processed': True
                }

            # Exécuter une requête
            elif 'query' in data:
                query = data['query']
                results = self.query(query)

                return {
                    'query': query,
                    'results': results,
                    'num_results': len(results),
                    'processed': True
                }

            # Prouver un goal
            elif 'goal' in data:
                goal = data['goal']
                provable = self.backward_chaining(goal)

                return {
                    'goal': goal,
                    'provable': provable,
                    'explanation': self.explain(goal) if provable else [],
                    'processed': True
                }

            # Règle à ajouter
            elif 'rule' in data and 'conditions' in data and 'conclusions' in data:
                rule_name = data['rule']
                conditions = data['conditions']
                conclusions = data['conclusions']
                confidence = data.get('confidence', 1.0)

                self.add_rule(rule_name, conditions, conclusions, confidence)

                return {
                    'rule_added': rule_name,
                    'processed': True
                }

            else:
                return {
                    'processed': False,
                    'error': 'Clés attendues: fact, query, goal, ou rule'
                }

        # Ajouter comme fait simple
        elif isinstance(data, str):
            self.add_fact(data)
            new_inferences = self.forward_chaining()

            return {
                'fact_added': data,
                'new_inferences': new_inferences,
                'processed': True
            }

        else:
            return {
                'processed': False,
                'error': 'Format de données invalide'
            }

    def get_inferences(self):
        """
        Retourne les inférences faites.

        Returns:
            Liste des inférences
        """
        return self.inferences

    def get_reasoning_state(self) -> dict:
        """
        Retourne l'état actuel du moteur de raisonnement.

        Returns:
            Dictionnaire avec statistiques
        """
        return {
            'num_facts': len(self.facts),
            'num_rules': len(self.rules),
            'num_inferences': len(self.inferences),
            'inference_log_size': len(self.inference_log),
            'facts': list(self.facts),
            'rules': [r.name for r in self.rules]
        }
