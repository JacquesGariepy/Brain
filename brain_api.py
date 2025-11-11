"""
Brain API - Interface simple pour utiliser le Brain dans n'importe quel projet.

Usage:
    from brain_api import BrainAPI

    brain = BrainAPI()
    brain.learn_from_examples(data, labels)
    result = brain.predict(new_data)
"""
import numpy as np
import logging
from typing import List, Dict, Any, Union, Optional
from core.brain import Brain

logger = logging.getLogger(__name__)


class BrainAPI:
    """
    API unifiée pour utiliser le Brain dans des projets réels.

    Le Brain peut:
    - Apprendre de n'importe quelles données (supervised, unsupervised, reinforcement)
    - Traiter du texte (NLP complet)
    - Raisonner logiquement (forward/backward chaining)
    - Prendre des décisions (drift-diffusion)
    - Gérer des émotions
    - Maintenir une mémoire (court et long terme)
    """

    def __init__(self, num_neurons: int = 50, learning_rate: float = 0.01):
        """
        Initialise le Brain.

        Args:
            num_neurons: Nombre de neurones (plus = plus de capacité)
            learning_rate: Vitesse d'apprentissage (0.001-0.1)
        """
        self.brain = Brain(num_neurons=num_neurons)
        self.brain.learning_module.learning_rate = learning_rate
        self.is_trained = False
        self.input_size = None
        self.output_size = None

        logger.info(f"BrainAPI initialisé: {num_neurons} neurones, lr={learning_rate}")

    # ============================================================================
    # 1. APPRENTISSAGE SUPERVISÉ - Classification & Régression
    # ============================================================================

    def fit(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray],
            epochs: int = 10) -> Dict[str, List[float]]:
        """
        Entraîne le Brain sur des données (comme scikit-learn).

        Args:
            X: Données d'entrée (samples x features)
            y: Labels/targets (samples,)
            epochs: Nombre d'époques d'entraînement

        Returns:
            Historique d'entraînement avec erreurs

        Example:
            >>> X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            >>> y = [0, 1, 0]
            >>> history = brain.fit(X, y, epochs=5)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.input_size = X.shape[1]
        self.output_size = 1 if y.ndim == 1 else y.shape[1]

        # Si y est des labels (0, 1, 2...), convertir en vecteurs
        if y.ndim == 1:
            y_vectors = self._labels_to_vectors(y, self.input_size)
        else:
            y_vectors = y

        history = {'train_error': []}

        logger.info(f"Entraînement: {len(X)} samples, {epochs} epochs")

        for epoch in range(epochs):
            epoch_errors = []

            for i in range(len(X)):
                error = self.brain.learn(
                    X[i].tolist(),
                    y_vectors[i].tolist(),
                    learning_type='supervised'
                )
                epoch_errors.append(error)

            mean_error = np.mean(epoch_errors)
            history['train_error'].append(mean_error)

            if (epoch + 1) % max(1, epochs // 10) == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: error={mean_error:.4f}")

        self.is_trained = True
        return history

    def predict(self, X: Union[List, np.ndarray]) -> np.ndarray:
        """
        Fait des prédictions sur nouvelles données.

        Args:
            X: Données d'entrée (samples x features)

        Returns:
            Prédictions (samples,)

        Example:
            >>> predictions = brain.predict([[0.2, 0.3]])
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        for sample in X:
            # Forward pass à travers le réseau
            output = self.brain.learning_module.forward_pass(sample.tolist())
            # Convertir output en prédiction
            pred = np.argmax(output) if len(output) > 1 else (1 if output[0] > 0.5 else 0)
            predictions.append(pred)

        return np.array(predictions)

    def score(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> float:
        """
        Calcule l'accuracy sur données de test.

        Args:
            X: Données de test
            y: Vraies labels

        Returns:
            Accuracy (0-1)
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y)
        accuracy = np.mean(y_pred == y_true)
        return accuracy

    # ============================================================================
    # 2. TRAITEMENT DU LANGAGE - NLP Complet
    # ============================================================================

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyse complète d'un texte (NLP).

        Args:
            text: Texte à analyser

        Returns:
            Dictionnaire avec:
            - tokens: Mots tokenisés
            - pos_tags: Étiquettes grammaticales
            - entities: Entités nommées
            - sentiment: Analyse de sentiment
            - syntax: Structure syntaxique
            - relations: Relations sémantiques

        Example:
            >>> result = brain.analyze_text("Le cerveau est intelligent")
            >>> print(result['sentiment'])  # {'score': 0.7, 'polarity': 'POSITIVE'}
        """
        return self.brain.modules['language'].process(text)

    def get_word_similarity(self, word1: str, word2: str) -> float:
        """
        Calcule la similarité entre deux mots.

        Args:
            word1, word2: Mots à comparer

        Returns:
            Similarité (0-1)
        """
        return self.brain.modules['language'].compute_similarity(word1, word2)

    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyse de sentiment rapide.

        Returns:
            {'score': float, 'polarity': 'POSITIVE/NEGATIVE/NEUTRAL', 'confidence': float}
        """
        result = self.analyze_text(text)
        return result.get('sentiment', {'score': 0, 'polarity': 'NEUTRAL', 'confidence': 0})

    # ============================================================================
    # 3. RAISONNEMENT LOGIQUE - Inférence
    # ============================================================================

    def add_knowledge(self, fact: str):
        """
        Ajoute un fait à la base de connaissances.

        Example:
            >>> brain.add_knowledge("parent john mary")
            >>> brain.add_knowledge("parent mary susan")
        """
        self.brain.modules['reasoning'].add_fact(fact)

    def add_rule(self, name: str, conditions: List[str], conclusions: List[str]):
        """
        Ajoute une règle logique.

        Example:
            >>> brain.add_rule(
            ...     "ancestor_rule",
            ...     conditions=["parent ?x ?y", "parent ?y ?z"],
            ...     conclusions=["ancestor ?x ?z"]
            ... )
        """
        self.brain.modules['reasoning'].add_rule(name, conditions, conclusions)

    def infer(self) -> List[str]:
        """
        Fait des inférences logiques (forward chaining).

        Returns:
            Liste de nouveaux faits inférés
        """
        return self.brain.modules['reasoning'].forward_chaining()

    def prove(self, goal: str) -> bool:
        """
        Prouve un goal (backward chaining).

        Returns:
            True si prouvable
        """
        return self.brain.modules['reasoning'].backward_chaining(goal)

    def query(self, pattern: str) -> List[Dict]:
        """
        Interroge la base de connaissances.

        Example:
            >>> results = brain.query("parent ?x mary")
            >>> print(results)  # [{'?x': 'john'}]
        """
        return self.brain.modules['reasoning'].query(pattern)

    # ============================================================================
    # 4. MÉMOIRE - Court et Long Terme
    # ============================================================================

    def remember(self, key: str, data: Any):
        """
        Stocke en mémoire long terme.

        Example:
            >>> brain.remember("user_preference", {"theme": "dark"})
        """
        self.brain.memory_module.store_long_term(key, data)

    def recall(self, key: str) -> Any:
        """
        Récupère de la mémoire long terme.

        Returns:
            Données stockées ou None
        """
        return self.brain.memory_module.retrieve_long_term(key)

    def get_recent_memories(self) -> List[Any]:
        """
        Récupère la mémoire court terme.

        Returns:
            Liste des derniers items mémorisés
        """
        return self.brain.memory_module.retrieve_short_term()

    # ============================================================================
    # 5. DÉCISION - Drift-Diffusion
    # ============================================================================

    def decide(self, evidence: float, dt: float = 1.0) -> Optional[str]:
        """
        Prend une décision basée sur l'évidence.

        Args:
            evidence: Évidence pour/contre (-1 à +1)
            dt: Temps d'intégration

        Returns:
            Décision ("Action positive" / "Action négative") ou None si pas encore décidé

        Example:
            >>> decision = brain.decide(evidence=0.7)
        """
        emotion_influence = self.brain.emotion_module.emotional_states.get("fear", 0)
        self.brain.decision_module.update_decision(evidence, emotion_influence, dt)

        if self.brain.decision_module.choice_made:
            decision = self.brain.decision_module.decision
            self.brain.decision_module.reset()
            return decision
        return None

    # ============================================================================
    # 6. ÉMOTIONS - Appraisal Theory
    # ============================================================================

    def get_emotions(self) -> Dict[str, float]:
        """
        Récupère l'état émotionnel actuel.

        Returns:
            {'joy': 0.5, 'sadness': 0.1, 'fear': 0.0, ...}
        """
        return self.brain.emotion_module.emotional_states.copy()

    def update_emotions(self, sensory_inputs: List[float], reward: float = 0.0):
        """
        Met à jour les émotions basées sur stimuli et récompense.

        Args:
            sensory_inputs: Stimuli sensoriels
            reward: Récompense (-1 à +1)
        """
        self.brain.emotion_module.update_emotions(
            sensory_inputs,
            self.brain.memory_module.retrieve_short_term(),
            reward,
            dt=1.0
        )

    # ============================================================================
    # 7. ÉTAT & PERSISTANCE
    # ============================================================================

    def save(self, filepath: str = "brain_state.json"):
        """
        Sauvegarde l'état du Brain.

        Args:
            filepath: Chemin du fichier
        """
        self.brain.memory_module.filename = filepath
        self.brain.save_state()
        logger.info(f"Brain sauvegardé: {filepath}")

    def load(self, filepath: str = "brain_state.json"):
        """
        Charge l'état du Brain.

        Args:
            filepath: Chemin du fichier
        """
        self.brain.memory_module.filename = filepath
        self.brain.load_state()
        logger.info(f"Brain chargé: {filepath}")

    def get_status(self) -> Dict[str, Any]:
        """
        Récupère l'état complet du Brain.

        Returns:
            Statistiques sur tous les modules
        """
        return self.brain.get_status()

    def reset(self):
        """
        Réinitialise le Brain (mais garde la structure).
        """
        self.brain.network.reset()
        self.brain.memory_module.short_term_memory.clear()
        self.brain.decision_module.reset()
        self.is_trained = False
        logger.info("Brain réinitialisé")

    # ============================================================================
    # MÉTHODES UTILITAIRES
    # ============================================================================

    def _labels_to_vectors(self, labels: np.ndarray, size: int) -> np.ndarray:
        """Convertit labels en vecteurs one-hot."""
        unique_labels = np.unique(labels)
        vectors = []

        for label in labels:
            vector = np.zeros(size)
            # Encoder le label dans le vecteur
            label_idx = np.where(unique_labels == label)[0][0]
            if label_idx < size:
                vector[label_idx] = 1.0
            vectors.append(vector)

        return np.array(vectors)

    def __repr__(self):
        return (f"BrainAPI(neurons={len(self.brain.neurons)}, "
                f"synapses={len(self.brain.synapses)}, "
                f"trained={self.is_trained})")


# ============================================================================
# EXEMPLES D'UTILISATION
# ============================================================================

def example_classification():
    """Exemple: Classification de données."""
    print("\n=== EXEMPLE 1: Classification ===")

    brain = BrainAPI(num_neurons=20, learning_rate=0.05)

    # Données d'entraînement
    X_train = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    y_train = [0, 1, 0, 1]

    # Entraîner
    history = brain.fit(X_train, y_train, epochs=5)
    print(f"Erreur finale: {history['train_error'][-1]:.4f}")

    # Prédire
    X_test = [[0.2, 0.3], [0.6, 0.7]]
    predictions = brain.predict(X_test)
    print(f"Prédictions: {predictions}")

    # Score
    accuracy = brain.score(X_test, [0, 1])
    print(f"Accuracy: {accuracy:.2%}")


def example_nlp():
    """Exemple: Traitement du langage."""
    print("\n=== EXEMPLE 2: NLP ===")

    brain = BrainAPI()

    # Analyser un texte
    text = "Le cerveau artificiel fonctionne très bien"
    result = brain.analyze_text(text)

    print(f"Texte: {text}")
    print(f"Tokens: {result['tokens']}")
    print(f"Sentiment: {result['sentiment']['polarity']} (score: {result['sentiment']['score']:.2f})")
    print(f"Entités: {result['entities']}")


def example_reasoning():
    """Exemple: Raisonnement logique."""
    print("\n=== EXEMPLE 3: Raisonnement ===")

    brain = BrainAPI()

    # Ajouter des connaissances
    brain.add_knowledge("parent john mary")
    brain.add_knowledge("parent mary susan")

    # Ajouter une règle
    brain.add_rule(
        "ancestor_rule",
        conditions=["parent ?x ?y", "parent ?y ?z"],
        conclusions=["ancestor ?x ?z"]
    )

    # Inférer
    inferences = brain.infer()
    print(f"Inférences: {inferences}")

    # Prouver
    provable = brain.prove("ancestor john susan")
    print(f"'ancestor john susan' est prouvable: {provable}")

    # Query
    results = brain.query("parent ?x mary")
    print(f"Qui est parent de mary? {results}")


def example_memory():
    """Exemple: Mémoire."""
    print("\n=== EXEMPLE 4: Mémoire ===")

    brain = BrainAPI()

    # Stocker
    brain.remember("user_name", "Alice")
    brain.remember("preferences", {"theme": "dark", "lang": "fr"})

    # Récupérer
    name = brain.recall("user_name")
    prefs = brain.recall("preferences")

    print(f"Nom: {name}")
    print(f"Préférences: {prefs}")


def example_decision():
    """Exemple: Prise de décision."""
    print("\n=== EXEMPLE 5: Décision ===")

    brain = BrainAPI()

    # Accumuler évidence
    decision = None
    for i in range(10):
        decision = brain.decide(evidence=0.3)
        if decision:
            break

    print(f"Décision prise: {decision}")


def example_emotions():
    """Exemple: Émotions."""
    print("\n=== EXEMPLE 6: Émotions ===")

    brain = BrainAPI()

    # Stimuli positifs
    brain.update_emotions([0.8, 0.9, 0.7], reward=0.5)
    emotions = brain.get_emotions()
    print(f"Émotions (stimuli positifs): joy={emotions['joy']:.2f}, fear={emotions['fear']:.2f}")

    # Stimuli négatifs
    brain.update_emotions([0.2, 0.1, 0.3], reward=-0.8)
    emotions = brain.get_emotions()
    print(f"Émotions (stimuli négatifs): sadness={emotions['sadness']:.2f}, anger={emotions['anger']:.2f}")


if __name__ == "__main__":
    # Exécuter tous les exemples
    example_classification()
    example_nlp()
    example_reasoning()
    example_memory()
    example_decision()
    example_emotions()

    print("\n=== TOUTES LES FONCTIONNALITÉS DU BRAIN SONT UTILISABLES ===")
