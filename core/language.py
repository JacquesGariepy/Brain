"""
Module de langage avec traitement NLP réel.

Implémente:
- Tokenization avancée (mots, ponctuations, nombres)
- POS Tagging basique (part-of-speech)
- Named Entity Recognition basique
- Analyse syntaxique simple (structure de phrase)
- Extraction de dépendances sémantiques
- Word embeddings basiques (co-occurrence)
- Analyse de sentiment

Sans dépendances ML externes. Implémentation algorithmique pure.
"""
from .interfaces import BrainModule
import logging
import re
from collections import defaultdict, Counter
import math

logger = logging.getLogger(__name__)


class LanguageModule(BrainModule):
    """
    Module de traitement du langage naturel.

    Implémente un pipeline NLP complet sans dépendances externes,
    utilisant des règles linguistiques et des modèles statistiques simples.

    Attributes:
        vocabulary (dict): Vocabulaire avec fréquences
        word_vectors (dict): Vecteurs de mots basés sur co-occurrence
        pos_patterns (dict): Patterns pour POS tagging
        entity_patterns (dict): Patterns pour NER
        sentiment_lexicon (dict): Lexique de sentiment
    """

    def __init__(self):
        """Initialise le module de langage."""
        self.vocabulary = defaultdict(int)
        self.word_vectors = {}
        self.pos_patterns = self._init_pos_patterns()
        self.entity_patterns = self._init_entity_patterns()
        self.sentiment_lexicon = self._init_sentiment_lexicon()
        self.processed_sentences = []

        # Co-occurrence matrix for embeddings
        self.cooccurrence = defaultdict(lambda: defaultdict(int))

        logger.info("LanguageModule (core) initialisé avec NLP complet")

    def _init_pos_patterns(self) -> dict:
        """
        Initialise les patterns de POS tagging.

        Returns:
            Dictionnaire de patterns regex pour identifier les parties du discours
        """
        return {
            'VERB': r'\b(est|sont|était|faire|va|aller|parler|manger|boire|voir|entendre).*\b',
            'NOUN': r'\b(intelligence|artificielle|cerveau|neurone|réseau|apprentissage|connaissance|système|donnée)s?\b',
            'ADJ': r'\b(bon|mauvais|grand|petit|nouveau|ancien|rapide|lent|intelligent)e?s?\b',
            'DET': r'\b(le|la|les|un|une|des|ce|cette|ces|mon|ma|mes|ton|ta|tes|son|sa|ses)\b',
            'PREP': r'\b(à|de|dans|sur|sous|avec|sans|pour|par|en)\b',
            'PRON': r'\b(je|tu|il|elle|nous|vous|ils|elles|me|te|se|lui|leur)\b',
            'CONJ': r'\b(et|ou|mais|donc|car|ni|or)\b',
            'NUM': r'\b\d+\b',
        }

    def _init_entity_patterns(self) -> dict:
        """
        Initialise les patterns de Named Entity Recognition.

        Returns:
            Dictionnaire de patterns pour identifier les entités nommées
        """
        return {
            'PERSON': r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',  # Prénom Nom
            'ORGANIZATION': r'\b(MIT|NASA|Google|Microsoft|Apple|Amazon|Tesla)\b',
            'LOCATION': r'\b(Paris|France|États-Unis|Canada|Europe|Amérique)\b',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b',
            'TIME': r'\b\d{1,2}:\d{2}(?::\d{2})?\b',
        }

    def _init_sentiment_lexicon(self) -> dict:
        """
        Initialise le lexique de sentiment.

        Returns:
            Dictionnaire de mots avec leur polarité (-1 à +1)
        """
        return {
            # Positifs
            'bon': 0.7, 'excellent': 0.9, 'super': 0.8, 'génial': 0.8,
            'heureux': 0.7, 'joie': 0.8, 'amour': 0.9, 'succès': 0.7,
            'magnifique': 0.8, 'merveilleux': 0.9, 'fantastique': 0.9,

            # Négatifs
            'mauvais': -0.7, 'terrible': -0.9, 'horrible': -0.9,
            'triste': -0.7, 'peur': -0.6, 'échec': -0.8, 'douleur': -0.7,
            'difficile': -0.5, 'problème': -0.5, 'erreur': -0.6,

            # Neutres ou modérateurs
            'pas': -0.5,  # Inverseur
            'très': 1.5,  # Intensificateur
            'peu': 0.5,   # Atténuateur
        }

    def tokenize(self, text: str) -> list:
        """
        Tokenise le texte en mots, ponctuations et nombres.

        Args:
            text: Texte à tokeniser

        Returns:
            Liste de tokens
        """
        # Préserver la ponctuation comme tokens séparés
        text = re.sub(r'([.,!?;:()])', r' \1 ', text)

        # Split sur les espaces et filtrer les tokens vides
        tokens = [t.strip() for t in text.split() if t.strip()]

        # Normaliser en minuscules (sauf pour NER)
        return tokens

    def pos_tag(self, tokens: list) -> list:
        """
        Effectue le POS tagging (étiquetage morphosyntaxique).

        Args:
            tokens: Liste de tokens

        Returns:
            Liste de tuples (token, pos_tag)
        """
        tagged = []

        for token in tokens:
            token_lower = token.lower()
            pos = 'UNKNOWN'

            # Vérifier chaque pattern
            for tag, pattern in self.pos_patterns.items():
                if re.match(pattern, token_lower):
                    pos = tag
                    break

            # Par défaut, si commence par majuscule et pas en début, c'est un nom propre
            if pos == 'UNKNOWN' and token[0].isupper() and token not in ['.', '!', '?']:
                pos = 'PROPN'
            elif pos == 'UNKNOWN':
                pos = 'NOUN'  # Fallback: considérer comme nom commun

            tagged.append((token, pos))

        return tagged

    def named_entity_recognition(self, text: str) -> list:
        """
        Effectue la reconnaissance d'entités nommées (NER).

        Args:
            text: Texte à analyser

        Returns:
            Liste de tuples (entity, entity_type)
        """
        entities = []

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append((match.group(), entity_type))

        return entities

    def parse_syntax(self, pos_tagged: list) -> dict:
        """
        Analyse syntaxique simple pour identifier la structure de phrase.

        Args:
            pos_tagged: Tokens avec POS tags

        Returns:
            Dictionnaire avec la structure syntaxique
        """
        structure = {
            'subject': [],
            'verb': [],
            'object': [],
            'modifiers': []
        }

        # Règles simplifiées
        found_verb = False

        for i, (token, pos) in enumerate(pos_tagged):
            if pos == 'VERB':
                structure['verb'].append(token)
                found_verb = True
            elif pos in ['NOUN', 'PROPN', 'PRON']:
                if not found_verb:
                    structure['subject'].append(token)
                else:
                    structure['object'].append(token)
            elif pos in ['ADJ', 'DET']:
                structure['modifiers'].append(token)

        return structure

    def extract_semantic_relations(self, syntax_structure: dict, entities: list) -> list:
        """
        Extrait les relations sémantiques du texte.

        Args:
            syntax_structure: Structure syntaxique
            entities: Entités nommées

        Returns:
            Liste de relations (sujet, prédicat, objet)
        """
        relations = []

        # Construire des triplets (sujet, verbe, objet)
        subjects = syntax_structure['subject']
        verbs = syntax_structure['verb']
        objects = syntax_structure['object']

        for verb in verbs:
            subj = ' '.join(subjects) if subjects else 'UNKNOWN'
            obj = ' '.join(objects) if objects else 'UNKNOWN'

            relations.append((subj, verb, obj))

        return relations

    def compute_sentiment(self, tokens: list) -> dict:
        """
        Calcule le sentiment du texte.

        Args:
            tokens: Liste de tokens

        Returns:
            Dictionnaire avec score de sentiment et polarité
        """
        score = 0.0
        count = 0
        modifier = 1.0

        for token in tokens:
            token_lower = token.lower()

            # Gestion des modificateurs
            if token_lower == 'très':
                modifier = 1.5
                continue
            elif token_lower == 'peu':
                modifier = 0.5
                continue
            elif token_lower == 'pas':
                modifier = -1.0
                continue

            # Calculer sentiment
            if token_lower in self.sentiment_lexicon:
                sentiment_value = self.sentiment_lexicon[token_lower] * modifier
                score += sentiment_value
                count += 1
                modifier = 1.0  # Reset modificateur

        # Normaliser le score
        if count > 0:
            avg_score = score / count
        else:
            avg_score = 0.0

        # Déterminer la polarité
        if avg_score > 0.2:
            polarity = 'POSITIVE'
        elif avg_score < -0.2:
            polarity = 'NEGATIVE'
        else:
            polarity = 'NEUTRAL'

        return {
            'score': avg_score,
            'polarity': polarity,
            'confidence': min(abs(avg_score), 1.0)
        }

    def update_cooccurrence(self, tokens: list, window_size: int = 2):
        """
        Met à jour la matrice de co-occurrence pour les embeddings.

        Args:
            tokens: Liste de tokens
            window_size: Taille de la fenêtre de contexte
        """
        for i, target_word in enumerate(tokens):
            target_lower = target_word.lower()

            # Fenêtre de contexte
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_word = tokens[j].lower()
                    self.cooccurrence[target_lower][context_word] += 1

    def compute_word_embedding(self, word: str, dimensions: int = 10) -> list:
        """
        Calcule un embedding simple basé sur la co-occurrence.

        Args:
            word: Mot à encoder
            dimensions: Nombre de dimensions de l'embedding

        Returns:
            Vecteur d'embedding
        """
        word_lower = word.lower()

        if word_lower not in self.cooccurrence:
            return [0.0] * dimensions

        # Utiliser les N mots les plus co-occurrents comme dimensions
        cooccur_words = self.cooccurrence[word_lower]
        sorted_words = sorted(cooccur_words.items(), key=lambda x: x[1], reverse=True)

        # Créer le vecteur
        vector = []
        for i, (context_word, count) in enumerate(sorted_words[:dimensions]):
            # Normaliser avec PMI (Pointwise Mutual Information) simplifié
            vector.append(math.log(count + 1))

        # Padding si nécessaire
        while len(vector) < dimensions:
            vector.append(0.0)

        # Normaliser le vecteur (L2 norm)
        norm = math.sqrt(sum(v**2 for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def process(self, data):
        """
        Traite les données linguistiques avec le pipeline NLP complet.

        Pipeline:
        1. Tokenization
        2. POS Tagging
        3. Named Entity Recognition
        4. Analyse syntaxique
        5. Extraction de relations sémantiques
        6. Analyse de sentiment
        7. Mise à jour des embeddings

        Args:
            data: Données linguistiques à traiter

        Returns:
            Données linguistiques traitées avec toutes les analyses
        """
        logger.debug("Module de Langage traite les données")

        # Traitement réel des données linguistiques
        if isinstance(data, str):
            # 1. Tokenization
            tokens = self.tokenize(data)

            # 2. POS Tagging
            pos_tagged = self.pos_tag(tokens)

            # 3. Named Entity Recognition
            entities = self.named_entity_recognition(data)

            # 4. Analyse syntaxique
            syntax_structure = self.parse_syntax(pos_tagged)

            # 5. Relations sémantiques
            relations = self.extract_semantic_relations(syntax_structure, entities)

            # 6. Analyse de sentiment
            sentiment = self.compute_sentiment(tokens)

            # 7. Mise à jour co-occurrence pour embeddings
            self.update_cooccurrence(tokens)

            # 8. Mettre à jour vocabulaire
            for token in tokens:
                self.vocabulary[token.lower()] += 1

            # Enregistrer la phrase traitée
            self.processed_sentences.append(data)

            # Retourner résultats complets
            return {
                'text': data,
                'tokens': tokens,
                'pos_tags': pos_tagged,
                'entities': entities,
                'syntax': syntax_structure,
                'relations': relations,
                'sentiment': sentiment,
                'word_count': len(tokens),
                'vocabulary_size': len(self.vocabulary),
                'processed': True
            }

        elif isinstance(data, dict) and 'text' in data:
            return self.process(data['text'])
        else:
            return {'processed': False, 'error': 'Format de données invalide'}

    def get_word_embedding(self, word: str) -> list:
        """
        Retourne l'embedding d'un mot.

        Args:
            word: Mot à encoder

        Returns:
            Vecteur d'embedding
        """
        if word in self.word_vectors:
            return self.word_vectors[word]
        else:
            # Calculer à la volée si pas en cache
            embedding = self.compute_word_embedding(word)
            self.word_vectors[word] = embedding
            return embedding

    def compute_similarity(self, word1: str, word2: str) -> float:
        """
        Calcule la similarité cosinus entre deux mots.

        Args:
            word1: Premier mot
            word2: Deuxième mot

        Returns:
            Similarité (0-1)
        """
        vec1 = self.get_word_embedding(word1)
        vec2 = self.get_word_embedding(word2)

        # Cosine similarity
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

        return max(0.0, dot_product)  # Déjà normalisés, donc dot product = cosine

    def get_processed_sentences(self):
        """
        Retourne les phrases traitées.

        Returns:
            Liste des phrases traitées
        """
        return self.processed_sentences

    def get_vocabulary_stats(self) -> dict:
        """
        Retourne des statistiques sur le vocabulaire.

        Returns:
            Dictionnaire avec statistiques
        """
        if len(self.vocabulary) == 0:
            return {
                'vocabulary_size': 0,
                'total_words': 0,
                'top_words': []
            }

        total_words = sum(self.vocabulary.values())
        top_words = Counter(self.vocabulary).most_common(10)

        return {
            'vocabulary_size': len(self.vocabulary),
            'total_words': total_words,
            'top_words': top_words
        }
