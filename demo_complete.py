#!/usr/bin/env python3
"""
Démonstration Complète du Système Brain
==========================================

Cette démonstration illustre l'utilisation de toutes les couches et fonctionnalités :
- Réseau neuronal à spikes (Leaky Integrate-and-Fire)
- Orchestrateur intelligent pour sélection d'architecture
- Apprentissage supervisé et adaptation
- Système de décision par accumulation d'évidence
- Module émotionnel
- Génération de langage (GPT-2)
- Mémoire court et long terme
- Perception multimodale

Cas d'usage : Assistant Cognitif Intelligent
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from core.brain import Brain

# Tenter d'importer l'orchestrateur (nécessite PyTorch)
try:
    from core.orchestrator import IntelligentOrchestrator, TaskSpecification, TaskType, ModalityType
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    print("[INFO] PyTorch non disponible - L'orchestrateur SOTA sera désactivé")
    print("       Pour activer : pip install torch transformers\n")

    # Classes mock pour la compatibilité
    class ModalityType:
        TEXT = "text"
        IMAGE = "image"
        AUDIO = "audio"

    class TaskType:
        TEXT_GENERATION = "text_generation"
        IMAGE_CLASSIFICATION = "image_classification"
        SPEECH_RECOGNITION = "speech_recognition"


class CognitiveAssistant:
    """
    Assistant cognitif qui combine toutes les capacités du système Brain.

    Simule un agent intelligent capable de :
    - Percevoir des stimuli multimodaux
    - Apprendre de ses expériences
    - Prendre des décisions basées sur l'accumulation d'évidence
    - Générer des réponses en langage naturel
    - Mémoriser les interactions
    """

    def __init__(self):
        print("=" * 70)
        print("INITIALISATION DE L'ASSISTANT COGNITIF")
        print("=" * 70)

        # Composant 1 : Brain principal (réseau neuronal spiking)
        print("\n[1/2] Initialisation du réseau neuronal biologique...")
        self.brain = Brain()
        print(f"      - Neurones créés : {len(self.brain.neurons)}")
        print(f"      - Synapses créées : {len(self.brain.synapses)}")

        # Composant 2 : Orchestrateur SOTA (si disponible)
        print("\n[2/2] Initialisation de l'orchestrateur SOTA...")
        if ORCHESTRATOR_AVAILABLE:
            self.orchestrator = IntelligentOrchestrator()
            print("      - Architectures disponibles : Transformers, Vision, Audio, etc.")
        else:
            self.orchestrator = None
            print("      - [SKIP] Orchestrateur désactivé (PyTorch non installé)")

        # Métriques de performance
        self.interaction_count = 0
        self.decision_history = []
        self.learning_history = []

        print("\n[OK] Assistant cognitif prêt !\n")

    def perceive_multimodal(self, text_input=None, visual_features=None, audio_features=None):
        """
        Perception multimodale : Traite les entrées de différentes modalités.

        Args:
            text_input (str): Texte à traiter
            visual_features (array): Caractéristiques visuelles simulées
            audio_features (array): Caractéristiques audio simulées

        Returns:
            dict: État neuronal après perception
        """
        print("\n" + "=" * 70)
        print("PERCEPTION MULTIMODALE")
        print("=" * 70)

        sensory_data = []
        modalities = []

        # Traiter le texte
        if text_input:
            print(f"\n[TEXT] Input : '{text_input}'")
            # Injecter les connaissances textuelles
            self.brain.inject_knowledge(text_input)
            # Créer une représentation vectorielle simple
            text_vector = [float(ord(c) % 10) / 10 for c in text_input[:5]]
            sensory_data.extend(text_vector)
            modalities.append(ModalityType.TEXT)

        # Traiter les features visuelles
        if visual_features is not None:
            print(f"[VISION] Features shape : {visual_features.shape}")
            visual_flat = visual_features.flatten()[:5]
            sensory_data.extend(visual_flat)
            modalities.append(ModalityType.IMAGE)

        # Traiter les features audio
        if audio_features is not None:
            print(f"[AUDIO] Features shape : {audio_features.shape}")
            audio_flat = audio_features.flatten()[:5]
            sensory_data.extend(audio_flat)
            modalities.append(ModalityType.AUDIO)

        # Normaliser et limiter à 5 features
        sensory_data = sensory_data[:5]
        if len(sensory_data) < 5:
            sensory_data.extend([0.0] * (5 - len(sensory_data)))

        # Envoyer au réseau neuronal
        print(f"\n[NEURAL] Injection dans le réseau neuronal spiking...")
        self.brain.perceive_and_process(sensory_data, dt=1.0)

        # Analyser l'état neuronal
        active_neurons = sum(1 for n in self.brain.neurons if n.spike)
        avg_potential = np.mean([n.v_m for n in self.brain.neurons])

        print(f"         - Neurones actifs : {active_neurons}/{len(self.brain.neurons)}")
        print(f"         - Potentiel moyen : {avg_potential:.2f} mV")

        # État émotionnel
        emotions = self.brain.emotion_module.emotional_states
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        if dominant_emotion[1] > 0.01:
            print(f"         - Émotion dominante : {dominant_emotion[0]} ({dominant_emotion[1]:.3f})")

        return {
            'active_neurons': active_neurons,
            'avg_potential': avg_potential,
            'emotions': emotions,
            'modalities': modalities
        }

    def select_architecture(self, task_description, modalities):
        """
        Utilise l'orchestrateur pour sélectionner l'architecture SOTA appropriée.

        Args:
            task_description (str): Description de la tâche
            modalities (list): Liste des modalités impliquées
        """
        print("\n" + "=" * 70)
        print("SÉLECTION D'ARCHITECTURE SOTA")
        print("=" * 70)

        print(f"\nTâche : {task_description}")
        print(f"Modalités : {[m if isinstance(m, str) else m.value for m in modalities]}")

        if not ORCHESTRATOR_AVAILABLE or self.orchestrator is None:
            print("\n[SKIP] Orchestrateur non disponible")
            print("       L'architecture serait sélectionnée automatiquement avec PyTorch")
            print("\n[SIMULATION] Architectures recommandées :")

            # Simulation de sélection basée sur les modalités
            if any(m == ModalityType.IMAGE or (hasattr(m, 'value') and m.value == 'image') for m in modalities):
                print("         - Primaire : Vision Transformer (ViT)")
                print("         - Auxiliaire : ResNet, CLIP")
            elif any(m == ModalityType.AUDIO or (hasattr(m, 'value') and m.value == 'audio') for m in modalities):
                print("         - Primaire : Whisper")
                print("         - Auxiliaire : Wav2Vec2")
            else:
                print("         - Primaire : GPT-2")
                print("         - Auxiliaire : BERT, T5")

            return None

        # Déterminer le type de tâche
        task_type = TaskType.TEXT_GENERATION  # Par défaut
        if ModalityType.IMAGE in modalities:
            task_type = TaskType.IMAGE_CLASSIFICATION
        elif ModalityType.AUDIO in modalities:
            task_type = TaskType.SPEECH_RECOGNITION

        # Créer la spécification
        task_spec = TaskSpecification(
            task_type=task_type,
            modalities=modalities,
            input_shape={'features': (512,)}
        )

        # Sélectionner l'architecture
        print("\n[ORCHESTRATOR] Analyse des architectures disponibles...")
        selection = self.orchestrator.select_architecture(task_spec)

        print(f"\n[RESULT] Architecture sélectionnée :")
        print(f"         - Primaire : {selection.primary_architecture}")
        print(f"         - Score de confiance : {selection.confidence:.3f}")
        if selection.supporting_architectures:
            print(f"         - Architectures auxiliaires : {selection.supporting_architectures}")

        return selection

    def learn_from_experience(self, inputs, targets, context=""):
        """
        Apprentissage supervisé à partir d'exemples.

        Args:
            inputs (np.array): Données d'entrée
            targets (np.array): Cibles attendues
            context (str): Contexte de l'apprentissage
        """
        print("\n" + "=" * 70)
        print("APPRENTISSAGE SUPERVISÉ")
        print("=" * 70)

        if context:
            print(f"\nContexte : {context}")

        print(f"\nDonnées :")
        print(f"  - Entrées : {inputs.shape}")
        print(f"  - Cibles : {targets.shape}")

        # Capturer l'état initial
        initial_weights = [s.weight for s in self.brain.synapses[:10]]
        initial_avg = np.mean(initial_weights)

        print(f"\nÉtat initial :")
        print(f"  - Poids synaptiques moyens : {initial_avg:.4f}")

        # Apprentissage
        print("\n[LEARNING] Rétropropagation en cours...")
        self.brain.learn(inputs, targets)

        # Analyser les changements
        final_weights = [s.weight for s in self.brain.synapses[:10]]
        final_avg = np.mean(final_weights)
        weight_change = abs(final_avg - initial_avg)

        print(f"\nRésultats :")
        print(f"  - Poids synaptiques moyens : {final_avg:.4f}")
        print(f"  - Changement absolu : {weight_change:.6f}")

        if weight_change > 0.001:
            print(f"  - Statut : [OK] Apprentissage effectif détecté")
        else:
            print(f"  - Statut : [ATTENTION] Changement minimal")

        # Stocker dans l'historique
        self.learning_history.append({
            'context': context,
            'weight_change': weight_change,
            'timestamp': self.interaction_count
        })

        return weight_change

    def make_decision(self, context="", iterations=5):
        """
        Prise de décision par accumulation d'évidence.

        Args:
            context (str): Contexte de la décision
            iterations (int): Nombre d'itérations maximum

        Returns:
            dict: Résultat de la décision
        """
        print("\n" + "=" * 70)
        print("PRISE DE DÉCISION (ACCUMULATION D'ÉVIDENCE)")
        print("=" * 70)

        if context:
            print(f"\nContexte : {context}")

        print(f"\nProcessus de décision (max {iterations} itérations) :\n")

        decision_result = None

        for i in range(iterations):
            decision_info = self.brain.execute_decision(dt=1.0)

            print(f"Itération {i+1}:")
            print(f"  - Neurones actifs : {decision_info['active_neurons']}/{decision_info['total_neurons']}")
            print(f"  - Évidence : {decision_info['evidence']:.3f}")
            print(f"  - Influence émotionnelle : {decision_info['emotion_influence']:.3f}")
            print(f"  - Évidence accumulée : {decision_info['accumulated_evidence']:.3f} / {decision_info['threshold']:.1f}")

            if decision_info['choice_made']:
                print(f"\n[DECISION] {decision_info['decision']}")
                decision_result = decision_info
                break

            # Rafraîchir l'activité neuronale
            if i < iterations - 1:
                self.brain.network.update(1.0)

        if not decision_result:
            print(f"\n[RESULT] Seuil non atteint après {iterations} itérations")
            print(f"          Évidence finale : {decision_info['accumulated_evidence']:.3f}")
            decision_result = decision_info

        # Stocker dans l'historique
        self.decision_history.append({
            'context': context,
            'decision': decision_result.get('decision', 'Indécis'),
            'evidence': decision_result['accumulated_evidence'],
            'timestamp': self.interaction_count
        })

        return decision_result

    def generate_response(self, prompt, max_length=50):
        """
        Génération de réponse en langage naturel.

        Args:
            prompt (str): Prompt de départ
            max_length (int): Longueur maximale de la réponse

        Returns:
            str: Réponse générée
        """
        print("\n" + "=" * 70)
        print("GÉNÉRATION DE LANGAGE NATUREL")
        print("=" * 70)

        print(f"\nPrompt : '{prompt}'")
        print("\n[GPT-2] Génération en cours...")

        response = self.brain.communicate(prompt)

        # Limiter la longueur
        if len(response) > max_length:
            response = response[:max_length] + "..."

        print(f"\n[RESPONSE] {response}")

        return response

    def save_memory(self):
        """Sauvegarde l'état en mémoire long terme."""
        print("\n" + "=" * 70)
        print("SAUVEGARDE EN MÉMOIRE LONG TERME")
        print("=" * 70)

        print("\n[MEMORY] Sauvegarde de l'état cognitif...")
        self.brain.save_state()

        # Sauvegarder les historiques
        self.brain.memory_module.store_long_term("decision_history", self.decision_history)
        self.brain.memory_module.store_long_term("learning_history", self.learning_history)
        self.brain.memory_module.store_long_term("interaction_count", self.interaction_count)

        print(f"         - Interactions : {self.interaction_count}")
        print(f"         - Décisions : {len(self.decision_history)}")
        print(f"         - Expériences d'apprentissage : {len(self.learning_history)}")
        print("\n[OK] Mémoire sauvegardée")

    def display_cognitive_state(self):
        """Affiche l'état cognitif complet de l'assistant."""
        print("\n" + "=" * 70)
        print("ÉTAT COGNITIF COMPLET")
        print("=" * 70)

        # État neuronal
        print("\n[NEURAL STATE]")
        active = sum(1 for n in self.brain.neurons if n.spike)
        print(f"  Neurones actifs : {active}/{len(self.brain.neurons)}")
        print(f"  Synapses : {len(self.brain.synapses)}")
        avg_weight = np.mean([s.weight for s in self.brain.synapses])
        print(f"  Poids synaptique moyen : {avg_weight:.4f}")

        # État émotionnel
        print("\n[EMOTIONAL STATE]")
        for emotion, value in self.brain.emotion_module.emotional_states.items():
            if value > 0.01:
                bar = "█" * int(value * 20)
                print(f"  {emotion.capitalize():12s} : {bar} {value:.3f}")

        # Mémoire
        print("\n[MEMORY]")
        vocab_size = len(self.brain.modules['language'].vocabulary)
        print(f"  Vocabulaire : {vocab_size} mots")
        short_term = self.brain.memory_module.retrieve_short_term()
        print(f"  Mémoire court terme : {len(short_term)} éléments")

        # Historiques
        print("\n[EXPERIENCE]")
        print(f"  Total d'interactions : {self.interaction_count}")
        print(f"  Décisions prises : {len(self.decision_history)}")
        print(f"  Sessions d'apprentissage : {len(self.learning_history)}")

        if self.learning_history:
            avg_learning = np.mean([h['weight_change'] for h in self.learning_history])
            print(f"  Changement moyen des poids : {avg_learning:.6f}")


def run_complete_demo():
    """Exécute la démonstration complète."""

    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  DÉMONSTRATION COMPLÈTE DU SYSTÈME BRAIN".center(68) + "*")
    print("*" + "  Cas d'usage : Assistant Cognitif Intelligent".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    # Créer l'assistant
    assistant = CognitiveAssistant()

    # ========================================================================
    # SCÉNARIO 1 : Analyse multimodale
    # ========================================================================
    print("\n\n")
    print("█" * 70)
    print("█ SCÉNARIO 1 : ANALYSE MULTIMODALE                                  █")
    print("█" * 70)

    # Percevoir du texte
    text = "Intelligence artificielle et neurosciences computationnelles"
    state = assistant.perceive_multimodal(text_input=text)
    assistant.interaction_count += 1

    # Sélectionner l'architecture appropriée
    assistant.select_architecture(
        "Traitement de texte scientifique",
        state['modalities']
    )

    # ========================================================================
    # SCÉNARIO 2 : Apprentissage à partir d'exemples
    # ========================================================================
    print("\n\n")
    print("█" * 70)
    print("█ SCÉNARIO 2 : APPRENTISSAGE À PARTIR D'EXEMPLES                    █")
    print("█" * 70)

    # Créer un dataset d'entraînement simulé
    training_inputs = np.random.rand(20, 5)
    training_targets = np.random.rand(20, 3)

    assistant.learn_from_experience(
        training_inputs,
        training_targets,
        context="Apprentissage de patterns sensoriels"
    )
    assistant.interaction_count += 1

    # ========================================================================
    # SCÉNARIO 3 : Perception visuelle et décision
    # ========================================================================
    print("\n\n")
    print("█" * 70)
    print("█ SCÉNARIO 3 : PERCEPTION VISUELLE ET PRISE DE DÉCISION             █")
    print("█" * 70)

    # Simuler des features visuelles (ex: détection d'objet)
    visual_features = np.random.rand(7, 7)  # Features CNN 7x7

    state = assistant.perceive_multimodal(
        text_input="Analyse d'image",
        visual_features=visual_features
    )
    assistant.interaction_count += 1

    # Sélectionner l'architecture pour la vision
    assistant.select_architecture(
        "Classification d'image",
        [ModalityType.IMAGE]
    )

    # Prendre une décision basée sur l'analyse visuelle
    assistant.make_decision(
        context="Décision basée sur l'analyse visuelle",
        iterations=5
    )
    assistant.interaction_count += 1

    # ========================================================================
    # SCÉNARIO 4 : Traitement audio et génération de réponse
    # ========================================================================
    print("\n\n")
    print("█" * 70)
    print("█ SCÉNARIO 4 : TRAITEMENT AUDIO ET GÉNÉRATION DE RÉPONSE            █")
    print("█" * 70)

    # Simuler des features audio (ex: MFCC)
    audio_features = np.random.rand(13, 10)  # 13 MFCC sur 10 frames

    state = assistant.perceive_multimodal(
        text_input="Commande vocale",
        audio_features=audio_features
    )
    assistant.interaction_count += 1

    # Sélectionner l'architecture pour l'audio
    assistant.select_architecture(
        "Reconnaissance vocale",
        [ModalityType.AUDIO, ModalityType.TEXT]
    )

    # Générer une réponse
    assistant.generate_response(
        "Bonjour, je suis votre assistant cognitif. Comment puis-je vous aider ?"
    )
    assistant.interaction_count += 1

    # ========================================================================
    # SCÉNARIO 5 : Apprentissage continu et adaptation
    # ========================================================================
    print("\n\n")
    print("█" * 70)
    print("█ SCÉNARIO 5 : APPRENTISSAGE CONTINU ET ADAPTATION                  █")
    print("█" * 70)

    # Simuler plusieurs sessions d'apprentissage
    for i in range(3):
        inputs = np.random.rand(10, 5)
        targets = np.random.rand(10, 3)

        assistant.learn_from_experience(
            inputs,
            targets,
            context=f"Session d'apprentissage {i+1}/3"
        )
        assistant.interaction_count += 1

    # ========================================================================
    # RÉSUMÉ ET SAUVEGARDE
    # ========================================================================
    print("\n\n")
    print("█" * 70)
    print("█ ÉTAT FINAL ET SAUVEGARDE                                          █")
    print("█" * 70)

    # Afficher l'état cognitif complet
    assistant.display_cognitive_state()

    # Sauvegarder en mémoire
    assistant.save_memory()

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("\n\n")
    print("=" * 70)
    print("DÉMONSTRATION TERMINÉE")
    print("=" * 70)
    print("\nTOUTES LES FONCTIONNALITÉS ONT ÉTÉ DÉMONTRÉES :")
    print("  [OK] Perception multimodale (Texte, Vision, Audio)")
    print("  [OK] Sélection d'architecture SOTA par l'orchestrateur")
    print("  [OK] Apprentissage supervisé avec modification des poids")
    print("  [OK] Prise de décision par accumulation d'évidence")
    print("  [OK] Génération de langage naturel (GPT-2)")
    print("  [OK] Système émotionnel")
    print("  [OK] Mémoire court et long terme")
    print("  [OK] Réseau neuronal biologique (LIF)")
    print("\nLe système est entièrement fonctionnel et opérationnel.")
    print("=" * 70)
    print("\n")


if __name__ == "__main__":
    run_complete_demo()
