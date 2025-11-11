#!/usr/bin/env python3
"""
Script de démonstration rapide pour tester le Brain
Utilisation: python demo_brain.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def demo_brain_basic():
    """Démonstration basique du Brain"""
    print("[START] Démonstration du système Brain")
    print("=" * 40)

    try:
        from core.brain import Brain

        # Créer le cerveau
        print("Création du cerveau...")
        brain = Brain()
        print("[OK] Cerveau initialisé avec succès!")

        # Injecter des connaissances
        print("\nApprentissage de connaissances...")
        knowledge = "Le machine learning est une branche de l'intelligence artificielle."
        brain.inject_knowledge(knowledge)
        print(f"[OK] Connaissances apprises: '{knowledge}'")

        # Tester la communication
        print("\nTest de communication...")
        prompt = "Salut"
        response = brain.communicate(prompt)
        print(f"[OK] Réponse: '{response}'")

        # Tester la perception
        print("\nTest de perception...")
        sensory_input = [0.7, 0.3, 0.9]  # Stimuli simulés
        brain.perceive_and_process(sensory_input, dt=1.0)
        print(f"[OK] Stimuli traités: {sensory_input}")

        print("\n[SUCCESS] Démonstration terminée avec succès!")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

def demo_orchestrator():
    """Démonstration de l'orchestrateur"""
    print("\n[TOOL] Démonstration de l'Orchestrateur Intelligent")
    print("=" * 50)

    try:
        from core.orchestrator import IntelligentOrchestrator, TaskSpecification, TaskType, ModalityType

        orchestrator = IntelligentOrchestrator()
        print("[OK] Orchestrateur initialisé")

        # Définir une tâche
        task = TaskSpecification(
            task_type=TaskType.TEXT_CLASSIFICATION,
            modalities=[ModalityType.TEXT],
            input_shape={'text': (512,)}
        )
        print("[OK] Tâche définie: Classification de texte")

        # Sélection d'architecture
        selection = orchestrator.select_architecture(task)
        print(f"[OK] Architecture choisie: {selection.primary_architecture}")

    except Exception as e:
        print(f"❌ Erreur orchestrateur: {e}")

def show_available_tests():
    """Afficher les tests disponibles"""
    print("\n[LIST] Tests disponibles:")
    print("-" * 30)
    print("1. python test_brain_simple.py     # Tests de base du Brain")
    print("2. python test_brain_advanced.py   # Tests des architectures SOTA")
    print("3. python demo_brain.py           # Cette démonstration")
    print("4. python examples/training_example.py     # Exemple d'entraînement")
    print("5. python examples/multimodal_example.py   # Exemples multimodaux")
    print("6. python examples/inference_example.py     # Exemples d'inférence")
    print("7. python run_all_tests.py        # Tous les tests unitaires")

def main():
    """Fonction principale"""
    print("[BRAIN] Démonstration du Framework Brain")
    print("Système d'IA avec architectures SOTA 2023-2025")
    print()

    # Vérifier les dépendances
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} détecté")
    except ImportError:
        print("[ATTENTION]️  PyTorch non installé. Installez avec: pip install torch")

    # Démonstration basique
    demo_brain_basic()

    # Démonstration orchestrateur
    demo_orchestrator()

    # Afficher les options
    show_available_tests()

    print("\n[INFO] Conseils:")
    print("- Commencez par test_brain_simple.py pour comprendre le fonctionnement de base")
    print("- Utilisez test_brain_advanced.py pour tester les architectures individuelles")
    print("- Consultez examples/ pour des exemples complets")
    print("- Lisez README.md et USAGE_GUIDE.md pour la documentation")

if __name__ == "__main__":
    main()