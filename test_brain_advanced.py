#!/usr/bin/env python3
"""
Exemples avancés pour tester l'orchestrateur intelligent du Brain
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch

def test_orchestrator_basic():
    """Test basique de l'orchestrateur"""
    print("=" * 60)
    print("Test Orchestrateur - Sélection automatique d'architecture")
    print("=" * 60)

    try:
        from core.orchestrator import IntelligentOrchestrator, TaskSpecification, TaskType, ModalityType

        # Initialiser l'orchestrateur
        orchestrator = IntelligentOrchestrator()
        print("✓ Orchestrateur initialisé")

        # Définir une tâche de question-réponse visuelle
        task = TaskSpecification(
            task_type=TaskType.VISUAL_QUESTION_ANSWERING,
            modalities=[ModalityType.TEXT, ModalityType.IMAGE],
            input_shape={'image': (3, 224, 224), 'text': (512,)}
        )
        print("✓ Tâche définie: Visual Question Answering")

        # Sélection automatique d'architecture
        selection = orchestrator.select_architecture(task)
        print(f"✓ Architecture sélectionnée: {selection.primary_architecture}")
        print(f"✓ Raisonnement: {selection.reasoning}")

        # Test avec des données simulées
        inputs = {
            'image': torch.randn(1, 3, 224, 224),
            'text': "Qu'y a-t-il dans cette image?"
        }

        print("✓ Exécution de l'inférence...")
        # Note: Cette partie peut échouer si les architectures ne sont pas complètement implémentées
        try:
            output = orchestrator.forward(inputs, task)
            print(f"✓ Sortie: {type(output)}")
        except Exception as e:
            print(f"⚠ Inférence non disponible: {e}")
            print("  (Ceci est normal si les architectures ne sont pas toutes implémentées)")

    except ImportError as e:
        print(f"✗ Module non disponible: {e}")
    except Exception as e:
        print(f"✗ Erreur orchestrateur: {e}")

def test_clip_example():
    """Test de CLIP pour la classification zero-shot"""
    print("\n" + "=" * 60)
    print("Test CLIP - Classification zero-shot")
    print("=" * 60)

    try:
        from architectures.multimodal.clip import CLIPModel

        # Créer le modèle CLIP
        model = CLIPModel(
            image_size=224,
            patch_size=16,
            hidden_size=512,
            num_heads=8,
            num_layers=12,
            vocab_size=49408,
            max_text_length=77,
        )
        print("✓ Modèle CLIP créé")

        # Données de test
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224)
        text_tokens = torch.randint(0, 49408, (batch_size, 77))

        print(f"✓ Image shape: {image.shape}")
        print(f"✓ Text tokens shape: {text_tokens.shape}")

        # Forward pass
        with torch.no_grad():
            outputs = model(image, text_tokens)

        print("✓ Forward pass réussi")
        print(f"  - Image embeddings: {outputs['image_embeds'].shape}")
        print(f"  - Text embeddings: {outputs['text_embeds'].shape}")
        print(".4f"
    except ImportError as e:
        print(f"✗ CLIP non disponible: {e}")
    except Exception as e:
        print(f"✗ Erreur CLIP: {e}")

def test_whisper_example():
    """Test de Whisper pour la reconnaissance vocale"""
    print("\n" + "=" * 60)
    print("Test Whisper - Reconnaissance vocale")
    print("=" * 60)

    try:
        from architectures.audio.whisper import Whisper, WhisperConfig

        # Configuration
        config = WhisperConfig(model_size='tiny')  # Utiliser tiny pour les tests
        model = Whisper(config)
        print("✓ Modèle Whisper créé")

        # Audio simulé (spectrogramme Mel)
        # Whisper attend: (batch_size, 80, time)
        audio = torch.randn(1, 80, 300)  # ~3 secondes à 100Hz

        print(f"✓ Audio shape: {audio.shape}")

        # Transcription
        transcription = model.generate(
            audio,
            task='transcribe',
            language='fr'
        )

        print("✓ Transcription réussie")
        print(f"  - Résultat: '{transcription}'")

    except ImportError as e:
        print(f"✗ Whisper non disponible: {e}")
    except Exception as e:
        print(f"✗ Erreur Whisper: {e}")

def test_yolo_example():
    """Test de YOLO pour la détection d'objets"""
    print("\n" + "=" * 60)
    print("Test YOLO - Détection d'objets")
    print("=" * 60)

    try:
        from architectures.computer_vision.yolo import YOLOv8, YOLOv8Config

        # Configuration
        config = YOLOv8Config(
            model_size='n',  # nano pour les tests
            num_classes=80,
            image_size=640
        )
        model = YOLOv8(config)
        print("✓ Modèle YOLOv8 créé")

        # Images de test
        images = torch.randn(1, 3, 640, 640)
        print(f"✓ Images shape: {images.shape}")

        # Prédiction
        with torch.no_grad():
            detections = model.predict(images, conf_threshold=0.25)

        print("✓ Détection réussie")
        print(f"  - Nombre d'images: {len(detections)}")
        if detections:
            print(f"  - Objets détectés dans l'image 0: {len(detections[0]['boxes'])}")

    except ImportError as e:
        print(f"✗ YOLO non disponible: {e}")
    except Exception as e:
        print(f"✗ Erreur YOLO: {e}")

def test_musicgen_example():
    """Test de MusicGen pour la génération de musique"""
    print("\n" + "=" * 60)
    print("Test MusicGen - Génération de musique")
    print("=" * 60)

    try:
        from architectures.audio.musicgen import MusicGen, MusicGenConfig

        # Configuration
        config = MusicGenConfig(
            num_codebooks=4,
            vocab_size=2048
        )
        model = MusicGen(config)
        print("✓ Modèle MusicGen créé")

        # Texte descriptif
        text_descriptions = ["musique électronique entraînante"]
        print(f"✓ Description: {text_descriptions[0]}")

        # Génération (courte pour le test)
        music = model.generate(
            text=text_descriptions,
            duration=5.0,  # 5 secondes seulement
            temperature=1.0
        )

        print("✓ Génération musicale réussie")
        print(f"  - Shape audio: {music.shape}")

    except ImportError as e:
        print(f"✗ MusicGen non disponible: {e}")
    except Exception as e:
        print(f"✗ Erreur MusicGen: {e}")

def test_maml_example():
    """Test de MAML pour l'apprentissage meta"""
    print("\n" + "=" * 60)
    print("Test MAML - Apprentissage meta (Few-shot)")
    print("=" * 60)

    try:
        import torch.nn as nn
        from architectures.meta_learning.maml import MAML, MAMLConfig

        # Modèle simple
        base_model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Classification binaire
        )

        # Configuration MAML
        config = MAMLConfig(
            inner_lr=0.01,
            inner_steps=3,  # Peu d'étapes pour le test
            meta_lr=0.001
        )

        maml = MAML(base_model, config)
        print("✓ MAML initialisé")

        # Données few-shot
        support_x = torch.randn(5, 10)  # 5 exemples
        support_y = torch.randint(0, 2, (5,))
        query_x = torch.randn(10, 10)   # 10 queries

        print(f"✓ Support set: {support_x.shape}")
        print(f"✓ Query set: {query_x.shape}")

        # Adaptation
        adapted_model = maml.adapt(support_x, support_y)
        print("✓ Adaptation réussie")

        # Prédiction
        with torch.no_grad():
            predictions = adapted_model(query_x)

        print("✓ Prédictions réussies")
        print(f"  - Shape prédictions: {predictions.shape}")

    except ImportError as e:
        print(f"✗ MAML non disponible: {e}")
    except Exception as e:
        print(f"✗ Erreur MAML: {e}")

def main():
    """Fonction principale"""
    print("🧠 Tests Avancés du système Brain")
    print("Tests des architectures SOTA individuelles")
    print()

    # Vérifier PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    print()

    # Tests des différentes architectures
    test_orchestrator_basic()
    test_clip_example()
    test_whisper_example()
    test_yolo_example()
    test_musicgen_example()
    test_maml_example()

    print("\n" + "=" * 60)
    print("🎉 Tests avancés terminés!")
    print("=" * 60)
    print("\nPour plus d'exemples:")
    print("- Consultez le dossier examples/")
    print("- Lisez USAGE_GUIDE.md")
    print("- Vérifiez IMPLEMENTATION_STATUS.md")

if __name__ == "__main__":
    main()