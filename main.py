#!/usr/bin/env python3
"""
Point d'entrée principal pour le système Brain Neural Network.
"""

import sys
import argparse
import numpy as np
from core.brain import Brain
from utils.logging import brain_logger, BrainLogger
from utils.exceptions import BrainException


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Brain Neural Network System')
    parser.add_argument('--neurons', type=int, default=10, help='Nombre de neurones à créer')
    parser.add_argument('--timesteps', type=int, default=100, help='Nombre de pas de temps de simulation')
    parser.add_argument('--dt', type=float, default=1.0, help='Pas de temps (ms)')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Niveau de logging')
    parser.add_argument('--demo', action='store_true', help='Exécuter une démonstration')
    parser.add_argument('--interactive', action='store_true', help='Mode interactif')
    
    args = parser.parse_args()
    
    # Configurer le logging
    import logging
    log_level = getattr(logging, args.log_level)
    brain_logger.logger.setLevel(log_level)
    
    try:
        brain_logger.info("=" * 60)
        brain_logger.info("DÉMARRAGE DU SYSTÈME BRAIN NEURAL NETWORK")
        brain_logger.info("=" * 60)
        
        # Créer l'instance du cerveau
        brain = Brain(num_neurons=args.neurons)
        
        if args.demo:
            run_demo(brain, args)
        elif args.interactive:
            run_interactive(brain)
        else:
            run_simulation(brain, args)
        
        brain_logger.info("=" * 60)
        brain_logger.info("ARRÊT DU SYSTÈME BRAIN NEURAL NETWORK")
        brain_logger.info("=" * 60)
        
    except KeyboardInterrupt:
        brain_logger.info("\nInterruption par l'utilisateur")
        sys.exit(0)
    except BrainException as e:
        brain_logger.error(f"Erreur Brain: {str(e)}")
        sys.exit(1)
    except Exception as e:
        brain_logger.exception(f"Erreur inattendue: {str(e)}")
        sys.exit(1)


def run_simulation(brain, args):
    """
    Exécute une simulation du réseau neuronal.
    
    Args:
        brain (Brain): Instance du cerveau.
        args: Arguments de ligne de commande.
    """
    brain_logger.info(f"Démarrage de la simulation ({args.timesteps} pas de temps, dt={args.dt}ms)")
    
    for t in range(args.timesteps):
        # Entrées sensorielles aléatoires
        sensory_input = np.random.randn(args.neurons) * 0.1
        
        # Perception et traitement
        brain.perceive_and_process(sensory_input, args.dt)
        
        # Exécuter une décision
        brain.execute_decision(args.dt)
        
        # Afficher l'état tous les 20 pas de temps
        if t % 20 == 0:
            active_neurons = sum(1 for n in brain.neurons if n.spike)
            brain_logger.debug(f"t={t}: {active_neurons}/{args.neurons} neurones actifs")
    
    brain_logger.info("Simulation terminée")
    brain.save_state()


def run_demo(brain, args):
    """
    Exécute une démonstration des capacités du cerveau.
    
    Args:
        brain (Brain): Instance du cerveau.
        args: Arguments de ligne de commande.
    """
    brain_logger.info("=== DÉMONSTRATION DES CAPACITÉS ===")
    
    # 1. Test de perception
    brain_logger.info("\n1. Test de perception")
    test_input = {'visual': 1.0, 'auditory': 0.5}
    result = brain.modules['perception'].process(test_input)
    brain_logger.info(f"   Entrée: {test_input}")
    brain_logger.info(f"   Résultat: {result}")
    
    # 2. Test de langage
    brain_logger.info("\n2. Test de génération de langage")
    prompt = "The brain is"
    try:
        sentence = brain.communicate(prompt)
        brain_logger.info(f"   Prompt: '{prompt}'")
        brain_logger.info(f"   Généré: '{sentence}'")
    except Exception as e:
        brain_logger.warning(f"   Erreur lors de la génération: {str(e)}")
    
    # 3. Test de mémoire
    brain_logger.info("\n3. Test de mémoire")
    brain.memory_module.store_short_term("Souvenir 1")
    brain.memory_module.store_short_term("Souvenir 2")
    brain.memory_module.store_long_term("knowledge", "Information importante")
    memories = brain.memory_module.retrieve_short_term()
    knowledge = brain.memory_module.retrieve_long_term("knowledge")
    brain_logger.info(f"   Mémoires à court terme: {memories}")
    brain_logger.info(f"   Connaissance stockée: {knowledge}")
    
    # 4. Test d'apprentissage
    brain_logger.info("\n4. Test d'apprentissage supervisé")
    inputs = np.random.rand(args.neurons)
    targets = np.random.rand(args.neurons)
    brain.learn(inputs, targets)
    brain_logger.info("   Apprentissage effectué")
    
    # 5. Test d'émotions
    brain_logger.info("\n5. Test du système émotionnel")
    brain.emotion_module.update_emotions(
        {'threat': True}, 
        brain.memory_module.retrieve_short_term(),
        reward=0.5,
        dt=1.0
    )
    brain_logger.info(f"   États émotionnels: {brain.emotion_module.emotional_states}")
    
    brain_logger.info("\n=== FIN DE LA DÉMONSTRATION ===")


def run_interactive(brain):
    """
    Mode interactif pour interagir avec le cerveau.
    
    Args:
        brain (Brain): Instance du cerveau.
    """
    brain_logger.info("=== MODE INTERACTIF ===")
    brain_logger.info("Commandes disponibles:")
    brain_logger.info("  - 'generate <prompt>': Générer du texte")
    brain_logger.info("  - 'learn <text>': Apprendre du texte")
    brain_logger.info("  - 'memory': Afficher les mémoires")
    brain_logger.info("  - 'emotions': Afficher les émotions")
    brain_logger.info("  - 'simulate <n>': Simuler n pas de temps")
    brain_logger.info("  - 'quit': Quitter")
    
    while True:
        try:
            command = input("\nBrain> ").strip()
            
            if not command:
                continue
            
            if command == 'quit':
                break
            
            parts = command.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd == 'generate':
                if not arg:
                    print("Usage: generate <prompt>")
                    continue
                try:
                    sentence = brain.communicate(arg)
                    print(f"Généré: {sentence}")
                except Exception as e:
                    print(f"Erreur: {str(e)}")
            
            elif cmd == 'learn':
                if not arg:
                    print("Usage: learn <text>")
                    continue
                try:
                    brain.inject_knowledge(arg)
                    print("Texte appris avec succès")
                except Exception as e:
                    print(f"Erreur: {str(e)}")
            
            elif cmd == 'memory':
                st_mem = brain.memory_module.retrieve_short_term()
                print(f"Mémoire à court terme: {st_mem}")
            
            elif cmd == 'emotions':
                print(f"États émotionnels: {brain.emotion_module.emotional_states}")
            
            elif cmd == 'simulate':
                try:
                    n = int(arg) if arg else 10
                    for _ in range(n):
                        sensory_input = np.random.randn(brain.num_neurons) * 0.1
                        brain.perceive_and_process(sensory_input, 1.0)
                    print(f"{n} pas de temps simulés")
                except ValueError:
                    print("Usage: simulate <nombre>")
            
            else:
                print(f"Commande inconnue: {cmd}")
        
        except KeyboardInterrupt:
            print("\nUtilisez 'quit' pour quitter")
        except Exception as e:
            print(f"Erreur: {str(e)}")
    
    brain_logger.info("Fin du mode interactif")


if __name__ == '__main__':
    main()
