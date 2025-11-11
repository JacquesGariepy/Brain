"""
Brain - Artificial Brain Simulation System

Un système de cerveau artificiel complet avec:
- Réseau neuronal spiking (LIF, STDP, STP)
- Apprentissage (supervisé, non supervisé, renforcement)
- NLP complet (tokenization, POS, NER, sentiment, embeddings)
- Raisonnement logique (forward/backward chaining)
- Mémoire (court et long terme avec persistance)
- Décisions (drift-diffusion)
- Émotions (appraisal theory)
- Attention (saliency, competition)

Usage:
    from brain_api import BrainAPI

    brain = BrainAPI()
    brain.fit(X, y, epochs=10)
    predictions = brain.predict(X_test)

Voir QUICKSTART.md pour plus d'exemples.
"""

__version__ = "2.0.0"
__author__ = "Brain Development Team"

from brain_api import BrainAPI

__all__ = ['BrainAPI']
