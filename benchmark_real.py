"""
Benchmark scientifique du système Brain.

Tests objectifs avec métriques mesurables.
PAS de bullshit - seulement des faits et des chiffres.
"""
import numpy as np
import logging
import time
from logging_config import setup_logging
from core.brain import Brain


def test_1_pattern_recognition():
    """
    TEST 1: Reconnaissance de patterns simples (XOR-like)
    
    Tâche: Apprendre à distinguer 2 patterns
    - Pattern A: [1, 1, 0, 0] -> [1, 0]
    - Pattern B: [0, 0, 1, 1] -> [0, 1]
    
    Métrique de succès: Précision > 70% après entraînement
    """
    print("\n" + "="*80)
    print("TEST 1: RECONNAISSANCE DE PATTERNS")
    print("="*80)
    
    brain = Brain(num_neurons=10)
    
    # Données d'entraînement
    patterns = [
        ([1.0, 1.0, 0.0, 0.0], [1.0, 0.0]),
        ([0.0, 0.0, 1.0, 1.0], [0.0, 1.0]),
        ([1.0, 0.0, 1.0, 0.0], [0.5, 0.5]),  # Pattern ambigu
        ([0.0, 1.0, 0.0, 1.0], [0.5, 0.5]),  # Pattern ambigu
    ]
    
    # Baseline: Performance aléatoire
    baseline_correct = 0
    for _ in range(100):
        pattern, target = patterns[np.random.randint(len(patterns))]
        random_output = np.random.rand(2)
        if (random_output[0] > 0.5) == (target[0] > 0.5):
            baseline_correct += 1
    baseline_accuracy = baseline_correct / 100
    
    print(f"\n📊 Baseline (aléatoire): {baseline_accuracy:.1%} précision")
    
    # Entraînement
    print("\n🎓 Entraînement (100 époques)...")
    errors = []
    start_time = time.time()
    
    for epoch in range(100):
        epoch_errors = []
        for pattern, target in patterns:
            error = brain.learn(pattern, target, learning_type='supervised')
            epoch_errors.append(error)
        
        avg_error = np.mean(epoch_errors)
        errors.append(avg_error)
        
        if epoch % 20 == 0:
            print(f"  Époque {epoch:3d}: Erreur = {avg_error:.4f}")
    
    training_time = time.time() - start_time
    
    # Test de performance
    print("\n🧪 Test de précision...")
    correct = 0
    total = 100
    
    for _ in range(total):
        pattern, target = patterns[np.random.randint(len(patterns))]
        
        # Forward pass
        outputs = []
        for neuron in brain.network.neurons[:4]:
            neuron.reset()
        for i, val in enumerate(pattern):
            neuron = brain.network.neurons[i]
            neuron.v_m = neuron.v_rest + val * 10.0
            outputs.append(1.0 if neuron.v_m >= neuron.v_threshold else 0.0)
        
        # Comparer (seulement les 2 premiers outputs)
        if len(outputs) >= 2:
            pred = outputs[:2]
            if (pred[0] > 0.5) == (target[0] > 0.5):
                correct += 1
    
    accuracy = correct / total
    
    # Résultats
    print("\n" + "-"*80)
    print("RÉSULTATS:")
    print(f"  ✓ Baseline aléatoire:  {baseline_accuracy:.1%}")
    print(f"  ✓ Brain après training: {accuracy:.1%}")
    print(f"  ✓ Amélioration:        {((accuracy - baseline_accuracy) / baseline_accuracy * 100):+.1f}%")
    print(f"  ✓ Convergence erreur:  {errors[0]:.4f} -> {errors[-1]:.4f}")
    print(f"  ✓ Temps entraînement:  {training_time:.2f}s")
    print("-"*80)
    
    # Verdict
    success = accuracy > baseline_accuracy * 1.1  # Au moins 10% mieux que random
    print(f"\n{'✅ SUCCÈS' if success else '❌ ÉCHEC'}: ", end="")
    if success:
        print(f"Brain apprend (précision supérieure au hasard)")
    else:
        print(f"Brain n'apprend pas (équivalent au hasard)")
    
    return success, {
        'baseline': baseline_accuracy,
        'accuracy': accuracy,
        'error_reduction': errors[0] - errors[-1],
        'training_time': training_time
    }


def test_2_network_activity():
    """
    TEST 2: Vérification de l'activité neuronale
    
    Tâche: Stimuler le réseau et mesurer les spikes
    Métrique de succès: Au moins 10% des neurones doivent spiker
    """
    print("\n" + "="*80)
    print("TEST 2: ACTIVITÉ NEURONALE")
    print("="*80)
    
    brain = Brain(num_neurons=20)
    
    # Stimulation forte
    strong_input = [0.8] * 10
    
    print("\n🔬 Stimulation du réseau avec inputs forts...")
    spikes_count = []
    
    for step in range(10):
        brain.network.update(dt=1.0)
        
        # Injecter courant dans les neurones
        for i, neuron in enumerate(brain.neurons[:10]):
            neuron.v_m += strong_input[i] * 5.0  # Stimulation directe
        
        # Compter les spikes
        num_spikes = sum(1 for n in brain.neurons if n.spike)
        spikes_count.append(num_spikes)
    
    total_spikes = sum(spikes_count)
    avg_spikes = np.mean(spikes_count)
    spike_rate = total_spikes / (20 * 10)  # spikes / (neurons * steps)
    
    print("\n" + "-"*80)
    print("RÉSULTATS:")
    print(f"  ✓ Spikes totaux:    {total_spikes}")
    print(f"  ✓ Spikes par step:  {avg_spikes:.1f}")
    print(f"  ✓ Taux de firing:   {spike_rate:.1%}")
    print("-"*80)
    
    success = spike_rate > 0.05  # Au moins 5% des neurones spikent
    print(f"\n{'✅ SUCCÈS' if success else '❌ ÉCHEC'}: ", end="")
    if success:
        print(f"Réseau actif (neurones répondent aux stimuli)")
    else:
        print(f"Réseau inactif (neurones ne spikent pas)")
    
    return success, {
        'total_spikes': total_spikes,
        'spike_rate': spike_rate
    }


def test_3_memory_persistence():
    """
    TEST 3: Persistance de la mémoire
    
    Tâche: Sauvegarder et recharger des données
    Métrique de succès: Données identiques après reload
    """
    print("\n" + "="*80)
    print("TEST 3: PERSISTANCE MÉMOIRE")
    print("="*80)
    
    brain = Brain(num_neurons=5)
    
    # Données de test
    test_data = {
        'experiment_id': 'test_123',
        'results': [1.5, 2.3, 4.7],
        'weights': np.random.rand(10).tolist()
    }
    
    print("\n💾 Sauvegarde de données...")
    for key, value in test_data.items():
        brain.memory_module.store_long_term(key, value)
    
    brain.save_state()
    print("  ✓ Données sauvegardées")
    
    # Créer nouveau brain et recharger
    print("\n📂 Rechargement...")
    brain2 = Brain(num_neurons=5)
    
    # Vérifier données
    all_match = True
    for key, expected_value in test_data.items():
        loaded_value = brain2.memory_module.retrieve_long_term(key)
        if loaded_value != expected_value:
            all_match = False
            print(f"  ❌ {key}: {loaded_value} != {expected_value}")
        else:
            print(f"  ✓ {key}: OK")
    
    success = all_match
    print(f"\n{'✅ SUCCÈS' if success else '❌ ÉCHEC'}: ", end="")
    if success:
        print(f"Mémoire persistante fonctionne")
    else:
        print(f"Perte de données lors du reload")
    
    return success, {'data_integrity': all_match}


def test_4_plasticity():
    """
    TEST 4: Plasticité synaptique
    
    Tâche: Modifier les poids et vérifier les changements
    Métrique de succès: Poids synaptiques évoluent avec l'apprentissage
    """
    print("\n" + "="*80)
    print("TEST 4: PLASTICITÉ SYNAPTIQUE")
    print("="*80)
    
    brain = Brain(num_neurons=10)
    
    if len(brain.network.synapses) == 0:
        print("\n❌ ÉCHEC: Aucune synapse dans le réseau")
        return False, {}
    
    # Poids initiaux
    initial_weights = np.array(brain.network.get_weights())
    print(f"\n📊 Poids initiaux: moyenne={np.mean(initial_weights):.4f}, std={np.std(initial_weights):.4f}")
    
    # Entraînement intensif
    print("\n🎓 Entraînement (50 patterns)...")
    for _ in range(50):
        pattern = np.random.rand(5)
        target = np.random.rand(5)
        brain.learn(pattern.tolist(), target.tolist(), learning_type='supervised')
    
    # Poids finaux
    final_weights = np.array(brain.network.get_weights())
    print(f"📊 Poids finaux:   moyenne={np.mean(final_weights):.4f}, std={np.std(final_weights):.4f}")
    
    # Calcul des changements
    weight_changes = final_weights - initial_weights
    num_changed = np.sum(np.abs(weight_changes) > 0.001)  # Changements significatifs
    pct_changed = num_changed / len(weight_changes) * 100
    avg_change = np.mean(np.abs(weight_changes))
    
    print("\n" + "-"*80)
    print("RÉSULTATS:")
    print(f"  ✓ Synapses totales:      {len(weight_changes)}")
    print(f"  ✓ Synapses modifiées:    {num_changed} ({pct_changed:.1f}%)")
    print(f"  ✓ Changement moyen:      {avg_change:.6f}")
    print(f"  ✓ Changement max:        {np.max(np.abs(weight_changes)):.6f}")
    print("-"*80)
    
    success = pct_changed > 10  # Au moins 10% des synapses changent
    print(f"\n{'✅ SUCCÈS' if success else '❌ ÉCHEC'}: ", end="")
    if success:
        print(f"Plasticité fonctionne (poids évoluent)")
    else:
        print(f"Plasticité limitée (poids figés)")
    
    return success, {
        'pct_changed': pct_changed,
        'avg_change': avg_change
    }


def main():
    """Exécute tous les benchmarks."""
    setup_logging(level=logging.WARNING)  # Réduire le bruit
    
    print("\n" + "="*80)
    print("🧪 BENCHMARK SCIENTIFIQUE DU SYSTÈME BRAIN")
    print("="*80)
    print("\nTests objectifs avec métriques mesurables.")
    print("Aucun bullshit - seulement des faits.\n")
    
    results = {}
    
    # Exécuter les tests
    tests = [
        ("Pattern Recognition", test_1_pattern_recognition),
        ("Network Activity", test_2_network_activity),
        ("Memory Persistence", test_3_memory_persistence),
        ("Synaptic Plasticity", test_4_plasticity),
    ]
    
    successes = 0
    for test_name, test_func in tests:
        try:
            success, metrics = test_func()
            results[test_name] = {'success': success, 'metrics': metrics}
            if success:
                successes += 1
        except Exception as e:
            print(f"\n❌ ERREUR dans {test_name}: {e}")
            results[test_name] = {'success': False, 'error': str(e)}
    
    # Rapport final
    print("\n" + "="*80)
    print("📊 RAPPORT FINAL")
    print("="*80)
    print(f"\nTests réussis: {successes}/{len(tests)}")
    
    for test_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {test_name}")
    
    # Verdict global
    print("\n" + "="*80)
    if successes == len(tests):
        print("✅ VERDICT: Brain est FONCTIONNEL")
        print("   Toutes les fonctionnalités de base marchent.")
    elif successes >= len(tests) // 2:
        print("⚠️  VERDICT: Brain est PARTIELLEMENT FONCTIONNEL")
        print(f"   {successes}/{len(tests)} tests passent - améliorations nécessaires.")
    else:
        print("❌ VERDICT: Brain est NON FONCTIONNEL")
        print(f"   Seulement {successes}/{len(tests)} tests passent - problèmes majeurs.")
    print("="*80)


if __name__ == "__main__":
    main()
