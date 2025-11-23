# Brain - Réseau Neuronal Bio-Inspiré 🧠

Un modèle de réseau neuronal avancé inspiré des mécanismes biologiques du cerveau, implémentant des capacités cognitives sophistiquées incluant la perception, l'apprentissage, la mémoire, les émotions, et la prise de décision.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Caractéristiques Principales

### Architecture Neuronale
- **Modèle LIF (Leaky Integrate-and-Fire)**: Neurones réalistes avec dynamique membranaire
- **Plasticité Synaptique**: STDP (Spike-Timing-Dependent Plasticity)
- **Modulation Astrocytaire**: Influence gliale sur la transmission synaptique
- **Plasticité Homéostatique**: Maintien de l'équilibre neuronal

### Modules Cognitifs
- **Perception**: Traitement multi-sensoriel (visuel, auditif, tactile)
- **Attention**: Modulation dynamique de l'importance des stimuli
- **Mémoire**: Mémoire à court terme et long terme avec consolidation
- **Apprentissage**: Supervisé, non supervisé et par renforcement
- **Langage**: Génération et compréhension via modèles Transformer
- **Émotions**: Système émotionnel influençant le comportement
- **Décision**: Accumulation d'évidence pour la prise de décision
- **Raisonnement**: Inférence logique et déduction

### Fonctionnalités Avancées
- **Système de Plugins**: Architecture extensible
- **Logging Complet**: Traçabilité des opérations
- **Gestion d'Exceptions**: Hiérarchie d'exceptions personnalisées
- **Docker Support**: Déploiement containerisé
- **Tests Complets**: Suite de tests unitaires et d'intégration

## 📋 Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- (Optionnel) Docker pour le déploiement containerisé

## 🚀 Installation

### Installation Standard

```bash
# Cloner le repository
git clone https://github.com/JacquesGariepy/Brain.git
cd Brain

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Ou installer le package
pip install -e .
```

### Installation pour Développement

```bash
# Installer les dépendances de développement
pip install -r requirements-dev.txt

# Ou utiliser le Makefile
make install-dev
```

### Installation via Docker

```bash
# Construire l'image
docker build -t brain-neural-network .

# Ou utiliser docker-compose
docker-compose up -d
```

## 🎯 Utilisation

### Mode Démonstration

```bash
python main.py --demo
```

Exécute une démonstration complète des capacités du système.

### Mode Interactif

```bash
python main.py --interactive
```

Mode interactif permettant de:
- Générer du texte
- Apprendre du contenu
- Consulter la mémoire
- Observer les émotions
- Simuler le réseau

### Mode Simulation

```bash
python main.py --neurons 100 --timesteps 1000 --dt 1.0
```

Options:
- `--neurons`: Nombre de neurones (défaut: 10)
- `--timesteps`: Nombre de pas de temps (défaut: 100)
- `--dt`: Pas de temps en ms (défaut: 1.0)
- `--log-level`: Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Utilisation Programmatique

```python
from core.brain import Brain
import numpy as np

# Créer une instance du cerveau
brain = Brain(num_neurons=50)

# Perception
sensory_input = {'visual': 1.0, 'auditory': 0.5}
result = brain.modules['perception'].process(sensory_input)

# Génération de langage
sentence = brain.communicate("The brain is")
print(sentence)

# Apprentissage
inputs = np.random.rand(50)
targets = np.random.rand(50)
brain.learn(inputs, targets)

# Mémoire
brain.memory_module.store_long_term("key", "value")
value = brain.memory_module.retrieve_long_term("key")

# Sauvegarder l'état
brain.save_state()
```

## 📁 Structure du Projet

```
brain_model/
├── main.py                    # Point d'entrée principal
├── core/                      # Modules core
│   ├── __init__.py
│   ├── brain.py              # Classe principale Brain
│   ├── interfaces.py         # Interfaces abstraites
│   ├── perception.py         # Module de perception
│   ├── language.py           # Module de langage
│   └── reasoning.py          # Module de raisonnement
├── modules/                   # Modules neuronaux
│   ├── __init__.py
│   ├── neuron.py             # Classe Neuron (LIF)
│   ├── synapse.py            # Classe Synapse (STDP)
│   ├── network.py            # Réseau neuronal
│   ├── attention.py          # Module d'attention
│   ├── emotion.py            # Gestion des émotions
│   ├── memory.py             # Mémoire court/long terme
│   ├── decision.py           # Prise de décision
│   ├── learning.py           # Apprentissage
│   └── language.py           # Traitement du langage
├── utils/                     # Utilitaires
│   ├── __init__.py
│   ├── logging.py            # Système de logging
│   └── exceptions.py         # Exceptions personnalisées
├── plugins/                   # Système de plugins
│   ├── __init__.py
│   └── plugin_interface.py   # Interface des plugins
├── tests/                     # Tests unitaires
│   ├── __init__.py
│   ├── test_brain.py
│   ├── test_neuron.py
│   ├── test_synapse.py
│   ├── test_network.py
│   ├── test_memory.py
│   ├── test_learning.py
│   └── test_language.py
├── requirements.txt           # Dépendances production
├── requirements-dev.txt       # Dépendances développement
├── setup.py                   # Configuration du package
├── Dockerfile                 # Configuration Docker
├── docker-compose.yml         # Orchestration Docker
├── pytest.ini                 # Configuration pytest
├── Makefile                   # Commandes make
├── .gitignore                # Fichiers à ignorer
├── .env.example              # Variables d'environnement
└── README.md                 # Ce fichier
```

## 🧪 Tests

### Exécuter tous les tests

```bash
make test
# ou
pytest
```

### Avec couverture de code

```bash
make test-coverage
# ou
pytest --cov=. --cov-report=html
```

### Tests spécifiques

```bash
pytest tests/test_neuron.py
pytest tests/test_memory.py -v
```

## 🔧 Développement

### Formatage du code

```bash
make format
# ou
black .
isort .
```

### Vérification de la qualité

```bash
make lint
# ou
flake8 .
pylint core modules utils
```

### Nettoyage

```bash
make clean
```

## 🐳 Docker

### Construction

```bash
make docker-build
# ou
docker build -t brain-neural-network .
```

### Lancement

```bash
make docker-run
# ou
docker-compose up -d
```

### Logs

```bash
make docker-logs
# ou
docker-compose logs -f brain
```

### Arrêt

```bash
make docker-stop
# ou
docker-compose down
```

## 🧠 Mécanismes Biologiques Implémentés

### Plasticité Synaptique
Le système implémente la plasticité synaptique qui permet au cerveau de renforcer ou affaiblir les connexions neuronales en fonction de l'activité.

### Potentialisation à Long Terme (LTP)
Mécanisme cellulaire primaire pour l'apprentissage et la mémoire basé sur le renforcement persistant des synapses.

### Rôle de l'Hippocampe
L'hippocampe joue un rôle crucial dans la consolidation des informations de la mémoire à court terme vers la mémoire à long terme.

### Consolidation de la Mémoire
Processus de stabilisation des traces mnésiques au fil du temps, transformant les souvenirs à court terme en souvenirs à long terme plus stables.

### Stockage Distribué
Contrairement à un "centre de mémoire" unique, les souvenirs sont stockés dans des réseaux distribués à travers le cerveau.

### Neurogenèse
Formation de nouveaux neurones dans l'hippocampe contribuant à la formation de la mémoire et à la flexibilité cognitive.

### Synthèse Protéique
La formation de souvenirs à long terme nécessite la synthèse de nouvelles protéines pour renforcer les connexions synaptiques.

### Reconsolidation
Lorsque les souvenirs sont rappelés, ils deviennent temporairement malléables et peuvent être modifiés avant d'être re-stockés.

### Étiquetage Émotionnel
L'amygdale joue un rôle dans l'attachement de signification émotionnelle aux souvenirs, influençant leur force et leur rappel.

## 📊 Performance

Le système est conçu pour être performant avec:
- Support multi-threading pour les simulations
- Optimisations NumPy pour les calculs vectoriels
- Gestion efficace de la mémoire
- Cache des modèles Transformer

## 🤝 Contribution

Les contributions sont les bienvenues! Pour contribuer:

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Auteurs

- **Brain Team** - *Travail initial*

## 🙏 Remerciements

- Inspiré par les mécanismes biologiques du cerveau
- Utilise Hugging Face Transformers pour le traitement du langage
- Basé sur des modèles neuronaux bio-réalistes

## 📚 Références

- Leaky Integrate-and-Fire neuron model
- Spike-Timing-Dependent Plasticity (STDP)
- Synaptic plasticity and memory formation
- Emotional processing in neural networks

## 🔮 Roadmap

- [ ] Support GPU pour accélération
- [ ] Interface web interactive
- [ ] API REST
- [ ] Plus de types de neurones (Izhikevich, Hodgkin-Huxley)
- [ ] Visualisation en temps réel
- [ ] Support de réseaux de neurones convolutifs
- [ ] Intégration avec frameworks de deep learning

## 📞 Contact

Pour questions ou suggestions:
- GitHub Issues: [https://github.com/JacquesGariepy/Brain/issues](https://github.com/JacquesGariepy/Brain/issues)

---

Fait avec ❤️ par l'équipe Brain
