# Brain

brain_model/
├── main.py                    # Point d'entrée principal du programme
├── modules/
│   ├── neuron.py              # Classe Neuron
│   ├── synapse.py             # Classe Synapse
│   ├── network.py             # Gestion du réseau neuronal
│   ├── attention.py           # Gestion de l'attention
│   ├── emotion.py             # Gestion des émotions
│   ├── memory.py              # Mémoire à court terme et long terme
│   ├── perception.py          # Perception sensorielle
│   ├── decision.py            # Module de prise de décision
│   ├── learning.py            # Module d'apprentissage supervisé, non supervisé et par renforcement
│   ├── language.py            # Intégration des modèles Hugging Face pour le langage
├── utils/
│   ├── logging.py             # Gestion des logs
│   ├── exceptions.py          # Gestion des exceptions
└── tests/
    ├── test_neuron.py         # Tests unitaires pour la classe Neuron
    ├── test_synapse.py        # Tests unitaires pour la classe Synapse
    ├── test_network.py        # Tests unitaires pour le réseau neuronal
    ├── ...

## Example Agent

An example agent has been created to demonstrate the functionality of the `Brain` class. The agent initializes a `Brain` instance, processes sensory input, makes decisions, learns new knowledge, and communicates.

### Running the Example Agent

To run the example agent, execute the following command:

```bash
python examples/agent_example.py
```

### Example Agent Functionality

The example agent performs the following tasks:

1. Initializes a `Brain` instance.
2. Processes sensory input.
3. Makes decisions based on the processed data.
4. Learns new knowledge by injecting text.
5. Communicates by generating a response to a prompt.

The example agent demonstrates how to use the `Brain` class and its methods to create an intelligent agent capable of perceiving, learning, and making decisions.
