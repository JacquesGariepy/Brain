class Brain:
    """
    Classe principale représentant un cerveau artificiel intégrant toutes les fonctionnalités cérébrales.
    
    Attributes:
        network (Network): Le réseau neuronal de neurones et synapses.
        memory_module (MemoryModule): Module de gestion de la mémoire.
        perception_module (PerceptionModule): Module de perception sensorielle.
        attention_module (AttentionModule): Module d'attention.
        emotion_module (EmotionModule): Module d'émotions.
        decision_module (DecisionModule): Module de prise de décision.
        learning_module (LearningModule): Module d'apprentissage.
        language_module (LanguageModule): Module de gestion du langage.
    """
    
    def __init__(self):
        self.network = Network()
        self.create_neurons_and_synapses()
        
        # Modules
        self.memory_module = MemoryModule()
        self.perception_module = PerceptionModule(self.network)
        self.attention_module = AttentionModule(self.network.neurons)
        self.emotion_module = EmotionModule()
        self.decision_module = DecisionModule()
        self.learning_module = LearningModule(self.network, self.memory_module)
        self.language_module = LanguageModule(self.memory_module)

    def create_neurons_and_synapses(self):
        """
        Crée les neurones et les synapses nécessaires pour le réseau.
        Les connexions entre les neurones sont établies de manière aléatoire pour simplification.
        """
        num_neurons = 100  # Par exemple, 100 neurones dans le réseau
        for i in range(num_neurons):
            neuron = Neuron(neuron_id=i)
            self.network.add_neuron(neuron)
        
        # Connexions synaptiques aléatoires entre les neurones
        for pre_neuron in self.network.neurons:
            for post_neuron in random.sample(self.network.neurons, k=10):  # Connecte chaque neurone à 10 autres
                if pre_neuron != post_neuron:
                    self.network.connect_neurons(pre_neuron, post_neuron, weight=random.uniform(0.1, 0.9), delay=random.uniform(1.0, 5.0))

        # Les 10 premiers neurones seront considérés comme des neurones sensoriels
        self.perception_module.add_sensory_neurons(self.network.neurons[:10])

    def perceive_and_process(self, sensory_input, dt):
        """
        Gère le flux de perception des stimuli sensoriels et le traitement des informations par le cerveau.
        
        Args:
            sensory_input (list): Liste des stimuli sensoriels reçus.
            dt (float): Pas de temps pour la mise à jour du réseau.
        """
        # Perception sensorielle
        self.perception_module.encode_sensory_input(sensory_input)
        
        # Mise à jour de l'attention
        relevance_signal = {neuron.neuron_id: random.uniform(0, 1) for neuron in self.network.neurons}
        self.attention_module.update_attention(relevance_signal)
        
        # Mise à jour des émotions
        self.emotion_module.update_emotions(sensory_inputs=sensory_input, memories=self.memory_module.retrieve_short_term(), reward=0.0, dt=dt)
        self.emotion_module.influence_on_neurons(self.network.neurons)
        
        # Mise à jour du réseau neuronal
        self.network.update(dt)
        
        # Processus de prise de décision
        evidence = sum(neuron.spike for neuron in self.network.neurons) / len(self.network.neurons)
        emotion_influence = self.emotion_module.emotional_states["joy"] - self.emotion_module.emotional_states["fear"]
        self.decision_module.update_decision(evidence, emotion_influence, dt)
        
        if self.decision_module.choice_made:
            self.execute_decision()
            self.decision_module.reset()

    def execute_decision(self):
        """
        Exécute la décision prise par le cerveau et l'enregistre dans la mémoire.
        """
        action = self.decision_module.decision
        print(f"Action exécutée : {action}")
        self.memory_module.store_short_term(action)
        self.memory_module.store_long_term("last_action", action)

    def learn(self):
        """
        Active les différents types d'apprentissage : supervisé, non supervisé, et par renforcement.
        """
        inputs = np.random.rand(len(self.network.neurons))
        targets = np.random.randint(0, 2, len(self.network.neurons))
        
        # Apprentissage supervisé
        self.learning_module.supervised_learning(inputs, targets)
        
        # Apprentissage non supervisé (k-means clustering)
        self.learning_module.unsupervised_learning(inputs)
        
        # Apprentissage par renforcement (récompense aléatoire)
        reward = random.choice([-1, 0, 1])
        self.learning_module.reinforcement_learning(reward)

    def communicate(self):
        """
        Utilise le module de langage pour générer une phrase ou comprendre une phrase fournie.
        """
        sentence = self.language_module.generate_sentence()
        print(f"Phrase générée : {sentence}")
        
        understanding = self.language_module.understand_sentence(sentence)
        print(f"Compréhension de la phrase : {understanding}")

    def inject_knowledge(self, text):
        """
        Injecte des compétences ou des connaissances dans le cerveau via le module de langage.
        
        Args:
            text (str): Texte à apprendre (par exemple, un texte sur une nouvelle compétence).
        """
        self.language_module.learn_text(text)
        print("Nouvelle compétence injectée dans le cerveau.")

    def save_state(self):
        """
        Sauvegarde l'état actuel du cerveau, y compris la mémoire à long terme.
        """
        self.memory_module.save_long_term_memory()

    def load_state(self):
        """
        Charge l'état précédent du cerveau, y compris la mémoire à long terme.
        """
        self.memory_module.load_long_term_memory()
