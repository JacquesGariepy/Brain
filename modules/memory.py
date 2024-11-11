import json
from collections import deque

class MemoryModule:
    """
    Module de mémoire gérant la mémoire à court et long terme, avec persistance des informations.
    
    Attributes:
        short_term_memory (deque): Mémoire à court terme (MCT) limitée en capacité.
        long_term_memory (dict): Mémoire à long terme (MLT) persistante.
        filename (str): Nom du fichier où les données de la mémoire à long terme sont sauvegardées.
    """
    
    def __init__(self, filename="long_term_memory.json"):
        self.short_term_memory = deque(maxlen=5)  # MCT avec une capacité limitée
        self.long_term_memory = {}
        self.filename = filename
        self.load_long_term_memory()

    def store_short_term(self, data):
        """
        Stocke des données dans la mémoire à court terme.
        
        Args:
            data (any): Données à stocker.
        """
        self.short_term_memory.append(data)

    def retrieve_short_term(self):
        """
        Récupère les données de la mémoire à court terme.
        
        Returns:
            list: Liste des éléments actuellement en MCT.
        """
        return list(self.short_term_memory)

    def store_long_term(self, key, data):
        """
        Stocke des données dans la mémoire à long terme.
        
        Args:
            key (str): Clé pour identifier les données.
            data (any): Données à stocker.
        """
        self.long_term_memory[key] = data
        self.save_long_term_memory()

    def retrieve_long_term(self, key):
        """
        Récupère les données de la mémoire à long terme.
        
        Args:
            key (str): Clé pour identifier les données.
            
        Returns:
            any: Données récupérées ou None si la clé n'existe pas.
        """
        return self.long_term_memory.get(key, None)

    def save_long_term_memory(self):
        """Sauvegarde la mémoire à long terme sur disque."""
        with open(self.filename, "w") as f:
            json.dump(self.long_term_memory, f)

    def load_long_term_memory(self):
        """Charge la mémoire à long terme depuis un fichier."""
        try:
            with open(self.filename, "r") as f:
                self.long_term_memory = json.load(f)
        except FileNotFoundError:
            self.long_term_memory = {}

    def synaptic_plasticity(self, synapse):
        """
        Implémente la plasticité synaptique pour renforcer les connexions entre les neurones.
        
        Args:
            synapse (Synapse): La synapse à renforcer.
        """
        synapse.weight += 0.05  # Exemple de renforcement
        synapse.weight = min(synapse.weight, 1.0)  # Limite supérieure du poids

    def long_term_potentiation(self, synapse):
        """
        Implémente la potentialisation à long terme (LTP) pour renforcer les connexions synaptiques.
        
        Args:
            synapse (Synapse): La synapse à renforcer.
        """
        synapse.weight += 0.1  # Exemple de renforcement
        synapse.weight = min(synapse.weight, 1.0)  # Limite supérieure du poids

    def hippocampal_involvement(self, data):
        """
        Simule l'implication de l'hippocampe dans la consolidation de la mémoire.
        
        Args:
            data (any): Données à consolider.
        """
        self.store_short_term(data)
        if len(self.short_term_memory) == self.short_term_memory.maxlen:
            consolidated_data = " ".join(self.retrieve_short_term())
            self.store_long_term("consolidated_memory", consolidated_data)
            self.short_term_memory.clear()

    def memory_consolidation(self):
        """
        Consolide les souvenirs de la mémoire à court terme à la mémoire à long terme.
        """
        for item in self.short_term_memory:
            self.store_long_term(f"consolidated_{item}", item)
        self.short_term_memory.clear()

    def distributed_storage(self, data):
        """
        Stocke les souvenirs de manière distribuée dans différentes régions de la mémoire.
        
        Args:
            data (any): Données à stocker.
        """
        regions = ["region_1", "region_2", "region_3"]
        for region in regions:
            self.store_long_term(f"{region}_{data}", data)

    def neurogenesis(self, new_neurons):
        """
        Simule la neurogenèse en ajoutant de nouveaux neurones à la mémoire.
        
        Args:
            new_neurons (list): Liste des nouveaux neurones à ajouter.
        """
        self.long_term_memory["new_neurons"] = new_neurons
        self.save_long_term_memory()

    def protein_synthesis(self, synapse):
        """
        Simule la synthèse protéique pour renforcer les connexions synaptiques.
        
        Args:
            synapse (Synapse): La synapse à renforcer.
        """
        synapse.weight += 0.2  # Exemple de renforcement
        synapse.weight = min(synapse.weight, 1.0)  # Limite supérieure du poids

    def reconsolidation(self, data):
        """
        Simule la reconsolidation des souvenirs rappelés.
        
        Args:
            data (any): Données à reconsolider.
        """
        modified_data = f"modified_{data}"
        self.store_long_term("reconsolidated_memory", modified_data)

    def emotional_labeling(self, data, emotion):
        """
        Simule l'étiquetage émotionnel des souvenirs.
        
        Args:
            data (any): Données à étiqueter.
            emotion (str): Émotion associée.
        """
        labeled_data = f"{data}_{emotion}"
        self.store_long_term("emotional_memory", labeled_data)
