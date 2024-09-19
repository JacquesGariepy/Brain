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
