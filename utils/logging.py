import logging
import sys
from datetime import datetime
from pathlib import Path

class BrainLogger:
    """
    Système de logging centralisé pour le projet Brain.
    
    Attributes:
        logger (logging.Logger): Instance du logger.
        log_file (str): Chemin vers le fichier de log.
    """
    
    def __init__(self, name='Brain', log_level=logging.INFO, log_dir='logs'):
        """
        Initialise le système de logging.
        
        Args:
            name (str): Nom du logger.
            log_level (int): Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_dir (str): Répertoire pour stocker les fichiers de log.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Créer le répertoire de logs s'il n'existe pas
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Nom du fichier de log avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = log_path / f'brain_{timestamp}.log'
        
        # Formatter pour les logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler pour fichier
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Handler pour console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Ajouter les handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def debug(self, message):
        """Log un message de niveau DEBUG."""
        self.logger.debug(message)
    
    def info(self, message):
        """Log un message de niveau INFO."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log un message de niveau WARNING."""
        self.logger.warning(message)
    
    def error(self, message, exc_info=False):
        """Log un message de niveau ERROR."""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message, exc_info=False):
        """Log un message de niveau CRITICAL."""
        self.logger.critical(message, exc_info=exc_info)
    
    def exception(self, message):
        """Log une exception avec traceback complet."""
        self.logger.exception(message)


# Instance globale du logger
brain_logger = BrainLogger()
