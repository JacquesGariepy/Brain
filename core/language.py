from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils.exceptions import LanguageException
from utils.logging import brain_logger
from .interfaces import BrainModule


class LanguageModule(BrainModule):
    """
    Module de gestion du langage utilisant des modèles Hugging Face pour l'apprentissage et la génération de texte.
    
    Attributes:
        tokenizer (AutoTokenizer): Tokenizer pour convertir le texte en tokens utilisables.
        model (AutoModelForCausalLM): Modèle pré-entraîné pour générer du langage.
        vocabulary (set): Ensemble de mots appris par le cerveau.
        grammar_rules (dict): Règles de grammaire pour la génération de phrases.
    """
    
    def __init__(self, model_name="gpt2"):
        """
        Initialise le module de langage.
        
        Args:
            model_name (str): Nom du modèle Hugging Face à utiliser.
        """
        try:
            brain_logger.info(f"Initialisation du module de langage avec le modèle {model_name}")
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.vocabulary = set()
            self.grammar_rules = {}
            brain_logger.info("Module de langage initialisé avec succès")
        except Exception as e:
            brain_logger.error(f"Erreur lors de l'initialisation du module de langage: {str(e)}")
            raise LanguageException(f"Impossible d'initialiser le module de langage: {str(e)}")

    def process(self, data):
        """
        Traite les données linguistiques.
        
        Args:
            data: Données à traiter.
            
        Returns:
            Données traitées.
        """
        if isinstance(data, str):
            return self.understand_sentence(data)
        return data

    def learn_text(self, text):
        """
        Apprend du texte en utilisant le modèle Hugging Face et met à jour le vocabulaire.
        
        Args:
            text (str): Texte à apprendre.
        """
        try:
            inputs = self.tokenizer.encode(text, return_tensors='pt')
            outputs = self.model(inputs, labels=inputs)
            loss = outputs.loss
            
            # Mise à jour du vocabulaire
            tokens = self.tokenizer.tokenize(text)
            self.vocabulary.update(tokens)
            
            brain_logger.info(f"Texte appris avec succès. Vocabulaire: {len(self.vocabulary)} tokens")
        except Exception as e:
            brain_logger.error(f"Erreur lors de l'apprentissage du texte: {str(e)}")
            raise LanguageException(f"Impossible d'apprendre le texte: {str(e)}")

    def generate_sentence(self, prompt="", max_length=50):
        """
        Génère une phrase en utilisant le modèle Hugging Face.
        
        Args:
            prompt (str): Prompt initial pour générer du texte.
            max_length (int): Longueur maximale de la génération.
            
        Returns:
            str: Phrase générée.
        """
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.model.generate(inputs, max_length=max_length, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
            sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            brain_logger.debug(f"Phrase générée: {sentence}")
            return sentence
        except Exception as e:
            brain_logger.error(f"Erreur lors de la génération de phrase: {str(e)}")
            raise LanguageException(f"Impossible de générer la phrase: {str(e)}")

    def understand_sentence(self, sentence):
        """
        Évalue la compréhension d'une phrase en analysant sa probabilité sous le modèle GPT-2.
        
        Args:
            sentence (str): Phrase à analyser.
            
        Returns:
            str: Indication de la compréhension.
        """
        try:
            inputs = self.tokenizer.encode(sentence, return_tensors='pt')
            outputs = self.model(inputs, labels=inputs)
            loss = outputs.loss.item()
            
            if loss < 1.0:
                result = "Phrase comprise."
            else:
                result = "Phrase partiellement comprise."
            
            brain_logger.debug(f"Compréhension de la phrase (loss={loss:.2f}): {result}")
            return result
        except Exception as e:
            brain_logger.error(f"Erreur lors de la compréhension de la phrase: {str(e)}")
            raise LanguageException(f"Impossible de comprendre la phrase: {str(e)}")
