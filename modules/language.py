from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as transformers_logging
import torch
import warnings

class LanguageModule:
    """
    Module de gestion du langage utilisant des modèles Hugging Face pour l'apprentissage et la génération de texte.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer pour convertir le texte en tokens utilisables.
        model (AutoModelForCausalLM): Modèle pré-entraîné pour générer du langage.
        vocabulary (set): Ensemble de mots appris par le cerveau.
        grammar_rules (dict): Règles de grammaire pour la génération de phrases.
    """

    def __init__(self, memory_module):
        self.memory = memory_module

        # Suppress transformers warnings during model loading
        previous_level = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        finally:
            # Restore previous logging level
            transformers_logging.set_verbosity(previous_level)

        self.vocabulary = set(self.memory.retrieve_long_term("vocabulary") or [])
        self.grammar_rules = self.memory.retrieve_long_term("grammar_rules") or {}

    def learn_text(self, text):
        """
        Apprend du texte en utilisant le modèle Hugging Face et met à jour le vocabulaire.
        
        Args:
            text (str): Texte à apprendre.
        """
        inputs = self.tokenizer.encode(text, return_tensors='pt')
        outputs = self.model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        # Mise à jour du vocabulaire
        tokens = self.tokenizer.tokenize(text)
        self.vocabulary.update(tokens)
        self.memory.store_long_term("vocabulary", list(self.vocabulary))

    def generate_sentence(self, prompt=""):
        """
        Génère une phrase en utilisant le modèle Hugging Face.

        Args:
            prompt (str): Prompt initial pour générer du texte.

        Returns:
            str: Phrase générée.
        """
        # Set pad_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize with attention mask
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Generate with proper parameters
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=50,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return sentence

    def understand_sentence(self, sentence):
        """
        Évalue la compréhension d'une phrase en analysant sa probabilité sous le modèle GPT-2.
        
        Args:
            sentence (str): Phrase à analyser.
            
        Returns:
            str: Indication de la compréhension.
        """
        inputs = self.tokenizer.encode(sentence, return_tensors='pt')
        outputs = self.model(inputs, labels=inputs)
        loss = outputs.loss.item()
        if loss < 1.0:
            return "Phrase comprise."
        else:
            return "Phrase partiellement comprise."
