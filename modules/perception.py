"""
Module de perception avec encodage sensoriel réel.

Implémente:
- Rate coding (codage par taux de décharge)
- Temporal coding (codage temporel)
- Feature extraction (extraction de caractéristiques)
- Normalisation adaptative
- Support multi-modal (visuel, auditif, tactile)

Conforme aux modèles de codage neuronal (Rieke et al., 1999)
et de traitement sensoriel (Hubel & Wiesel, 1962).
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PerceptionModule:
    """
    Module de perception qui encode les entrées sensorielles pour le réseau neuronal.

    Attributes:
        network (Network): Réseau neuronal auquel les entrées sont transmises.
        sensory_neurons (list): Liste des neurones sensoriels.
        encoding_type (str): Type d'encodage ('rate', 'temporal', 'hybrid').
        normalization_window (list): Fenêtre glissante pour normalisation adaptative.
        modality_ranges (dict): Plages de neurones pour chaque modalité sensorielle.
    """

    def __init__(self, network, encoding_type: str = 'rate', normalize: bool = True):
        """
        Initialise le module de perception.

        Args:
            network: Réseau neuronal
            encoding_type: Type d'encodage ('rate', 'temporal', 'hybrid')
            normalize: Activer la normalisation adaptative
        """
        self.network = network
        self.sensory_neurons = []
        self.encoding_type = encoding_type
        self.normalize = normalize

        # Normalisation adaptative
        self.normalization_window = []
        self.window_size = 100  # Taille de la fenêtre glissante

        # Modalités sensorielles (assignées dynamiquement)
        self.modality_ranges = {}

        logger.info(f"PerceptionModule initialisé: encoding={encoding_type}, normalize={normalize}")

    def add_sensory_neurons(self, neurons, modality: str = 'visual'):
        """
        Ajoute des neurones sensoriels au module de perception.

        Args:
            neurons (list): Liste des neurones sensoriels à ajouter.
            modality (str): Modalité sensorielle ('visual', 'auditory', 'tactile', 'proprioceptive')
        """
        start_idx = len(self.sensory_neurons)
        self.sensory_neurons.extend(neurons)
        end_idx = len(self.sensory_neurons)

        # Enregistrer la plage pour cette modalité
        self.modality_ranges[modality] = (start_idx, end_idx)

        logger.info(f"Modalité '{modality}': {len(neurons)} neurones ajoutés (indices {start_idx}-{end_idx})")

    def normalize_input(self, sensory_input: np.ndarray) -> np.ndarray:
        """
        Normalise les entrées sensorielles de manière adaptative.

        Utilise une fenêtre glissante pour calculer la moyenne et l'écart-type,
        permettant une adaptation aux changements de distribution.

        Args:
            sensory_input: Entrées sensorielles brutes

        Returns:
            Entrées normalisées (z-score)
        """
        if not self.normalize:
            return sensory_input

        # Ajouter à la fenêtre glissante
        self.normalization_window.append(sensory_input)
        if len(self.normalization_window) > self.window_size:
            self.normalization_window.pop(0)

        # Calculer statistiques sur la fenêtre
        if len(self.normalization_window) > 10:  # Au moins 10 échantillons
            window_data = np.array(self.normalization_window)
            mean = np.mean(window_data, axis=0)
            std = np.std(window_data, axis=0) + 1e-9

            # Z-score normalization
            normalized = (sensory_input - mean) / std
        else:
            # Pas assez de données, normalisation simple
            mean = np.mean(sensory_input)
            std = np.std(sensory_input) + 1e-9
            normalized = (sensory_input - mean) / std

        return normalized

    def extract_features(self, sensory_input: np.ndarray) -> np.ndarray:
        """
        Extrait des caractéristiques basiques des entrées sensorielles.

        Calcule:
        - Gradients (changements locaux)
        - Moyennes locales
        - Variances locales

        Args:
            sensory_input: Entrées sensorielles

        Returns:
            Vecteur de caractéristiques augmenté
        """
        sensory_input = np.asarray(sensory_input)

        # Si l'entrée est trop petite, pas d'extraction
        if len(sensory_input) < 3:
            return sensory_input

        features = []

        # 1. Valeurs brutes
        features.extend(sensory_input)

        # 2. Gradients (différences entre éléments adjacents)
        gradients = np.diff(sensory_input)
        # Ajouter un 0 au début pour garder la même longueur
        gradients = np.concatenate([[0], gradients])
        features.extend(gradients * 0.5)  # Pondérer moins que les valeurs brutes

        # 3. Moyennes locales (fenêtre de 3)
        local_means = []
        for i in range(len(sensory_input)):
            start = max(0, i - 1)
            end = min(len(sensory_input), i + 2)
            local_mean = np.mean(sensory_input[start:end])
            local_means.append(local_mean)
        features.extend(np.array(local_means) * 0.3)

        return np.array(features[:len(sensory_input)])  # Tronquer à la longueur originale

    def rate_encoding(self, value: float, baseline: float = 0.0, max_rate: float = 100.0) -> float:
        """
        Encode une valeur en taux de décharge (rate coding).

        Plus la valeur est élevée, plus le taux de décharge est élevé.

        Args:
            value: Valeur à encoder
            baseline: Taux de décharge au repos (Hz)
            max_rate: Taux de décharge maximal (Hz)

        Returns:
            Taux de décharge (Hz)
        """
        # Normaliser value entre 0 et 1
        normalized_value = np.clip(value, 0.0, 1.0)

        # Transformer en taux de décharge
        rate = baseline + (max_rate - baseline) * normalized_value

        return rate

    def temporal_encoding(self, value: float, time_window: float = 10.0) -> float:
        """
        Encode une valeur en latence de spike (temporal coding).

        Plus la valeur est élevée, plus le spike arrive tôt.

        Args:
            value: Valeur à encoder (0-1)
            time_window: Fenêtre temporelle maximale (ms)

        Returns:
            Latence du spike (ms)
        """
        # Normaliser value entre 0 et 1
        normalized_value = np.clip(value, 0.0, 1.0)

        # Transformer en latence (valeurs élevées -> latence faible)
        latency = time_window * (1.0 - normalized_value)

        return latency

    def encode_as_current(self, rate: float, gain: float = 1.0) -> float:
        """
        Convertit un taux de décharge en courant d'entrée pour le neurone.

        Args:
            rate: Taux de décharge (Hz)
            gain: Gain de conversion (courant par Hz)

        Returns:
            Courant d'entrée (mA)
        """
        # Conversion simple: rate * gain
        current = rate * gain

        return current

    def encode_sensory_input(self, sensory_input, dt: float = 1.0):
        """
        Encode les entrées sensorielles en courants neuronaux.

        Applique le pipeline complet:
        1. Normalisation adaptative
        2. Extraction de caractéristiques
        3. Encodage (rate ou temporal)
        4. Conversion en courants
        5. Application aux neurones sensoriels

        Args:
            sensory_input (list or np.ndarray): Entrées sensorielles à encoder.
            dt (float): Pas de temps de simulation (ms).
        """
        sensory_input = np.asarray(sensory_input)

        # 1. Normalisation adaptative
        if self.normalize:
            normalized_input = self.normalize_input(sensory_input)
        else:
            normalized_input = sensory_input

        # 2. Extraction de caractéristiques (optionnel)
        # Pour l'instant, on utilise directement les valeurs normalisées
        processed_input = normalized_input

        # 3. Encoder et appliquer aux neurones
        num_inputs = min(len(processed_input), len(self.sensory_neurons))

        for i in range(num_inputs):
            neuron = self.sensory_neurons[i]
            value = processed_input[i]

            if self.encoding_type == 'rate':
                # Rate coding: convertir valeur en taux puis en courant
                # Normaliser value à [0, 1]
                normalized_value = (value + 3.0) / 6.0  # Assume z-score entre -3 et +3
                normalized_value = np.clip(normalized_value, 0.0, 1.0)

                rate = self.rate_encoding(normalized_value, baseline=10.0, max_rate=100.0)
                current = self.encode_as_current(rate, gain=0.5)

                # Appliquer le courant au neurone
                neuron.receive_current(current)

            elif self.encoding_type == 'temporal':
                # Temporal coding: latence de spike
                normalized_value = (value + 3.0) / 6.0
                normalized_value = np.clip(normalized_value, 0.0, 1.0)

                latency = self.temporal_encoding(normalized_value, time_window=dt)

                # Si latency < dt, le neurone spike maintenant
                if latency < dt:
                    # Appliquer un courant très fort pour forcer un spike
                    neuron.receive_current(100.0)
                else:
                    # Pas de spike ce pas de temps
                    neuron.receive_current(0.0)

            elif self.encoding_type == 'hybrid':
                # Hybrid: combinaison de rate et temporal
                normalized_value = (value + 3.0) / 6.0
                normalized_value = np.clip(normalized_value, 0.0, 1.0)

                # 50% rate, 50% temporal
                rate = self.rate_encoding(normalized_value, baseline=10.0, max_rate=80.0)
                current_rate = self.encode_as_current(rate, gain=0.3)

                latency = self.temporal_encoding(normalized_value, time_window=dt)
                current_temporal = 50.0 if latency < dt else 0.0

                # Combinaison
                total_current = current_rate + current_temporal * 0.5
                neuron.receive_current(total_current)

        logger.debug(f"Encodage sensoriel: {num_inputs} neurones stimulés, type={self.encoding_type}")

    def set_encoding_type(self, encoding_type: str):
        """
        Modifie le type d'encodage sensoriel.

        Args:
            encoding_type: Type d'encodage ('rate', 'temporal', 'hybrid')
        """
        if encoding_type in ['rate', 'temporal', 'hybrid']:
            self.encoding_type = encoding_type
            logger.info(f"Type d'encodage modifié: {encoding_type}")
        else:
            logger.warning(f"Type d'encodage invalide: {encoding_type}")

    def get_modality_neurons(self, modality: str):
        """
        Retourne les neurones associés à une modalité spécifique.

        Args:
            modality: Nom de la modalité ('visual', 'auditory', etc.)

        Returns:
            Liste des neurones de cette modalité
        """
        if modality in self.modality_ranges:
            start, end = self.modality_ranges[modality]
            return self.sensory_neurons[start:end]
        else:
            logger.warning(f"Modalité inconnue: {modality}")
            return []

    def reset_normalization(self):
        """Réinitialise la fenêtre de normalisation adaptative."""
        self.normalization_window.clear()
        logger.debug("Fenêtre de normalisation réinitialisée")

    def get_perception_state(self) -> dict:
        """
        Retourne l'état actuel du module de perception.

        Returns:
            Dictionnaire avec les métriques de perception
        """
        return {
            'num_sensory_neurons': len(self.sensory_neurons),
            'encoding_type': self.encoding_type,
            'normalize': self.normalize,
            'normalization_samples': len(self.normalization_window),
            'modalities': list(self.modality_ranges.keys())
        }
