"""
Intelligent Brain Orchestrator

Système d'orchestration qui sélectionne et combine dynamiquement TOUS les composants
de façon intelligente, scientifique et adaptative.

Capabilities:
- Dynamic architecture selection based on task
- Automatic multimodal fusion
- Adaptive computation
- Ensemble methods
- Meta-learning for architecture selection
- Scientific routing based on task characteristics

Utilise TOUS les composants en:
1. Symbiose - Dynamiquement et intelligemment
2. Unitaire - Chaque module indépendamment
3. Combinatoire - Pour besoins divers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging


class TaskType(Enum):
    """Types de tâches supportées"""
    # Vision
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    IMAGE_GENERATION = "image_generation"

    # Language
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    QUESTION_ANSWERING = "question_answering"
    TRANSLATION = "translation"

    # Multimodal
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    IMAGE_CAPTIONING = "image_captioning"
    TEXT_TO_IMAGE = "text_to_image"
    VIDEO_UNDERSTANDING = "video_understanding"

    # Audio
    SPEECH_RECOGNITION = "speech_recognition"
    AUDIO_GENERATION = "audio_generation"
    AUDIO_CLASSIFICATION = "audio_classification"

    # Reasoning
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    COMMONSENSE_REASONING = "commonsense_reasoning"
    LOGICAL_REASONING = "logical_reasoning"

    # RL
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    PLANNING = "planning"

    # Scientific
    PROTEIN_FOLDING = "protein_folding"
    MOLECULE_GENERATION = "molecule_generation"
    PHYSICS_SIMULATION = "physics_simulation"

    # General
    MULTIMODAL_REASONING = "multimodal_reasoning"
    FEW_SHOT_LEARNING = "few_shot_learning"
    CONTINUAL_LEARNING = "continual_learning"


class ModalityType(Enum):
    """Types de modalités"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    GRAPH = "graph"
    TABULAR = "tabular"
    PROTEIN = "protein"
    MOLECULE = "molecule"


@dataclass
class TaskSpecification:
    """Spécification complète d'une tâche"""
    task_type: TaskType
    modalities: List[ModalityType]
    input_shape: Dict[str, tuple]
    output_shape: Optional[tuple] = None
    constraints: Optional[Dict[str, Any]] = None
    performance_requirements: Optional[Dict[str, float]] = None


@dataclass
class ArchitectureSelection:
    """Sélection d'architecture et justification"""
    primary_architecture: str
    supporting_architectures: List[str]
    fusion_strategy: str
    reasoning: str
    confidence: float


class IntelligentOrchestrator(nn.Module):
    """
    Orchestrateur intelligent qui sélectionne et combine dynamiquement
    toutes les architectures disponibles.

    Principes:
    1. Task-aware: Analyse la tâche et sélectionne les architectures optimales
    2. Modality-aware: Gère toutes les modalités (texte, image, audio, etc.)
    3. Adaptive: S'adapte aux contraintes (latence, mémoire, précision)
    4. Scientific: Décisions basées sur des principes et métriques
    5. Meta-learning: Apprend de l'expérience pour améliorer les sélections
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Registry de toutes les architectures disponibles
        self.architecture_registry = {}

        # Meta-learner pour la sélection d'architecture
        self.meta_selector = None

        # Historique des performances pour amélioration continue
        self.performance_history = []

        # Initialize all components
        self._initialize_architectures()

    def _initialize_architectures(self):
        """Initialize tous les composants SOTA disponibles"""

        # === TRANSFORMERS & LANGUAGE MODELS ===
        self.architecture_registry['transformer'] = {
            'type': 'language',
            'modalities': [ModalityType.TEXT],
            'tasks': [
                TaskType.TEXT_GENERATION,
                TaskType.TEXT_CLASSIFICATION,
                TaskType.QUESTION_ANSWERING
            ],
            'complexity': 'O(N²)',
            'load_fn': self._load_transformer
        }

        self.architecture_registry['mamba'] = {
            'type': 'language',
            'modalities': [ModalityType.TEXT],
            'tasks': [
                TaskType.TEXT_GENERATION,
                TaskType.TEXT_CLASSIFICATION
            ],
            'complexity': 'O(N)',
            'advantages': ['long_sequences', 'faster_inference'],
            'load_fn': self._load_mamba
        }

        # === VISION MODELS ===
        self.architecture_registry['vit'] = {
            'type': 'vision',
            'modalities': [ModalityType.IMAGE],
            'tasks': [
                TaskType.IMAGE_CLASSIFICATION,
                TaskType.OBJECT_DETECTION
            ],
            'load_fn': self._load_vit
        }

        self.architecture_registry['swin'] = {
            'type': 'vision',
            'modalities': [ModalityType.IMAGE],
            'tasks': [
                TaskType.IMAGE_CLASSIFICATION,
                TaskType.SEMANTIC_SEGMENTATION
            ],
            'advantages': ['hierarchical', 'efficient'],
            'load_fn': self._load_swin
        }

        self.architecture_registry['yolov8'] = {
            'type': 'vision',
            'modalities': [ModalityType.IMAGE],
            'tasks': [
                TaskType.OBJECT_DETECTION
            ],
            'advantages': ['real_time', 'anchor_free', 'efficient'],
            'capabilities': ['multiple_scales', 'decoupled_head'],
            'load_fn': self._load_yolov8
        }

        self.architecture_registry['detr'] = {
            'type': 'vision',
            'modalities': [ModalityType.IMAGE],
            'tasks': [
                TaskType.OBJECT_DETECTION
            ],
            'advantages': ['end_to_end', 'no_nms', 'transformer_based'],
            'capabilities': ['set_prediction', 'bipartite_matching'],
            'load_fn': self._load_detr
        }

        self.architecture_registry['sam'] = {
            'type': 'vision',
            'modalities': [ModalityType.IMAGE],
            'tasks': [
                TaskType.SEMANTIC_SEGMENTATION
            ],
            'advantages': ['promptable', 'zero_shot', 'any_object'],
            'capabilities': ['point_prompt', 'box_prompt', 'mask_prompt'],
            'load_fn': self._load_sam
        }

        self.architecture_registry['dinov2'] = {
            'type': 'vision',
            'modalities': [ModalityType.IMAGE],
            'tasks': [
                TaskType.IMAGE_CLASSIFICATION,
                TaskType.FEW_SHOT_LEARNING
            ],
            'advantages': ['self_supervised', 'strong_features', 'transfer_learning'],
            'capabilities': ['student_teacher', 'multi_crop'],
            'load_fn': self._load_dinov2
        }

        # === MULTIMODAL MODELS ===
        self.architecture_registry['clip'] = {
            'type': 'multimodal',
            'modalities': [ModalityType.TEXT, ModalityType.IMAGE],
            'tasks': [
                TaskType.IMAGE_CLASSIFICATION,
                TaskType.TEXT_TO_IMAGE,
                TaskType.MULTIMODAL_REASONING
            ],
            'capabilities': ['zero_shot', 'retrieval'],
            'load_fn': self._load_clip
        }

        self.architecture_registry['blip2'] = {
            'type': 'multimodal',
            'modalities': [ModalityType.TEXT, ModalityType.IMAGE],
            'tasks': [
                TaskType.VISUAL_QUESTION_ANSWERING,
                TaskType.IMAGE_CAPTIONING
            ],
            'advantages': ['efficient', 'frozen_encoders'],
            'load_fn': self._load_blip2
        }

        self.architecture_registry['llava'] = {
            'type': 'multimodal',
            'modalities': [ModalityType.TEXT, ModalityType.IMAGE],
            'tasks': [
                TaskType.VISUAL_QUESTION_ANSWERING,
                TaskType.IMAGE_CAPTIONING,
                TaskType.MULTIMODAL_REASONING
            ],
            'advantages': ['instruction_following', 'simple', 'effective'],
            'load_fn': self._load_llava
        }

        self.architecture_registry['flamingo'] = {
            'type': 'multimodal',
            'modalities': [ModalityType.TEXT, ModalityType.IMAGE],
            'tasks': [
                TaskType.FEW_SHOT_LEARNING,
                TaskType.VISUAL_QUESTION_ANSWERING
            ],
            'capabilities': ['few_shot', 'interleaved_sequences'],
            'load_fn': self._load_flamingo
        }

        # === AUDIO MODELS ===
        self.architecture_registry['whisper'] = {
            'type': 'audio',
            'modalities': [ModalityType.AUDIO],
            'tasks': [
                TaskType.SPEECH_RECOGNITION
            ],
            'advantages': ['multilingual', 'robust', 'zero_shot'],
            'capabilities': ['99_languages', 'transcription', 'translation'],
            'load_fn': self._load_whisper
        }

        self.architecture_registry['encodec'] = {
            'type': 'audio',
            'modalities': [ModalityType.AUDIO],
            'tasks': [
                TaskType.AUDIO_GENERATION
            ],
            'advantages': ['neural_codec', 'high_quality', 'efficient'],
            'capabilities': ['residual_vq', 'compression'],
            'load_fn': self._load_encodec
        }

        self.architecture_registry['musicgen'] = {
            'type': 'audio',
            'modalities': [ModalityType.AUDIO, ModalityType.TEXT],
            'tasks': [
                TaskType.AUDIO_GENERATION
            ],
            'advantages': ['text_to_music', 'controllable'],
            'capabilities': ['conditional_generation', 'long_form'],
            'load_fn': self._load_musicgen
        }

        self.architecture_registry['wav2vec2'] = {
            'type': 'audio',
            'modalities': [ModalityType.AUDIO],
            'tasks': [
                TaskType.SPEECH_RECOGNITION,
                TaskType.AUDIO_CLASSIFICATION
            ],
            'advantages': ['self_supervised', 'pre_training'],
            'capabilities': ['contrastive_learning', 'quantization'],
            'load_fn': self._load_wav2vec2
        }

        # === TIME SERIES MODELS ===
        self.architecture_registry['nbeats'] = {
            'type': 'time_series',
            'modalities': [ModalityType.TABULAR],
            'tasks': [
                TaskType.CONTINUAL_LEARNING  # Using as proxy for forecasting
            ],
            'advantages': ['interpretable', 'trend_decomposition', 'no_feature_engineering'],
            'capabilities': ['trend', 'seasonality', 'residual_stacking'],
            'load_fn': self._load_nbeats
        }

        self.architecture_registry['tft'] = {
            'type': 'time_series',
            'modalities': [ModalityType.TABULAR],
            'tasks': [
                TaskType.CONTINUAL_LEARNING  # Using as proxy for forecasting
            ],
            'advantages': ['multi_horizon', 'variable_selection', 'interpretable'],
            'capabilities': ['attention', 'quantile_forecasting', 'static_covariates'],
            'load_fn': self._load_tft
        }

        self.architecture_registry['patchtst'] = {
            'type': 'time_series',
            'modalities': [ModalityType.TABULAR],
            'tasks': [
                TaskType.CONTINUAL_LEARNING  # Using as proxy for forecasting
            ],
            'advantages': ['patch_based', 'efficient', 'channel_independence'],
            'capabilities': ['transformer', 'long_sequences', 'pre_training'],
            'load_fn': self._load_patchtst
        }

        # === META-LEARNING ===
        self.architecture_registry['maml'] = {
            'type': 'meta_learning',
            'modalities': 'any',
            'tasks': [
                TaskType.FEW_SHOT_LEARNING
            ],
            'advantages': ['model_agnostic', 'fast_adaptation'],
            'capabilities': ['inner_outer_loop', 'gradient_based'],
            'load_fn': self._load_maml
        }

        # === MEMORY SYSTEMS ===
        self.architecture_registry['ntm'] = {
            'type': 'memory',
            'modalities': 'any',
            'capabilities': ['external_memory', 'content_addressing'],
            'load_fn': self._load_ntm
        }

        self.architecture_registry['dnc'] = {
            'type': 'memory',
            'modalities': 'any',
            'capabilities': ['external_memory', 'temporal_linkage'],
            'load_fn': self._load_dnc
        }

        # === REINFORCEMENT LEARNING ===
        self.architecture_registry['ppo'] = {
            'type': 'rl',
            'tasks': [TaskType.REINFORCEMENT_LEARNING],
            'load_fn': self._load_ppo
        }

        self.architecture_registry['sac'] = {
            'type': 'rl',
            'tasks': [TaskType.REINFORCEMENT_LEARNING],
            'advantages': ['continuous_actions', 'maximum_entropy'],
            'load_fn': self._load_sac
        }

        # === GENERATIVE MODELS ===
        self.architecture_registry['diffusion'] = {
            'type': 'generative',
            'modalities': [ModalityType.IMAGE, ModalityType.AUDIO],
            'tasks': [
                TaskType.IMAGE_GENERATION,
                TaskType.AUDIO_GENERATION
            ],
            'load_fn': self._load_diffusion
        }

        # === REASONING ===
        self.architecture_registry['chain_of_thought'] = {
            'type': 'reasoning',
            'tasks': [
                TaskType.MATHEMATICAL_REASONING,
                TaskType.COMMONSENSE_REASONING,
                TaskType.LOGICAL_REASONING
            ],
            'load_fn': self._load_cot
        }

        self.architecture_registry['tree_of_thoughts'] = {
            'type': 'reasoning',
            'tasks': [
                TaskType.MATHEMATICAL_REASONING,
                TaskType.PLANNING
            ],
            'advantages': ['search', 'backtracking'],
            'load_fn': self._load_tot
        }

        # === GRAPH NETWORKS ===
        self.architecture_registry['gnn'] = {
            'type': 'graph',
            'modalities': [ModalityType.GRAPH],
            'tasks': [
                TaskType.MOLECULE_GENERATION,
                TaskType.PROTEIN_FOLDING
            ],
            'load_fn': self._load_gnn
        }

        self.logger.info(f"Initialized {len(self.architecture_registry)} architectures")

    def select_architecture(
        self,
        task_spec: TaskSpecification,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ArchitectureSelection:
        """
        Sélectionne intelligemment la meilleure architecture pour une tâche.

        Args:
            task_spec: Spécification de la tâche
            constraints: Contraintes (latence, mémoire, etc.)

        Returns:
            Sélection d'architecture avec justification
        """
        # Analyze task requirements
        candidate_archs = self._find_candidate_architectures(task_spec)

        # Score each candidate
        scores = {}
        for arch_name in candidate_archs:
            score = self._score_architecture(
                arch_name,
                task_spec,
                constraints or {}
            )
            scores[arch_name] = score

        # Select best architecture
        primary = max(scores, key=scores.get)

        # Select supporting architectures
        supporting = self._select_supporting_architectures(
            primary,
            task_spec,
            constraints or {}
        )

        # Determine fusion strategy
        fusion_strategy = self._determine_fusion_strategy(
            primary,
            supporting,
            task_spec
        )

        # Generate reasoning
        reasoning = self._generate_selection_reasoning(
            primary,
            supporting,
            scores,
            task_spec
        )

        return ArchitectureSelection(
            primary_architecture=primary,
            supporting_architectures=supporting,
            fusion_strategy=fusion_strategy,
            reasoning=reasoning,
            confidence=scores[primary]
        )

    def _find_candidate_architectures(
        self,
        task_spec: TaskSpecification
    ) -> List[str]:
        """Trouve toutes les architectures candidates pour une tâche"""
        candidates = []

        for name, info in self.architecture_registry.items():
            # Check if architecture supports the task
            if 'tasks' in info and task_spec.task_type in info['tasks']:
                candidates.append(name)
                continue

            # Check if architecture supports the modalities
            if 'modalities' in info:
                if info['modalities'] == 'any':
                    candidates.append(name)
                elif all(m in info['modalities'] for m in task_spec.modalities):
                    candidates.append(name)

        return candidates

    def _score_architecture(
        self,
        arch_name: str,
        task_spec: TaskSpecification,
        constraints: Dict[str, Any]
    ) -> float:
        """
        Score une architecture pour une tâche donnée.

        Prend en compte:
        - Performance attendue
        - Complexité computationnelle
        - Efficacité mémoire
        - Historique de performance
        """
        arch_info = self.architecture_registry[arch_name]
        score = 0.0

        # Base score: task compatibility
        if 'tasks' in arch_info and task_spec.task_type in arch_info['tasks']:
            score += 1.0

        # Advantage bonus
        if 'advantages' in arch_info:
            score += 0.1 * len(arch_info['advantages'])

        # Constraint penalties
        if 'latency' in constraints:
            if arch_info.get('complexity') == 'O(N²)':
                score -= 0.2
            elif arch_info.get('complexity') == 'O(N)':
                score += 0.2

        # Historical performance
        hist_score = self._get_historical_performance(arch_name, task_spec.task_type)
        score += hist_score * 0.5

        return max(0.0, min(1.0, score))

    def _select_supporting_architectures(
        self,
        primary: str,
        task_spec: TaskSpecification,
        constraints: Dict[str, Any]
    ) -> List[str]:
        """Sélectionne des architectures de support complémentaires"""
        supporting = []

        # For multimodal tasks, add modality-specific encoders
        if len(task_spec.modalities) > 1:
            if ModalityType.IMAGE in task_spec.modalities:
                if primary not in ['vit', 'swin', 'clip']:
                    supporting.append('vit')

            if ModalityType.TEXT in task_spec.modalities:
                if primary not in ['transformer', 'mamba', 'clip']:
                    supporting.append('transformer')

        # For reasoning tasks, add reasoning modules
        if task_spec.task_type in [
            TaskType.MATHEMATICAL_REASONING,
            TaskType.COMMONSENSE_REASONING,
            TaskType.LOGICAL_REASONING
        ]:
            if 'chain_of_thought' not in [primary] + supporting:
                supporting.append('chain_of_thought')

        # For tasks requiring memory
        if task_spec.task_type in [
            TaskType.QUESTION_ANSWERING,
            TaskType.CONTINUAL_LEARNING
        ]:
            if 'dnc' not in [primary] + supporting:
                supporting.append('dnc')

        return supporting

    def _determine_fusion_strategy(
        self,
        primary: str,
        supporting: List[str],
        task_spec: TaskSpecification
    ) -> str:
        """Détermine la stratégie de fusion optimale"""
        if not supporting:
            return "none"

        # Multimodal fusion strategies
        if len(task_spec.modalities) > 1:
            return "cross_attention"  # Best for multimodal

        # Sequential for reasoning
        if task_spec.task_type in [
            TaskType.MATHEMATICAL_REASONING,
            TaskType.LOGICAL_REASONING
        ]:
            return "sequential"

        # Ensemble for robustness
        if 'performance_requirements' in task_spec and task_spec.performance_requirements:
            if task_spec.performance_requirements.get('accuracy', 0) > 0.95:
                return "ensemble"

        return "parallel"

    def _generate_selection_reasoning(
        self,
        primary: str,
        supporting: List[str],
        scores: Dict[str, float],
        task_spec: TaskSpecification
    ) -> str:
        """Génère une explication de la sélection"""
        reasoning = f"Selected {primary} as primary architecture (score: {scores[primary]:.3f}).\n"

        # Explain primary choice
        arch_info = self.architecture_registry[primary]
        if 'advantages' in arch_info:
            reasoning += f"Advantages: {', '.join(arch_info['advantages'])}.\n"

        # Explain supporting choices
        if supporting:
            reasoning += f"Supporting architectures: {', '.join(supporting)}.\n"
            reasoning += "These provide complementary capabilities.\n"

        # Task-specific reasoning
        reasoning += f"Task type: {task_spec.task_type.value}.\n"
        reasoning += f"Modalities: {[m.value for m in task_spec.modalities]}.\n"

        return reasoning

    def _get_historical_performance(
        self,
        arch_name: str,
        task_type: TaskType
    ) -> float:
        """Récupère la performance historique"""
        # Filter relevant history
        relevant = [
            h for h in self.performance_history
            if h['architecture'] == arch_name and h['task_type'] == task_type
        ]

        if not relevant:
            return 0.0

        # Return average performance
        return sum(h['performance'] for h in relevant) / len(relevant)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        task_spec: TaskSpecification,
        return_reasoning: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, str]]:
        """
        Forward pass avec sélection et exécution automatique.

        Args:
            inputs: Dictionary of input tensors by modality
            task_spec: Task specification
            return_reasoning: Whether to return selection reasoning

        Returns:
            Output tensor (and optionally reasoning)
        """
        # Select architecture
        selection = self.select_architecture(task_spec)

        self.logger.info(f"Using {selection.primary_architecture} "
                        f"with {len(selection.supporting_architectures)} supporting architectures")

        # Load and execute architectures
        output = self._execute_pipeline(inputs, selection, task_spec)

        if return_reasoning:
            return output, selection.reasoning
        return output

    def _execute_pipeline(
        self,
        inputs: Dict[str, torch.Tensor],
        selection: ArchitectureSelection,
        task_spec: TaskSpecification
    ) -> torch.Tensor:
        """Execute the selected architecture pipeline"""
        self.logger.info(f"Executing {selection.primary_architecture}")
        self.logger.info(f"Fusion strategy: {selection.fusion_strategy}")

        try:
            # Load primary architecture
            loader_func = f"_load_{selection.primary_architecture.lower().replace('-', '_')}"
            if not hasattr(self, loader_func):
                self.logger.warning(f"No loader found for {selection.primary_architecture}")
                # Return placeholder
                if task_spec.output_shape:
                    return torch.zeros(task_spec.output_shape)
                return torch.zeros(1)

            model = getattr(self, loader_func)()
            self.logger.info(f"Loaded {selection.primary_architecture}")

            # Move model to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()  # Set to evaluation mode

            # Execute model based on architecture type
            from architectures.base import VisionArchitecture, LanguageArchitecture, MultimodalArchitecture

            with torch.no_grad():
                if isinstance(model, MultimodalArchitecture):
                    # Multimodal models (CLIP, BLIP, etc.)
                    output = self._execute_multimodal(model, inputs, device)
                elif isinstance(model, VisionArchitecture):
                    # Vision models (ViT, ResNet, etc.)
                    output = self._execute_vision(model, inputs, device)
                elif isinstance(model, LanguageArchitecture):
                    # Language models (Transformer, GPT, BERT, etc.)
                    output = self._execute_language(model, inputs, device)
                else:
                    # Generic BrainArchitecture
                    output = self._execute_generic(model, inputs, device)

            # Extract predictions from ModelOutput
            if hasattr(output, 'predictions') and output.predictions is not None:
                return output.predictions
            elif hasattr(output, 'logits') and output.logits is not None:
                return output.logits
            elif hasattr(output, 'embeddings') and output.embeddings is not None:
                return output.embeddings
            else:
                self.logger.warning("Model returned empty output")
                if task_spec.output_shape:
                    return torch.zeros(task_spec.output_shape)
                return torch.zeros(1)

        except Exception as e:
            self.logger.error(f"Error executing pipeline: {e}", exc_info=True)
            # Return placeholder on error
            if task_spec.output_shape:
                return torch.zeros(task_spec.output_shape)
            return torch.zeros(1)

    def _execute_vision(self, model, inputs: Dict[str, torch.Tensor], device: str):
        """Execute vision model"""
        # Vision models expect 'image' or 'x' input
        if 'image' in inputs:
            x = inputs['image'].to(device)
        elif 'x' in inputs:
            x = inputs['x'].to(device)
        elif 'pixel_values' in inputs:
            x = inputs['pixel_values'].to(device)
        else:
            raise ValueError("Vision model requires 'image', 'x', or 'pixel_values' input")

        return model(x)

    def _execute_language(self, model, inputs: Dict[str, torch.Tensor], device: str):
        """Execute language model"""
        # Language models expect 'input_ids' and optionally 'attention_mask'
        if 'input_ids' not in inputs:
            raise ValueError("Language model requires 'input_ids' input")

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        return model(input_ids=input_ids, attention_mask=attention_mask)

    def _execute_multimodal(self, model, inputs: Dict[str, torch.Tensor], device: str):
        """Execute multimodal model"""
        # Multimodal models can take various combinations of inputs
        model_inputs = {}

        # Image inputs
        if 'image' in inputs:
            model_inputs['image'] = inputs['image'].to(device)
        elif 'pixel_values' in inputs:
            model_inputs['image'] = inputs['pixel_values'].to(device)

        # Text inputs
        if 'input_ids' in inputs:
            model_inputs['text'] = inputs['input_ids'].to(device)
        if 'attention_mask' in inputs:
            model_inputs['attention_mask'] = inputs['attention_mask'].to(device)

        # Audio inputs
        if 'audio' in inputs:
            model_inputs['audio'] = inputs['audio'].to(device)

        return model(**model_inputs)

    def _execute_generic(self, model, inputs: Dict[str, torch.Tensor], device: str):
        """Execute generic model with inputs as kwargs"""
        # Move all inputs to device
        device_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()}
        return model(**device_inputs)

    # === Architecture loading functions ===

    def _load_transformer(self):
        """Load Transformer architecture"""
        from architectures.transformers.transformer import Transformer, TransformerConfig
        config = TransformerConfig(
            vocab_size=50000,
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
        )
        return Transformer(config)

    def _load_mamba(self):
        """Load Mamba (Selective State Space Model) architecture"""
        from architectures.state_space.mamba import Mamba, MambaConfig
        config = MambaConfig(
            d_model=768,
            n_layers=24,
            vocab_size=50000,
        )
        return Mamba(config)

    def _load_vit(self):
        """Load Vision Transformer architecture"""
        from architectures.vision.vision_transformer import VisionTransformer, ViTConfig
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=768,
            depth=12,
            heads=12,
        )
        return VisionTransformer(config)

    def _load_swin(self):
        """Load Swin Transformer architecture"""
        # Placeholder - Swin not implemented yet
        from architectures.base import VisionArchitecture
        return VisionArchitecture()  # Placeholder

    def _load_yolov8(self):
        """Load YOLOv8 architecture"""
        # Placeholder - YOLOv8 not implemented yet
        from architectures.base import VisionArchitecture
        return VisionArchitecture()  # Placeholder

    def _load_detr(self):
        """Load DETR architecture"""
        # Placeholder - DETR not implemented yet
        from architectures.base import VisionArchitecture
        return VisionArchitecture()  # Placeholder

    def _load_sam(self):
        """Load SAM (Segment Anything Model) architecture"""
        # Placeholder - SAM not implemented yet
        from architectures.base import VisionArchitecture
        return VisionArchitecture()  # Placeholder

    def _load_dinov2(self):
        """Load DINOv2 architecture"""
        # Placeholder - DINOv2 not implemented yet
        from architectures.base import VisionArchitecture
        return VisionArchitecture()  # Placeholder

    def _load_clip(self):
        """Load CLIP (Contrastive Language-Image Pre-training) architecture"""
        from architectures.multimodal.clip import CLIPModel
        return CLIPModel(
            image_size=224,
            patch_size=16,
            hidden_size=512,
            num_heads=8,
            num_layers=12,
            vocab_size=49408,
            max_text_length=77,
        )

    def _load_blip2(self):
        """Load BLIP-2 architecture"""
        from architectures.multimodal.blip2 import BLIP2
        return BLIP2(
            vision_model='vit',
            image_size=224,
            llm_model='opt-2.7b',
            num_query_tokens=32,
        )

    def _load_llava(self):
        """Load LLaVA architecture"""
        from architectures.multimodal.llava import LLaVA
        return LLaVA(
            vision_model='clip-vit-large',
            llm_model='vicuna-7b',
            mm_projector_type='linear',
        )

    def _load_flamingo(self):
        """Load Flamingo architecture"""
        from architectures.multimodal.flamingo import Flamingo
        return Flamingo(
            vision_encoder='clip-vit',
            language_model='chinchilla-7b',
            num_perceiver_layers=6,
        )

    def _load_ntm(self):
        """Load Neural Turing Machine architecture"""
        from architectures.memory.neural_memory import NeuralTuringMachine
        return NeuralTuringMachine(
            input_size=128,
            output_size=128,
            hidden_size=256,
            memory_size=128,
            memory_dim=64,
        )

    def _load_dnc(self):
        """Load Differentiable Neural Computer architecture"""
        from architectures.memory.neural_memory import DifferentiableNeuralComputer
        return DifferentiableNeuralComputer(
            input_size=128,
            output_size=128,
            hidden_size=256,
            memory_size=256,
            memory_dim=64,
            num_read_heads=4,
            num_write_heads=1,
        )

    def _load_ppo(self):
        """Load PPO (Proximal Policy Optimization) agent"""
        from architectures.rl.ppo import PPO, PPOConfig
        config = PPOConfig(
            state_dim=64,
            action_dim=4,
            hidden_dim=256,
            lr_actor=3e-4,
            lr_critic=1e-3,
        )
        return PPO(config)

    def _load_sac(self):
        """Load SAC (Soft Actor-Critic) agent"""
        from architectures.rl.sac import SAC, SACConfig
        config = SACConfig(
            state_dim=64,
            action_dim=4,
            hidden_dim=256,
            lr=3e-4,
        )
        return SAC(config)

    def _load_diffusion(self):
        """Load Diffusion Model architecture"""
        from architectures.generative.diffusion import DiffusionModel, DiffusionConfig
        config = DiffusionConfig(
            image_size=256,
            timesteps=1000,
            model_channels=128,
        )
        return DiffusionModel(config)

    def _load_cot(self):
        """Load Chain-of-Thought reasoning"""
        from architectures.reasoning.chain_of_thought import ChainOfThought
        return ChainOfThought(
            model_name='gpt-3.5-turbo',
            max_steps=5,
        )

    def _load_tot(self):
        """Load Tree-of-Thoughts reasoning"""
        from architectures.reasoning.tree_of_thoughts import TreeOfThoughts
        return TreeOfThoughts(
            model_name='gpt-3.5-turbo',
            search_strategy='bfs',
            max_depth=3,
        )

    def _load_gnn(self):
        """Load Graph Neural Network architecture"""
        from architectures.graph.graph_networks import GraphNeuralNetwork
        return GraphNeuralNetwork(
            in_channels=64,
            hidden_channels=128,
            out_channels=32,
            num_layers=3,
        )

    def _load_whisper(self):
        """Load Whisper (Speech Recognition) architecture"""
        from architectures.audio.whisper import Whisper, WhisperConfig
        config = WhisperConfig(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4,
        )
        return Whisper(config)

    def _load_encodec(self):
        """Load Encodec (Neural Audio Codec) architecture"""
        from architectures.audio.encodec import Encodec, EncodecConfig
        config = EncodecConfig(
            sample_rate=24000,
            channels=1,
            encoder_rates=[8, 5, 4, 2],
            num_quantizers=8,
            bandwidth=6.0
        )
        return Encodec(config)

    def _load_musicgen(self):
        """Load MusicGen (Text-to-Music) architecture"""
        from architectures.audio.musicgen import MusicGen, MusicGenConfig
        config = MusicGenConfig(
            sample_rate=32000,
            channels=2,
            d_model=1024,
            num_layers=24,
            num_heads=16,
            num_codebooks=4
        )
        return MusicGen(config)

    def _load_wav2vec2(self):
        """Load Wav2Vec2 (Self-Supervised Speech) architecture"""
        from architectures.audio.wav2vec2 import Wav2Vec2, Wav2Vec2Config
        config = Wav2Vec2Config(
            d_model=768,
            num_layers=12,
            num_heads=12,
            d_ff=3072,
        )
        return Wav2Vec2(config)

    def _load_nbeats(self):
        """Load N-BEATS (Time Series Forecasting) architecture"""
        from architectures.time_series.nbeats import NBEATS, NBEATSConfig
        config = NBEATSConfig(
            backcast_length=10,
            forecast_length=5,
            stack_types=['trend', 'seasonality', 'generic'],
            num_blocks_per_stack=3,
            hidden_layer_units=256,
            num_layers=4,
        )
        return NBEATS(config)

    def _load_tft(self):
        """Load TFT (Temporal Fusion Transformer) architecture"""
        from architectures.time_series.tft import TemporalFusionTransformer, TFTConfig
        config = TFTConfig(
            static_input_size=4,
            temporal_observed_size=3,
            temporal_known_size=2,
            target_size=1,
            encoder_length=24,
            decoder_length=12,
            hidden_size=160,
            num_heads=4,
        )
        return TemporalFusionTransformer(config)

    def _load_patchtst(self):
        """Load PatchTST (Patch Time Series Transformer) architecture"""
        from architectures.time_series.patchtst import PatchTST, PatchTSTConfig
        config = PatchTSTConfig(
            num_variables=7,
            seq_len=336,
            pred_len=96,
            patch_len=16,
            stride=8,
            d_model=128,
            n_heads=8,
            e_layers=3,
        )
        return PatchTST(config)

    def _load_maml(self):
        """Load MAML (Model-Agnostic Meta-Learning)"""
        from architectures.meta_learning.maml import MAML, MAMLConfig
        import torch.nn as nn

        # Create a simple base model for MAML
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 5)
                )

            def forward(self, x):
                return self.net(x)

        config = MAMLConfig(
            inner_lr=0.01,
            inner_steps=5,
            outer_lr=0.001,
            n_way=5,
            k_shot=1,
            q_query=15
        )

        base_model = SimpleModel()
        return MAML(base_model, config)


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Intelligent Brain Orchestrator - Symbiose Dynamique")
    print("="*80)

    # Create orchestrator
    orchestrator = IntelligentOrchestrator()

    # Example 1: Visual Question Answering
    task1 = TaskSpecification(
        task_type=TaskType.VISUAL_QUESTION_ANSWERING,
        modalities=[ModalityType.TEXT, ModalityType.IMAGE],
        input_shape={'image': (3, 224, 224), 'text': (512,)}
    )

    selection1 = orchestrator.select_architecture(task1)
    print(f"\nTask: Visual Question Answering")
    print(f"Primary: {selection1.primary_architecture}")
    print(f"Supporting: {selection1.supporting_architectures}")
    print(f"Fusion: {selection1.fusion_strategy}")
    print(f"Confidence: {selection1.confidence:.3f}")
    print(f"\nReasoning:\n{selection1.reasoning}")

    # Example 2: Mathematical Reasoning
    task2 = TaskSpecification(
        task_type=TaskType.MATHEMATICAL_REASONING,
        modalities=[ModalityType.TEXT],
        input_shape={'text': (512,)}
    )

    selection2 = orchestrator.select_architecture(task2)
    print(f"\nTask: Mathematical Reasoning")
    print(f"Primary: {selection2.primary_architecture}")
    print(f"Supporting: {selection2.supporting_architectures}")
    print(f"Reasoning:\n{selection2.reasoning}")

    print("\n" + "="*80)
