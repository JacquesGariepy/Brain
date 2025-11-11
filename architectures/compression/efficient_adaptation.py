"""
Efficient Model Adaptation and Compression Techniques

Implementations:
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Adapter Layers
- Prefix Tuning
- Prompt Tuning
- Model Pruning
- Quantization (INT8, INT4, FP16)
- Knowledge Distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer.

    Adds trainable low-rank decomposition to frozen pretrained weights.
    W = W_0 + BA, where B and A are low-rank matrices.

    Dramatically reduces trainable parameters while maintaining performance.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            LoRA output (..., out_features)
        """
        result = self.dropout(x) @ self.lora_A @ self.lora_B
        return result * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Combines frozen pretrained weights with trainable low-rank adaptation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        # Frozen pretrained layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

        # LoRA adaptation
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA"""
        return self.linear(x) + self.lora(x)

    def merge_weights(self):
        """
        Merge LoRA weights into base layer for inference.

        This eliminates the overhead of LoRA during inference.
        """
        with torch.no_grad():
            lora_weight = (self.lora.lora_A @ self.lora.lora_B) * self.lora.scaling
            self.linear.weight.data += lora_weight.T


class QLoRALinear(nn.Module):
    """
    Quantized Low-Rank Adaptation (QLoRA).

    Combines 4-bit quantization with LoRA for extreme efficiency.
    Enables fine-tuning 65B models on a single GPU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bits: int = 4,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # Quantized weight storage
        # Store in INT4/INT8 with scales
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # LoRA adaptation (FP16/BF16)
        self.lora = LoRALayer(in_features, out_features, rank, alpha)

    def quantize_weight(self, weight: torch.Tensor):
        """
        Quantize weight to 4-bit or 8-bit.

        Uses absmax quantization.
        """
        max_val = weight.abs().max(dim=-1, keepdim=True)[0]
        scale = max_val / (2 ** (self.bits - 1) - 1)

        weight_quantized = (weight / scale).round().to(torch.int8)

        self.weight_quantized.copy_(weight_quantized)
        self.weight_scale.copy_(scale.squeeze())

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize weight for computation"""
        return self.weight_quantized.float() * self.weight_scale.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weight and LoRA"""
        # Dequantize and compute
        weight = self.dequantize_weight()
        output = F.linear(x, weight, self.bias)

        # Add LoRA
        output = output + self.lora(x)

        return output


class AdapterLayer(nn.Module):
    """
    Adapter layer for parameter-efficient fine-tuning.

    Adds bottleneck layers within transformer blocks.
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.down_project = nn.Linear(d_model, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Initialize to near-identity
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            Residual to add to original input
        """
        h = self.down_project(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_project(h)
        return h


class PrefixTuning(nn.Module):
    """
    Prefix Tuning: Optimizing Continuous Prompts.

    Prepends trainable prefix vectors to keys and values in attention.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_length: int = 10,
        prefix_projection: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_length = prefix_length

        d_model = num_heads * head_dim

        if prefix_projection:
            # Use MLP reparameterization for better optimization
            self.prefix_encoder = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.Tanh(),
                nn.Linear(512, num_layers * 2 * d_model)  # 2 for K and V
            )
            # Input embeddings to project
            self.prefix_tokens = nn.Parameter(torch.randn(prefix_length, d_model))
        else:
            # Direct parameterization
            self.prefix_params = nn.Parameter(
                torch.randn(num_layers, 2, prefix_length, num_heads, head_dim)
            )

        self.prefix_projection = prefix_projection

    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prefix key and value for all layers.

        Returns:
            prefix_keys: (num_layers, batch, num_heads, prefix_length, head_dim)
            prefix_values: (num_layers, batch, num_heads, prefix_length, head_dim)
        """
        if self.prefix_projection:
            # Project prefix tokens
            prefix = self.prefix_encoder(self.prefix_tokens)  # (prefix_length, num_layers * 2 * d_model)

            # Reshape
            prefix = prefix.view(
                self.prefix_length,
                self.num_layers,
                2,
                self.num_heads,
                self.head_dim
            )
            prefix = prefix.permute(1, 2, 0, 3, 4)  # (num_layers, 2, prefix_length, num_heads, head_dim)

            # Expand batch
            prefix = prefix.unsqueeze(1).expand(-1, batch_size, -1, -1, -1, -1)

            prefix_keys = prefix[:, :, 0]  # (num_layers, batch, prefix_length, num_heads, head_dim)
            prefix_values = prefix[:, :, 1]

        else:
            # Direct params
            prefix_keys = self.prefix_params[:, 0].unsqueeze(1).expand(-1, batch_size, -1, -1, -1)
            prefix_values = self.prefix_params[:, 1].unsqueeze(1).expand(-1, batch_size, -1, -1, -1)

        return prefix_keys, prefix_values


class PromptTuning(nn.Module):
    """
    Prompt Tuning: Soft prompt optimization.

    Prepends trainable continuous prompt embeddings to input.
    """

    def __init__(
        self,
        num_tokens: int = 20,
        embedding_dim: int = 768,
        init_from_vocab: bool = False,
        vocab_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_tokens = num_tokens

        if init_from_vocab and vocab_embeddings is not None:
            # Initialize from random vocabulary embeddings
            indices = torch.randperm(len(vocab_embeddings))[:num_tokens]
            prompt_embeddings = vocab_embeddings[indices].clone()
        else:
            # Random initialization
            prompt_embeddings = torch.randn(num_tokens, embedding_dim)

        self.prompt_embeddings = nn.Parameter(prompt_embeddings)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get prompt embeddings for batch.

        Returns:
            Prompt embeddings (batch, num_tokens, embedding_dim)
        """
        return self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)


class MagnitudePruning:
    """
    Magnitude-based pruning for model compression.

    Prunes weights with smallest magnitudes.
    """

    @staticmethod
    def prune_weights(
        model: nn.Module,
        amount: float = 0.5,
        global_pruning: bool = True
    ):
        """
        Prune model weights.

        Args:
            model: Model to prune
            amount: Fraction of weights to prune (0-1)
            global_pruning: If True, prune globally. If False, prune per-layer.
        """
        import torch.nn.utils.prune as prune

        if global_pruning:
            # Global pruning across all layers
            parameters_to_prune = []
            for module in model.modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    parameters_to_prune.append((module, 'weight'))

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        else:
            # Per-layer pruning
            for module in model.modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=amount)

    @staticmethod
    def make_permanent(model: nn.Module):
        """Make pruning permanent by removing reparameterization"""
        import torch.nn.utils.prune as prune

        for module in model.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass


class DynamicQuantization:
    """
    Dynamic quantization for model compression.

    Quantizes weights to INT8 while keeping activations in FP32.
    """

    @staticmethod
    def quantize_model(model: nn.Module) -> nn.Module:
        """
        Quantize model to INT8.

        Args:
            model: Model to quantize

        Returns:
            Quantized model
        """
        return torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )


class KnowledgeDistillation:
    """
    Knowledge distillation for model compression.

    Train smaller student model to mimic larger teacher model.
    """

    @staticmethod
    def distillation_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 3.0,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            labels: Ground truth labels
            temperature: Softmax temperature for distillation
            alpha: Weight for distillation loss vs hard label loss

        Returns:
            Combined loss
        """
        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)

        # Soft label loss (KL divergence)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

        # Combined loss
        loss = alpha * soft_loss + (1 - alpha) * hard_loss

        return loss


class StructuredPruning:
    """
    Structured pruning: Prune entire channels/neurons.

    More hardware-friendly than unstructured pruning.
    """

    @staticmethod
    def prune_channels(
        conv: nn.Conv2d,
        amount: float = 0.5
    ) -> nn.Conv2d:
        """
        Prune output channels based on L1 norm.

        Args:
            conv: Convolution layer to prune
            amount: Fraction of channels to prune

        Returns:
            Pruned convolution layer
        """
        # Compute L1 norm of each output channel
        num_channels = conv.out_channels
        channel_norms = conv.weight.data.abs().sum(dim=(1, 2, 3))

        # Select channels to keep
        num_keep = int(num_channels * (1 - amount))
        _, indices = torch.topk(channel_norms, num_keep)
        indices = indices.sort()[0]

        # Create new layer with pruned channels
        new_conv = nn.Conv2d(
            conv.in_channels,
            num_keep,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            bias=conv.bias is not None
        )

        # Copy weights
        new_conv.weight.data = conv.weight.data[indices]
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data[indices]

        return new_conv
