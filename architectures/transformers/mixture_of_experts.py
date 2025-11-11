"""
Mixture of Experts (MoE) - SOTA for Scaling

Implements:
- Sparse MoE (like in Switch Transformer, GShard)
- Expert routing with load balancing
- Top-K routing
- Differentiable Top-K gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class Expert(nn.Module):
    """
    Single expert network (FFN).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert"""
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TopKGating(nn.Module):
    """
    Top-K gating mechanism for routing tokens to experts.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gating weights for routing.

        Args:
            x: Input tokens (batch * seq_len, d_model)

        Returns:
            top_k_indices: Expert indices (batch * seq_len, top_k)
            top_k_gates: Gating weights (batch * seq_len, top_k)
            load_balancing_loss: Loss for load balancing
        """
        # Compute logits
        logits = self.gate(x)  # (batch * seq_len, num_experts)

        # Add noise during training (for exploration)
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax over top-k
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # Load balancing loss
        # Encourage uniform distribution of tokens across experts
        gates_softmax = F.softmax(logits, dim=-1)  # (batch * seq_len, num_experts)
        expert_usage = gates_softmax.mean(dim=0)  # (num_experts,)

        # Auxiliary loss: variance of expert usage
        load_balancing_loss = (expert_usage.var() * self.num_experts)

        return top_k_indices, top_k_gates, load_balancing_loss


class MixtureOfExperts(nn.Module):
    """
    Sparse Mixture of Experts layer.

    Each token is routed to top-K experts.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        # Gating network
        self.gate = TopKGating(d_model, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE.

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            load_loss: Load balancing loss
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten for routing
        x_flat = x.reshape(-1, d_model)  # (batch * seq_len, d_model)

        # Get routing decisions
        top_k_indices, top_k_gates, load_loss = self.gate(x_flat, self.training)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Route tokens to experts
        for i in range(self.top_k):
            # Get expert indices for this k
            expert_indices = top_k_indices[:, i]  # (batch * seq_len,)
            gates = top_k_gates[:, i].unsqueeze(-1)  # (batch * seq_len, 1)

            # Process each expert
            for expert_id in range(self.num_experts):
                # Find tokens routed to this expert
                expert_mask = (expert_indices == expert_id)

                if expert_mask.any():
                    # Get tokens for this expert
                    expert_input = x_flat[expert_mask]

                    # Process through expert
                    expert_output = self.experts[expert_id](expert_input)

                    # Weight by gate
                    expert_output = expert_output * gates[expert_mask]

                    # Add to output
                    output[expert_mask] += expert_output

        # Reshape back
        output = output.reshape(batch_size, seq_len, d_model)

        return output, load_loss * self.load_balance_weight


class SwitchLayer(nn.Module):
    """
    Switch Transformer layer (simplified MoE with top-1 routing).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        dropout: float = 0.1,
        capacity_factor: float = 1.25
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with top-1 routing and capacity constraints.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output, routing_loss
        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len

        # Flatten
        x_flat = x.reshape(num_tokens, d_model)

        # Router logits
        router_logits = self.router(x_flat)  # (num_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-1 routing
        expert_indices = torch.argmax(router_probs, dim=-1)  # (num_tokens,)
        expert_gates = router_probs.max(dim=-1)[0]  # (num_tokens,)

        # Capacity per expert
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert
        for expert_id in range(self.num_experts):
            # Get tokens for this expert
            expert_mask = (expert_indices == expert_id)
            expert_tokens = x_flat[expert_mask]

            # Apply capacity constraint
            if expert_tokens.size(0) > capacity:
                # Keep only top-capacity tokens
                gates_for_expert = expert_gates[expert_mask]
                _, top_indices = torch.topk(gates_for_expert, capacity)
                expert_mask_indices = expert_mask.nonzero(as_tuple=True)[0]
                keep_indices = expert_mask_indices[top_indices]

                # Update mask
                new_mask = torch.zeros_like(expert_mask)
                new_mask[keep_indices] = True
                expert_mask = new_mask

                expert_tokens = x_flat[expert_mask]

            if expert_tokens.size(0) > 0:
                # Process through expert
                expert_output = self.experts[expert_id](expert_tokens)

                # Weight by routing probability
                expert_output = expert_output * expert_gates[expert_mask].unsqueeze(-1)

                # Add to output
                output[expert_mask] = expert_output

        # Routing loss (for load balancing)
        # Encourage uniform routing
        routing_loss = router_probs.sum(dim=0).var() * self.num_experts

        # Reshape
        output = output.reshape(batch_size, seq_len, d_model)

        return output, routing_loss


class MoETransformerBlock(nn.Module):
    """
    Transformer block with Mixture of Experts.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        use_switch: bool = False
    ):
        super().__init__()

        # Standard attention
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MoE or Switch FFN
        if use_switch:
            self.ffn = SwitchLayer(d_model, d_ff, num_experts, dropout)
        else:
            self.ffn = MixtureOfExperts(d_model, d_ff, num_experts, top_k, dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            output, auxiliary_loss
        """
        # Attention
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)

        # MoE FFN
        ffn_out, aux_loss = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)

        return x, aux_loss


class MoETransformer(nn.Module):
    """
    Transformer with Mixture of Experts.

    Scales to massive parameter counts while keeping compute fixed.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_switch: bool = False
    ):
        super().__init__()

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # MoE layers
        self.layers = nn.ModuleList([
            MoETransformerBlock(
                d_model,
                num_heads,
                d_ff,
                num_experts,
                top_k,
                dropout,
                use_switch
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            logits, total_aux_loss
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(pos_ids)

        # MoE layers
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, total_aux_loss
