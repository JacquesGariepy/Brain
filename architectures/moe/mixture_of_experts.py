"""
Mixture of Experts (MoE) Architectures

SOTA sparse models for massive scaling with constant compute.

Implementations:
- Switch Transformer: Sparse routing, 1 expert per token
- Expert Choice: Experts choose tokens (not vice versa)
- Soft MoE: Weighted combinations of experts
- GLaM-style: Generalist Language Model approach

Key benefits:
- 10x+ parameters with constant compute
- Better specialization per domain
- Improved sample efficiency

Used in: GPT-4, Mixtral 8x7B, Switch Transformer, GLaM

References:
- "Switch Transformers: Scaling to Trillion Parameter Models" (Fedus et al., 2021)
- "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts" (Du et al., 2021)
- "Mixture-of-Experts with Expert Choice Routing" (Zhou et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts"""
    # Model
    d_model: int = 512
    num_experts: int = 8
    expert_capacity: int = 64  # Tokens per expert

    # Expert architecture
    d_ff: int = 2048  # FFN hidden dimension
    dropout: float = 0.1

    # Routing
    routing_type: str = "switch"  # switch, expert_choice, soft
    num_experts_per_token: int = 1  # For switch (top-k)

    # Load balancing
    load_balance_loss_weight: float = 0.01
    z_loss_weight: float = 0.001  # Router z-loss for stability

    # Expert choice specific
    expert_choice_k: int = 2  # Tokens per expert for expert_choice


class SwitchFFN(nn.Module):
    """
    Expert FFN for Switch Transformer.

    Standard FFN with GELU activation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) or (num_tokens, d_model)

        Returns:
            output: Same shape as input
        """
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class TopKRouter(nn.Module):
    """
    Top-K routing for Switch Transformer.

    Routes each token to top-k experts based on learned router weights.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_experts_per_token: int = 1,
        jitter_noise: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.jitter_noise = jitter_noise

        # Router weights
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: Input tokens (batch, seq_len, d_model)
            training: Whether in training mode

        Returns:
            router_logits: (batch, seq_len, num_experts)
            expert_indices: (batch, seq_len, num_experts_per_token)
            router_probs: (batch, seq_len, num_experts)
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten for routing
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)

        # Router logits
        router_logits = self.router(x_flat)  # (batch * seq_len, num_experts)

        # Add jitter noise during training for load balancing
        if training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise

        # Softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k routing
        expert_weights, expert_indices = torch.topk(
            router_probs,
            self.num_experts_per_token,
            dim=-1
        )

        # Reshape
        router_logits = router_logits.view(batch_size, seq_len, self.num_experts)
        expert_indices = expert_indices.view(batch_size, seq_len, self.num_experts_per_token)
        router_probs = router_probs.view(batch_size, seq_len, self.num_experts)

        return router_logits, expert_indices, router_probs


class SwitchMoE(nn.Module):
    """
    Switch Transformer Mixture of Experts.

    Sparse MoE where each token is routed to exactly 1 expert.
    Includes load balancing and capacity factor for stability.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # Create experts
        self.experts = nn.ModuleList([
            SwitchFFN(config.d_model, config.d_ff, config.dropout)
            for _ in range(config.num_experts)
        ])

        # Router
        self.router = TopKRouter(
            config.d_model,
            config.num_experts,
            config.num_experts_per_token,
            jitter_noise=0.01 if config.load_balance_loss_weight > 0 else 0.0
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with sparse expert routing.

        Args:
            x: Input (batch, seq_len, d_model)
            return_aux_loss: Return auxiliary losses for training

        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: Optional auxiliary loss (load balance + z-loss)
        """
        batch_size, seq_len, d_model = x.shape

        # Route tokens to experts
        router_logits, expert_indices, router_probs = self.router(x, self.training)

        # Initialize output
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.config.num_experts):
            # Get expert
            expert = self.experts[expert_idx]

            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # (batch, seq_len)

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_input = x[expert_mask]  # (num_tokens_for_expert, d_model)

            # Apply expert
            expert_output = expert(expert_input)  # (num_tokens_for_expert, d_model)

            # Get routing weights for this expert
            expert_weights = router_probs[expert_mask, expert_idx].unsqueeze(-1)

            # Weighted output
            output[expert_mask] += expert_weights * expert_output

        # Compute auxiliary losses
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self._compute_aux_loss(router_logits, router_probs, expert_indices)

        return output, aux_loss

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary losses for training.

        Includes:
        1. Load balancing loss: Encourage uniform expert usage
        2. Router z-loss: Encourage router logits to stay small
        """
        # Load balancing loss
        # Encourage each expert to receive similar number of tokens
        batch_size, seq_len, num_experts = router_probs.shape

        # Fraction of tokens routed to each expert
        expert_usage = router_probs.mean(dim=[0, 1])  # (num_experts,)

        # Ideal uniform distribution
        uniform = torch.ones_like(expert_usage) / num_experts

        # Load balance loss (L2 between actual and uniform)
        load_balance_loss = ((expert_usage - uniform) ** 2).sum()

        # Router z-loss: Keep router logits small for stability
        z_loss = (router_logits ** 2).mean()

        # Combined auxiliary loss
        aux_loss = (
            self.config.load_balance_loss_weight * load_balance_loss +
            self.config.z_loss_weight * z_loss
        )

        return aux_loss


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice routing.

    Instead of tokens choosing experts, experts choose which tokens to process.
    This improves load balancing and allows dynamic capacity.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_choice_k: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_choice_k = expert_choice_k

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Expert choice routing.

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            router_logits: (batch, seq_len, num_experts)
            expert_assignments: List of token indices for each expert
        """
        batch_size, seq_len, d_model = x.shape

        # Router logits
        x_flat = x.view(-1, d_model)
        router_logits = self.router(x_flat)
        router_logits = router_logits.view(batch_size, seq_len, self.num_experts)

        # Each expert chooses top-k tokens
        expert_assignments = []

        for expert_idx in range(self.num_experts):
            # Get scores for this expert across all tokens
            expert_scores = router_logits[:, :, expert_idx]  # (batch, seq_len)

            # Flatten
            expert_scores_flat = expert_scores.view(-1)

            # Top-k tokens for this expert
            _, top_k_indices = torch.topk(
                expert_scores_flat,
                min(self.expert_choice_k, expert_scores_flat.numel())
            )

            expert_assignments.append(top_k_indices)

        return router_logits, expert_assignments


class ExpertChoiceMoE(nn.Module):
    """
    Expert Choice Mixture of Experts.

    Experts choose tokens instead of tokens choosing experts.
    Better load balancing and no dropped tokens.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # Experts
        self.experts = nn.ModuleList([
            SwitchFFN(config.d_model, config.d_ff, config.dropout)
            for _ in range(config.num_experts)
        ])

        # Expert choice router
        self.router = ExpertChoiceRouter(
            config.d_model,
            config.num_experts,
            config.expert_choice_k
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward with expert choice routing.

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: None (expert choice doesn't need aux loss)
        """
        batch_size, seq_len, d_model = x.shape

        # Route
        router_logits, expert_assignments = self.router(x)

        # Flatten input
        x_flat = x.view(-1, d_model)

        # Initialize output
        output = torch.zeros_like(x_flat)
        counts = torch.zeros(x_flat.shape[0], device=x.device)

        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Get assigned token indices
            token_indices = expert_assignments[expert_idx]

            if len(token_indices) == 0:
                continue

            # Get tokens
            expert_input = x_flat[token_indices]

            # Apply expert
            expert_output = expert(expert_input)

            # Get router weights
            router_weights = F.softmax(router_logits.view(-1, self.config.num_experts), dim=-1)
            expert_weights = router_weights[token_indices, expert_idx].unsqueeze(-1)

            # Accumulate output
            output[token_indices] += expert_weights * expert_output
            counts[token_indices] += expert_weights.squeeze(-1)

        # Normalize by counts (some tokens might be chosen by multiple experts)
        output = output / (counts.unsqueeze(-1) + 1e-8)

        # Reshape
        output = output.view(batch_size, seq_len, d_model)

        return output, None


class SoftMoE(nn.Module):
    """
    Soft Mixture of Experts.

    Uses weighted combinations of all experts instead of sparse routing.
    More stable but computationally expensive.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # Experts
        self.experts = nn.ModuleList([
            SwitchFFN(config.d_model, config.d_ff, config.dropout)
            for _ in range(config.num_experts)
        ])

        # Soft router (outputs weights for all experts)
        self.router = nn.Linear(config.d_model, config.num_experts)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward with soft (dense) expert routing.

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: None
        """
        batch_size, seq_len, d_model = x.shape

        # Router weights
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        router_weights = F.softmax(router_logits, dim=-1)

        # Apply all experts
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # (batch, seq_len, d_model)
            expert_outputs.append(expert_output)

        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (batch, seq_len, d_model, num_experts)

        # Weighted combination
        output = torch.einsum('bsde,bse->bsd', expert_outputs, router_weights)

        return output, None


# Factory function
def create_moe(config: MoEConfig) -> nn.Module:
    """
    Create MoE module based on config.

    Args:
        config: MoE configuration

    Returns:
        MoE module
    """
    if config.routing_type == "switch":
        return SwitchMoE(config)
    elif config.routing_type == "expert_choice":
        return ExpertChoiceMoE(config)
    elif config.routing_type == "soft":
        return SoftMoE(config)
    else:
        raise ValueError(f"Unknown routing type: {config.routing_type}")


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Mixture of Experts (MoE) - Sparse Scaling")
    print("="*80)

    batch_size = 4
    seq_len = 32
    d_model = 512

    # Test all MoE variants
    for routing_type in ["switch", "expert_choice", "soft"]:
        print(f"\n{'-'*80}")
        print(f"{routing_type.upper()} MoE")
        print('-'*80)

        config = MoEConfig(
            d_model=d_model,
            num_experts=8,
            d_ff=2048,
            routing_type=routing_type
        )

        moe = create_moe(config)

        # Test input
        x = torch.randn(batch_size, seq_len, d_model)

        # Forward
        output, aux_loss = moe(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        if aux_loss is not None:
            print(f"  Auxiliary loss: {aux_loss.item():.6f}")

        # Count parameters
        total_params = sum(p.numel() for p in moe.parameters())
        expert_params = sum(p.numel() for p in moe.experts[0].parameters())

        print(f"  Total parameters: {total_params:,}")
        print(f"  Parameters per expert: {expert_params:,}")
        print(f"  Effective parameters (sparse): {expert_params + (total_params - config.num_experts * expert_params):,}")

    print("\n" + "="*80)
    print("Scaling Benefits:")
    print(f"  Dense model (2048 FFN): ~1M params")
    print(f"  MoE (8 experts × 2048 FFN): ~8M params")
    print(f"  Compute per token: Same as dense (1 expert active)")
    print(f"  Scaling factor: 8x parameters, 1x compute")
    print("\nUsed in: GPT-4, Mixtral 8x7B, Switch Transformer, GLaM")
    print("="*80)
