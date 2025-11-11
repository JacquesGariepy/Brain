"""
State Space Models (S4, Mamba) - SOTA Sequence Modeling

These models are alternatives to Transformers with linear complexity.
- S4: Structured State Space for Sequence Modeling
- Mamba: Selective State Space Models with input-dependent transitions

Key advantages:
- O(N) complexity vs O(N²) for Transformers
- Better long-range dependencies
- Faster inference
- Competitive or better performance on many tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class S4Config:
    """Configuration for S4 layer"""
    d_model: int = 512
    d_state: int = 64  # State dimension (N)
    dropout: float = 0.1
    bidirectional: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1


class S4Layer(nn.Module):
    """
    Structured State Space Sequence Model (S4).

    Uses diagonal state space representation for efficiency.
    Implements the recurrence:
        x'(t) = Ax(t) + Bu(t)
        y(t) = Cx(t) + Du(t)

    Where A, B, C are learned parameters with special structure.
    """

    def __init__(self, config: S4Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state

        # Initialize state space parameters
        # A: Diagonal matrix (HiPPO initialization)
        A = torch.arange(1, config.d_state + 1).repeat(config.d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Log-space for stability

        # B, C: Input and output matrices
        self.B = nn.Parameter(torch.randn(config.d_model, config.d_state))
        self.C = nn.Parameter(torch.randn(config.d_model, config.d_state))

        # D: Skip connection
        self.D = nn.Parameter(torch.randn(config.d_model))

        # Discretization parameters
        self.dt_proj = nn.Linear(config.d_model, config.d_model)
        self.dt_min = config.dt_min
        self.dt_max = config.dt_max

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of S4 layer.

        Args:
            u: Input sequence (batch, length, d_model)

        Returns:
            Output sequence (batch, length, d_model)
        """
        batch, length, d_model = u.shape

        # Discretize continuous parameters
        A = -torch.exp(self.A_log)  # (d_model, d_state)

        # Compute timestep from input
        dt = F.softplus(self.dt_proj(u))  # (batch, length, d_model)
        dt = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(dt)

        # Discretize using Zero-Order Hold (ZOH)
        dt_expanded = dt.unsqueeze(-1)  # (batch, length, d_model, 1)
        A_discrete = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt_expanded)  # (batch, length, d_model, d_state)
        B_discrete = (A_discrete - 1) / (A.unsqueeze(0).unsqueeze(0) + 1e-8) * self.B.unsqueeze(0).unsqueeze(0)

        # Convolutional mode for training (parallel)
        if self.training:
            return self._convolutional_mode(u, A_discrete, B_discrete)
        else:
            # Recurrent mode for inference (sequential)
            return self._recurrent_mode(u, A_discrete, B_discrete)

    def _convolutional_mode(self, u, A_discrete, B_discrete):
        """Efficient parallel computation using convolution"""
        batch, length, d_model = u.shape

        # Compute kernel
        kernel_size = length
        kernel = torch.zeros(d_model, kernel_size, self.d_state, device=u.device)

        # Build convolution kernel from A and B
        for i in range(kernel_size):
            kernel[:, i] = torch.einsum(
                'ds,ds->ds',
                A_discrete[:, i] ** i,
                B_discrete[:, i]
            )

        # Convolve
        u_transposed = u.transpose(1, 2)  # (batch, d_model, length)

        # Simplified: Linear combination
        y = torch.zeros_like(u)
        for i in range(length):
            for j in range(min(i + 1, kernel_size)):
                y[:, i] += torch.einsum(
                    'bd,ds,ds->bd',
                    u[:, j],
                    kernel[:, i - j],
                    self.C
                )

        # Add skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)

        return self.dropout(y)

    def _recurrent_mode(self, u, A_discrete, B_discrete):
        """Sequential computation for inference"""
        batch, length, d_model = u.shape

        # Initialize state
        x = torch.zeros(batch, d_model, self.d_state, device=u.device)

        outputs = []
        for i in range(length):
            # Update state: x_{t+1} = A*x_t + B*u_t
            x = A_discrete[:, i] * x + B_discrete[:, i] * u[:, i:i+1].unsqueeze(-1)

            # Output: y_t = C*x_t + D*u_t
            y = torch.einsum('bds,ds->bd', x, self.C) + self.D * u[:, i]
            outputs.append(y)

        output = torch.stack(outputs, dim=1)
        return self.dropout(output)


@dataclass
class MambaConfig:
    """Configuration for Mamba model"""
    d_model: int = 512
    d_state: int = 16  # Smaller state for Mamba
    d_conv: int = 4  # Convolution kernel size
    expand: int = 2  # Expansion factor
    dt_rank: str = "auto"  # Rank of dt projection
    dropout: float = 0.1
    conv_bias: bool = True
    bias: bool = False


class MambaBlock(nn.Module):
    """
    Mamba: Selective State Space Model.

    Key innovation: Input-dependent state transitions.
    The parameters A, B, C, and dt are functions of the input.

    This allows the model to selectively propagate or forget information
    based on the input, similar to gating in RNNs/LSTMs but more efficient.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = config.expand * config.d_model

        # Input projection with expansion
        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=config.bias)

        # Depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
            bias=config.conv_bias
        )

        # SSM parameters (input-dependent)
        self.dt_rank = math.ceil(config.d_model / 16) if config.dt_rank == "auto" else config.dt_rank

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize A (structured matrix)
        A = torch.arange(1, config.d_state + 1).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Mamba block.

        Args:
            x: Input tensor (batch, length, d_model)

        Returns:
            Output tensor (batch, length, d_model)
        """
        batch, length, d_model = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # (batch, length, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, length, d_inner)

        # Depthwise convolution
        x = x.transpose(1, 2)  # (batch, d_inner, length)
        x = self.conv1d(x)[:, :, :length]  # Trim padding
        x = x.transpose(1, 2)  # (batch, length, d_inner)

        # Activation
        x = F.silu(x)

        # SSM computation with selective mechanism
        y = self.selective_scan(x)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return self.dropout(output)

    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective state space scan.

        The key innovation: B, C, dt are functions of input.
        """
        batch, length, d_inner = x.shape

        # Project input to get selective parameters
        x_proj = self.x_proj(x)  # (batch, length, dt_rank + 2*d_state)

        dt, B, C = torch.split(
            x_proj,
            [self.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )

        # Compute delta (timestep)
        dt = self.dt_proj(dt)  # (batch, length, d_inner)
        dt = F.softplus(dt)

        # Get A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan
        y = self._selective_scan_forward(x, dt, A, B, C, self.D)

        return y

    def _selective_scan_forward(
        self,
        u: torch.Tensor,  # (batch, length, d_inner)
        dt: torch.Tensor,  # (batch, length, d_inner)
        A: torch.Tensor,  # (d_inner, d_state)
        B: torch.Tensor,  # (batch, length, d_state)
        C: torch.Tensor,  # (batch, length, d_state)
        D: torch.Tensor   # (d_inner,)
    ) -> torch.Tensor:
        """
        Perform selective scan operation.

        This is where the magic happens - input-dependent state transitions.
        """
        batch, length, d_inner = u.shape
        d_state = A.shape[1]

        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)

        outputs = []
        for i in range(length):
            # Discretize A and B using current timestep
            dt_i = dt[:, i:i+1, :].unsqueeze(-1)  # (batch, 1, d_inner, 1)
            A_discrete = torch.exp(A.unsqueeze(0) * dt_i)  # (batch, 1, d_inner, d_state)
            A_discrete = A_discrete.squeeze(1)  # (batch, d_inner, d_state)

            B_i = B[:, i:i+1, :].unsqueeze(1)  # (batch, 1, 1, d_state)
            B_discrete = B_i * dt_i  # (batch, 1, d_inner, d_state)
            B_discrete = B_discrete.squeeze(1)  # (batch, d_inner, d_state)

            # Update state
            h = A_discrete * h + B_discrete * u[:, i:i+1, :].unsqueeze(-1)

            # Compute output
            C_i = C[:, i, :].unsqueeze(1)  # (batch, 1, d_state)
            y_i = torch.einsum('bds,bs->bd', h, C_i.squeeze(1))
            y_i = y_i + D * u[:, i]

            outputs.append(y_i)

        return torch.stack(outputs, dim=1)


class MambaModel(nn.Module):
    """
    Complete Mamba model with multiple blocks.

    This is a competitive alternative to Transformers with:
    - Linear complexity
    - Selective information propagation
    - Strong performance on long sequences
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )

        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = x + layer(x)  # Residual connection

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits
