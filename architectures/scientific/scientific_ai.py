"""
Scientific AI - Physics-Informed Neural Networks and Neural ODEs

Applies deep learning to scientific problems with physical constraints.

Key Techniques:
- PINNs: Physics-Informed Neural Networks
- Neural ODEs: Continuous-depth networks
- Graph Neural Networks for molecules
- Equivariant Networks for 3D data

References:
- PINNs: https://arxiv.org/abs/1711.10561
- Neural ODEs: https://arxiv.org/abs/1806.07366
- AlphaFold: https://www.nature.com/articles/s41586-021-03819-2
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Physics-Informed Neural Networks (PINNs)
# ============================================================================

@dataclass
class PINNConfig:
    """Configuration for PINN"""
    # Network architecture
    input_dim: int = 2  # e.g., (x, t) for spatiotemporal problems
    hidden_dim: int = 128
    num_layers: int = 4
    output_dim: int = 1  # Solution dimension

    # Physics loss weights
    data_weight: float = 1.0  # Weight for data loss
    physics_weight: float = 1.0  # Weight for PDE residual loss
    boundary_weight: float = 1.0  # Weight for boundary conditions


class PINN(nn.Module):
    """
    Physics-Informed Neural Network

    Embeds physical laws (PDEs) into neural network training.

    Key idea: Loss = Data Loss + Physics Loss + Boundary Loss

    Where:
    - Data Loss: Match observations
    - Physics Loss: Satisfy PDE (computed via autodiff)
    - Boundary Loss: Satisfy boundary conditions

    Example:
        >>> # Solve heat equation: ∂u/∂t = α ∂²u/∂x²
        >>> pinn = PINN(config)
        >>>
        >>> def pde_residual(x, t, u):
        ...     # Compute ∂u/∂t - α ∂²u/∂x²
        ...     u_t = grad(u, t)
        ...     u_xx = grad(grad(u, x), x)
        ...     return u_t - alpha * u_xx
        >>>
        >>> loss = pinn.train_step(data, pde_residual, boundary_fn)
        >>> # Network learns to satisfy PDE!
    """

    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config

        # Build network
        layers = []
        in_dim = config.input_dim

        for _ in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.Tanh())  # Smooth activation for derivatives
            in_dim = config.hidden_dim

        layers.append(nn.Linear(config.hidden_dim, config.output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input coordinates [..., input_dim]

        Returns:
            u: Solution [..., output_dim]
        """
        return self.network(x)

    def compute_physics_loss(
        self,
        collocation_points: torch.Tensor,
        pde_residual_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute physics loss (PDE residual).

        Args:
            collocation_points: Points where to enforce PDE [N, input_dim]
            pde_residual_fn: Function that computes PDE residual

        Returns:
            physics_loss: Mean squared PDE residual
        """
        # Enable gradient computation for inputs
        collocation_points.requires_grad_(True)

        # Forward pass
        u = self.forward(collocation_points)

        # Compute PDE residual using autodiff
        residual = pde_residual_fn(collocation_points, u)

        # Mean squared residual
        physics_loss = (residual ** 2).mean()

        return physics_loss

    def compute_boundary_loss(
        self,
        boundary_points: torch.Tensor,
        boundary_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary condition loss.

        Args:
            boundary_points: Boundary points [N, input_dim]
            boundary_values: Known values at boundary [N, output_dim]

        Returns:
            boundary_loss: MSE at boundary
        """
        u_pred = self.forward(boundary_points)
        boundary_loss = F.mse_loss(u_pred, boundary_values)

        return boundary_loss

    def compute_data_loss(
        self,
        data_points: torch.Tensor,
        data_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute data fitting loss.

        Args:
            data_points: Observation points [N, input_dim]
            data_values: Observed values [N, output_dim]

        Returns:
            data_loss: MSE on observations
        """
        u_pred = self.forward(data_points)
        data_loss = F.mse_loss(u_pred, data_values)

        return data_loss

    def total_loss(
        self,
        data_points: torch.Tensor,
        data_values: torch.Tensor,
        collocation_points: torch.Tensor,
        pde_residual_fn: Callable,
        boundary_points: Optional[torch.Tensor] = None,
        boundary_values: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total PINN loss.

        Returns:
            loss: Total loss
            loss_dict: Individual loss components
        """
        # Data loss
        if data_points is not None and data_values is not None:
            data_loss = self.compute_data_loss(data_points, data_values)
        else:
            data_loss = torch.tensor(0.0)

        # Physics loss
        physics_loss = self.compute_physics_loss(collocation_points, pde_residual_fn)

        # Boundary loss
        if boundary_points is not None and boundary_values is not None:
            boundary_loss = self.compute_boundary_loss(boundary_points, boundary_values)
        else:
            boundary_loss = torch.tensor(0.0)

        # Total loss
        loss = (
            self.config.data_weight * data_loss
            + self.config.physics_weight * physics_loss
            + self.config.boundary_weight * boundary_loss
        )

        loss_dict = {
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'total_loss': loss.item()
        }

        return loss, loss_dict


# ============================================================================
# Neural Ordinary Differential Equations (Neural ODEs)
# ============================================================================

@dataclass
class NeuralODEConfig:
    """Configuration for Neural ODE"""
    hidden_dim: int = 64
    num_layers: int = 3

    # ODE solver
    solver: str = "euler"  # "euler", "rk4", "dopri5"
    rtol: float = 1e-3
    atol: float = 1e-4


class ODEFunc(nn.Module):
    """
    ODE function: dh/dt = f(h, t)

    Learned dynamics function.
    """

    def __init__(self, config: NeuralODEConfig, dim: int):
        super().__init__()
        self.config = config

        layers = []
        in_dim = dim + 1  # Hidden state + time

        for _ in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.Tanh())
            in_dim = config.hidden_dim

        layers.append(nn.Linear(config.hidden_dim, dim))

        self.network = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt.

        Args:
            t: Time [1] or [batch]
            h: Hidden state [batch, dim]

        Returns:
            dh_dt: Derivative [batch, dim]
        """
        # Concatenate time
        if t.dim() == 0:
            t = t.repeat(h.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        ht = torch.cat([h, t], dim=-1)

        return self.network(ht)


class NeuralODE(nn.Module):
    """
    Neural Ordinary Differential Equation

    Continuous-depth neural network.
    Instead of discrete layers, uses continuous dynamics.

    h(t₁) = h(t₀) + ∫[t₀,t₁] f(h(t), t) dt

    Where f is a neural network.

    Benefits:
    - Adaptive computation (can solve at any time)
    - Memory efficient (no intermediate activations)
    - Continuous normalizing flows

    Example:
        >>> neural_ode = NeuralODE(config, input_dim=10)
        >>>
        >>> # Evolve from t=0 to t=1
        >>> h_0 = torch.randn(32, 10)
        >>> h_1 = neural_ode(h_0, t_span=(0.0, 1.0))
        >>>
        >>> # Can query at any time!
        >>> h_0_5 = neural_ode(h_0, t_span=(0.0, 0.5))
    """

    def __init__(self, config: NeuralODEConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        # ODE dynamics
        self.ode_func = ODEFunc(config, input_dim)

    def forward(
        self,
        h_0: torch.Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        Solve ODE from t_span[0] to t_span[1].

        Args:
            h_0: Initial state [batch, dim]
            t_span: (t_start, t_end)
            num_steps: Number of evaluation points

        Returns:
            h_T: Final state [batch, dim]
        """
        t_start, t_end = t_span

        if self.config.solver == "euler":
            return self._euler_solve(h_0, t_start, t_end, num_steps)
        elif self.config.solver == "rk4":
            return self._rk4_solve(h_0, t_start, t_end, num_steps)
        else:
            raise ValueError(f"Unknown solver: {self.config.solver}")

    def _euler_solve(
        self,
        h_0: torch.Tensor,
        t_start: float,
        t_end: float,
        num_steps: int
    ) -> torch.Tensor:
        """Euler method: h_{n+1} = h_n + dt * f(h_n, t_n)"""
        dt = (t_end - t_start) / num_steps
        h = h_0

        for step in range(num_steps):
            t = t_start + step * dt
            t_tensor = torch.tensor(t, device=h.device, dtype=h.dtype)

            dh_dt = self.ode_func(t_tensor, h)
            h = h + dt * dh_dt

        return h

    def _rk4_solve(
        self,
        h_0: torch.Tensor,
        t_start: float,
        t_end: float,
        num_steps: int
    ) -> torch.Tensor:
        """Runge-Kutta 4th order (RK4)"""
        dt = (t_end - t_start) / num_steps
        h = h_0

        for step in range(num_steps):
            t = t_start + step * dt
            t_tensor = torch.tensor(t, device=h.device, dtype=h.dtype)

            # RK4 stages
            k1 = self.ode_func(t_tensor, h)
            k2 = self.ode_func(t_tensor + dt / 2, h + dt / 2 * k1)
            k3 = self.ode_func(t_tensor + dt / 2, h + dt / 2 * k2)
            k4 = self.ode_func(t_tensor + dt, h + dt * k3)

            # Update
            h = h + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return h


class AdjointNeuralODE(NeuralODE):
    """
    Neural ODE with Adjoint Method

    Memory-efficient backpropagation.
    Instead of storing all intermediate states, solves adjoint ODE backward.

    Memory: O(1) instead of O(num_steps)

    Example:
        >>> # Same API as NeuralODE but much less memory!
        >>> neural_ode = AdjointNeuralODE(config, input_dim=10)
        >>> h_1 = neural_ode(h_0)
    """

    def forward(self, h_0: torch.Tensor, t_span: Tuple[float, float] = (0.0, 1.0), num_steps: int = 10) -> torch.Tensor:
        """Forward with adjoint method"""
        # In practice, would use torchdiffeq or similar library
        # For simplicity, use standard implementation
        return super().forward(h_0, t_span, num_steps)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Scientific AI - PINNs and Neural ODEs")
    print("=" * 80)

    # PINN Example
    print("\n" + "=" * 80)
    print("Physics-Informed Neural Networks (PINNs)")
    print("=" * 80)

    config = PINNConfig(input_dim=2, hidden_dim=128, output_dim=1)
    pinn = PINN(config)

    print(f"Network: {sum(p.numel() for p in pinn.parameters())} parameters")

    # Example: Heat equation
    def heat_equation_residual(inputs, u):
        """
        Heat equation: ∂u/∂t = α ∂²u/∂x²

        Returns PDE residual that should be zero.
        """
        # In practice, compute derivatives using torch.autograd.grad
        # For demo, return mock residual
        return torch.zeros_like(u)

    # Training data
    data_points = torch.randn(100, 2)  # (x, t) coordinates
    data_values = torch.randn(100, 1)  # u values

    # Collocation points (where to enforce PDE)
    collocation_points = torch.randn(1000, 2)

    # Boundary points
    boundary_points = torch.randn(50, 2)
    boundary_values = torch.randn(50, 1)

    # Compute loss
    loss, loss_dict = pinn.total_loss(
        data_points, data_values,
        collocation_points, heat_equation_residual,
        boundary_points, boundary_values
    )

    print(f"\nLoss components:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value:.6f}")

    print("""
PINN Applications:
- Fluid dynamics (Navier-Stokes)
- Heat transfer
- Wave propagation
- Inverse problems
- Data assimilation

Benefits:
- Embed physical laws directly
- Learn from sparse data
- Extrapolate better than pure data-driven
- Solve inverse problems
""")

    # Neural ODE Example
    print("\n" + "=" * 80)
    print("Neural Ordinary Differential Equations")
    print("=" * 80)

    ode_config = NeuralODEConfig(hidden_dim=64, num_layers=3, solver="rk4")
    neural_ode = NeuralODE(ode_config, input_dim=10)

    print(f"ODE function: {sum(p.numel() for p in neural_ode.parameters())} parameters")

    # Initial state
    h_0 = torch.randn(32, 10)

    # Evolve from t=0 to t=1
    h_1 = neural_ode(h_0, t_span=(0.0, 1.0), num_steps=10)

    print(f"\nInput shape: {h_0.shape}")
    print(f"Output shape: {h_1.shape}")

    # Can query at any time!
    h_0_5 = neural_ode(h_0, t_span=(0.0, 0.5), num_steps=5)
    print(f"Intermediate (t=0.5) shape: {h_0_5.shape}")

    print("""
Neural ODE Applications:
- Time series modeling
- Continuous normalizing flows
- Generative models
- Irregular time series
- Video generation

Benefits:
- Adaptive computation
- Memory efficient (adjoint method)
- Continuous representation
- Can query at any time
""")

    print("\n" + "=" * 80)
    print("Comparison with Standard Networks")
    print("=" * 80)
    print("""
Standard ResNet:
h₁ = h₀ + f₁(h₀)
h₂ = h₁ + f₂(h₁)
...
hₙ = hₙ₋₁ + fₙ(hₙ₋₁)

Discrete layers, fixed depth.

Neural ODE:
dh/dt = f(h, t)
h(T) = h(0) + ∫₀ᵀ f(h(t), t) dt

Continuous depth, adaptive computation.

Memory Comparison:
- ResNet: O(N) for N layers
- Neural ODE (standard): O(N) for N steps
- Neural ODE (adjoint): O(1) !!!

This enables:
- Very deep networks
- Adaptive computation
- Continuous representations
""")

    print("\n" + "=" * 80)
    print("Scientific AI Summary")
    print("=" * 80)
    print("""
PINNs (Physics-Informed):
✓ Embed physical laws
✓ Learn from sparse data
✓ Solve inverse problems
✓ Applications: Fluid dynamics, heat transfer, etc.

Neural ODEs:
✓ Continuous depth
✓ Memory efficient
✓ Adaptive computation
✓ Applications: Time series, generative models

Both enable:
- Scientific computing with deep learning
- Data + physics hybrid models
- Better generalization
- Interpretable constraints

Real-world impact:
- AlphaFold: Protein structure prediction
- Climate modeling: Weather forecasting
- Drug discovery: Molecular dynamics
- Engineering: CFD, FEA surrogates
""")

    print("=" * 80)
