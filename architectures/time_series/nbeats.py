"""
N-BEATS - Neural Basis Expansion Analysis for Interpretable Time Series Forecasting

SOTA time series forecasting with interpretability (2020-2025).

Key features:
- Deep learning for univariate time series
- Interpretable architecture (trend + seasonality)
- No feature engineering required
- Doubly residual stacking
- Backcast/forecast split
- SOTA on M4 competition

Architecture:
- Stack of blocks (trend + seasonality)
- Each block: FC layers + basis expansion
- Residual connections at two levels

References:
- "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting" (Oreshkin et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math


@dataclass
class NBEATSConfig:
    """Configuration for N-BEATS model"""
    # Data
    backcast_length: int = 10  # Input sequence length
    forecast_length: int = 5   # Output sequence length

    # Architecture
    stack_types: List[str] = None  # ['trend', 'seasonality', 'generic']
    num_blocks_per_stack: int = 3
    hidden_layer_units: int = 256
    num_layers: int = 4

    # Basis functions
    thetas_dim: List[int] = None  # Dimension of theta parameters per stack
    share_weights_in_stack: bool = False

    def __post_init__(self):
        if self.stack_types is None:
            self.stack_types = ['trend', 'seasonality', 'generic']
        if self.thetas_dim is None:
            # Default: polynomial degree for trend, harmonics for seasonality
            self.thetas_dim = [4, 8, 8]  # One per stack


class NBeatsBlock(nn.Module):
    """
    Single N-BEATS block.

    Produces backcast (past) and forecast (future) via basis expansion.
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        basis_function: nn.Module,
        num_layers: int = 4,
        hidden_size: int = 256
    ):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function

        # FC layers
        layers = []
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.ReLU()
            ])
        self.fc_layers = nn.Sequential(*layers)

        # Theta layers (backcast and forecast)
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input sequence (batch, input_size)

        Returns:
            backcast: Backcast (batch, input_size)
            forecast: Forecast (batch, forecast_size)
        """
        # FC layers
        h = self.fc_layers(x)

        # Compute theta parameters
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)

        # Basis expansion
        backcast = self.basis_function(theta_b, backcast=True)
        forecast = self.basis_function(theta_f, backcast=False)

        return backcast, forecast


class GenericBasis(nn.Module):
    """Generic basis function (fully learnable)"""

    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        theta_size: int
    ):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.theta_size = theta_size

        # Learnable basis matrices
        self.backcast_basis = nn.Parameter(
            torch.randn(theta_size, backcast_size) * 0.01
        )
        self.forecast_basis = nn.Parameter(
            torch.randn(theta_size, forecast_size) * 0.01
        )

    def forward(self, theta: torch.Tensor, backcast: bool) -> torch.Tensor:
        """
        Expand theta using basis functions.

        Args:
            theta: Coefficients (batch, theta_size)
            backcast: Whether this is for backcast (vs forecast)

        Returns:
            Expanded signal (batch, backcast_size or forecast_size)
        """
        if backcast:
            return theta @ self.backcast_basis
        else:
            return theta @ self.forecast_basis


class TrendBasis(nn.Module):
    """Trend basis function (polynomial)"""

    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        polynomial_degree: int = 3
    ):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.polynomial_degree = polynomial_degree

        # Create polynomial basis
        backcast_time = torch.arange(backcast_size).float() / backcast_size
        forecast_time = torch.arange(forecast_size).float() / forecast_size

        self.register_buffer(
            'backcast_basis',
            self._polynomial_basis(backcast_time, polynomial_degree)
        )
        self.register_buffer(
            'forecast_basis',
            self._polynomial_basis(forecast_time, polynomial_degree)
        )

    def _polynomial_basis(
        self,
        time: torch.Tensor,
        degree: int
    ) -> torch.Tensor:
        """Create polynomial basis vectors"""
        # Powers: t^0, t^1, t^2, ..., t^degree
        basis = torch.stack([time ** i for i in range(degree + 1)], dim=1)
        return basis.T  # (degree+1, time_steps)

    def forward(self, theta: torch.Tensor, backcast: bool) -> torch.Tensor:
        """
        Args:
            theta: Polynomial coefficients (batch, degree+1)
            backcast: Whether for backcast

        Returns:
            Polynomial curve (batch, time_steps)
        """
        if backcast:
            return theta @ self.backcast_basis
        else:
            return theta @ self.forecast_basis


class SeasonalityBasis(nn.Module):
    """Seasonality basis function (Fourier series)"""

    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        num_harmonics: int = 1
    ):
        super().__init__()
        self.num_harmonics = num_harmonics

        # Create Fourier basis
        backcast_basis = self._fourier_basis(backcast_size, num_harmonics)
        forecast_basis = self._fourier_basis(forecast_size, num_harmonics)

        self.register_buffer('backcast_basis', backcast_basis)
        self.register_buffer('forecast_basis', forecast_basis)

    def _fourier_basis(
        self,
        size: int,
        num_harmonics: int
    ) -> torch.Tensor:
        """Create Fourier basis (sin and cos)"""
        time = torch.arange(size).float()
        basis = []

        for k in range(1, num_harmonics + 1):
            basis.append(torch.sin(2 * math.pi * k * time / size))
            basis.append(torch.cos(2 * math.pi * k * time / size))

        return torch.stack(basis)  # (2*num_harmonics, size)

    def forward(self, theta: torch.Tensor, backcast: bool) -> torch.Tensor:
        """
        Args:
            theta: Fourier coefficients (batch, 2*num_harmonics)
            backcast: Whether for backcast

        Returns:
            Seasonal pattern (batch, time_steps)
        """
        if backcast:
            return theta @ self.backcast_basis
        else:
            return theta @ self.forecast_basis


class NBEATS(nn.Module):
    """
    Complete N-BEATS model.

    Interpretable time series forecasting with:
    - Trend modeling (polynomial basis)
    - Seasonality modeling (Fourier basis)
    - Generic modeling (learned basis)
    """

    def __init__(self, config: NBEATSConfig):
        super().__init__()
        self.config = config

        # Create stacks
        self.stacks = nn.ModuleList()

        for stack_id, stack_type in enumerate(config.stack_types):
            # Create blocks for this stack
            blocks = []

            for _ in range(config.num_blocks_per_stack):
                # Create basis function
                theta_dim = config.thetas_dim[stack_id]

                if stack_type == 'trend':
                    basis = TrendBasis(
                        config.backcast_length,
                        config.forecast_length,
                        polynomial_degree=theta_dim - 1
                    )
                elif stack_type == 'seasonality':
                    basis = SeasonalityBasis(
                        config.backcast_length,
                        config.forecast_length,
                        num_harmonics=theta_dim // 2
                    )
                else:  # generic
                    basis = GenericBasis(
                        config.backcast_length,
                        config.forecast_length,
                        theta_dim
                    )

                # Create block
                block = NBeatsBlock(
                    input_size=config.backcast_length,
                    theta_size=theta_dim,
                    basis_function=basis,
                    num_layers=config.num_layers,
                    hidden_size=config.hidden_layer_units
                )

                blocks.append(block)

            # Stack blocks
            self.stacks.append(nn.ModuleList(blocks))

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with doubly residual stacking.

        Args:
            x: Input sequence (batch, backcast_length)
            return_components: Whether to return trend/seasonality components

        Returns:
            forecast: Predictions (batch, forecast_length)
            components: Dict with trend/seasonality (if return_components=True)
        """
        batch_size = x.shape[0]

        # Initialize
        forecast = torch.zeros(
            batch_size, self.config.forecast_length,
            device=x.device
        )

        # Store components for interpretability
        components = {
            'trend': torch.zeros_like(forecast),
            'seasonality': torch.zeros_like(forecast),
            'generic': torch.zeros_like(forecast)
        }

        # Residual at input level
        residual = x

        # Process each stack
        for stack_id, (stack, stack_type) in enumerate(
            zip(self.stacks, self.config.stack_types)
        ):
            # Process each block in stack
            for block in stack:
                # Forward through block
                backcast, block_forecast = block(residual)

                # Accumulate forecast
                forecast = forecast + block_forecast
                components[stack_type] = components[stack_type] + block_forecast

                # Residual connection (backcast)
                residual = residual - backcast

        if return_components:
            return forecast, components
        else:
            return forecast

    def predict(
        self,
        x: torch.Tensor,
        steps_ahead: int = 1
    ) -> torch.Tensor:
        """
        Multi-step ahead forecasting.

        Args:
            x: Input sequence (batch, backcast_length)
            steps_ahead: Number of steps to forecast

        Returns:
            Predictions (batch, steps_ahead * forecast_length)
        """
        predictions = []

        for _ in range(steps_ahead):
            # Forecast next window
            forecast = self.forward(x)
            predictions.append(forecast)

            # Update input (rolling window)
            x = torch.cat([
                x[:, forecast.shape[1]:],
                forecast
            ], dim=1)

        return torch.cat(predictions, dim=1)


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("N-BEATS - Neural Basis Expansion for Time Series")
    print("="*80)

    # Create N-BEATS model
    config = NBEATSConfig(
        backcast_length=10,
        forecast_length=5,
        stack_types=['trend', 'seasonality'],
        num_blocks_per_stack=3,
        hidden_layer_units=256
    )

    model = NBEATS(config)

    # Example: Forecast
    batch_size = 4
    x = torch.randn(batch_size, config.backcast_length)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    forecast, components = model(x, return_components=True)

    print(f"\nForecast shape: {forecast.shape}")
    print(f"Trend component shape: {components['trend'].shape}")
    print(f"Seasonality component shape: {components['seasonality'].shape}")

    # Multi-step forecast
    multi_forecast = model.predict(x, steps_ahead=3)
    print(f"\nMulti-step forecast shape: {multi_forecast.shape}")

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "="*80)
