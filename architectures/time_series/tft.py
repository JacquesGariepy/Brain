"""
TFT - Temporal Fusion Transformer

SOTA multi-horizon time series forecasting (Google Research, 2021).

Key features:
- Multi-horizon forecasting with quantile outputs
- Variable selection networks (attention for features)
- Static covariate encoders
- Temporal self-attention (from Transformer LLMs)
- Gated Residual Networks (GRN)
- Interpretable attention weights
- Handles both static and time-varying features

Architecture:
- Variable selection for static/temporal features
- LSTM encoder for temporal processing
- Multi-head attention for temporal relationships
- Gated skip connections throughout
- Quantile regression for uncertainty

Applications:
- Financial forecasting
- Energy demand prediction
- Retail sales forecasting
- Any multi-variate time series

References:
- "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (Lim et al., 2021)
- Google Research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class TFTConfig:
    """Configuration for TFT"""
    # Input dimensions
    static_input_size: int = 0  # Number of static features
    temporal_known_size: int = 0  # Known future inputs
    temporal_observed_size: int = 0  # Observed historical inputs
    target_size: int = 1  # Target variable dimension

    # Sequence lengths
    encoder_length: int = 24  # Historical window
    decoder_length: int = 12  # Forecast horizon

    # Architecture
    hidden_size: int = 160
    num_heads: int = 4
    num_lstm_layers: int = 1
    dropout: float = 0.1

    # Quantile forecasting
    quantiles: List[float] = None  # [0.1, 0.5, 0.9]

    # Variable selection
    use_variable_selection: bool = True

    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN).

    Core building block with:
    - ELU activation
    - Layer normalization
    - Gating mechanism for controlling information flow
    - Skip connection
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        context_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.context_size = context_size

        # Primary layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)

        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Gate
        self.gate_fc = nn.Linear(hidden_size, self.output_size)

        # Skip connection (if sizes don't match)
        if input_size != self.output_size:
            self.skip_fc = nn.Linear(input_size, self.output_size)
        else:
            self.skip_fc = None

        self.layer_norm = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch, ..., input_size)
            context: Optional context (batch, ..., context_size)

        Returns:
            Output (batch, ..., output_size)
        """
        # Skip connection
        if self.skip_fc is not None:
            skip = self.skip_fc(x)
        else:
            skip = x

        # Primary path
        hidden = F.elu(self.fc1(x))

        # Add context if provided
        if context is not None and self.context_size is not None:
            hidden = hidden + self.context_fc(context)

        hidden = F.elu(self.fc2(hidden))
        hidden = self.dropout(hidden)

        # Gating mechanism (GLU-like)
        gate = torch.sigmoid(self.gate_fc(hidden))
        gated = gate * skip + (1 - gate) * hidden

        # Layer norm
        output = self.layer_norm(gated)

        return output


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network.

    Selects relevant features using attention-like mechanism.
    Provides interpretable feature importance.
    """

    def __init__(
        self,
        input_size: int,
        num_variables: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None
    ):
        super().__init__()
        self.input_size = input_size
        self.num_variables = num_variables
        self.hidden_size = hidden_size

        # GRN for each variable
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
                context_size=context_size
            )
            for _ in range(num_variables)
        ])

        # Attention for variable selection
        self.flattened_grn = GatedResidualNetwork(
            input_size=input_size * num_variables,
            hidden_size=hidden_size,
            output_size=num_variables,
            dropout=dropout,
            context_size=context_size
        )

    def forward(
        self,
        variables: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            variables: Input variables (batch, ..., num_variables, input_size)
            context: Optional context (batch, ..., context_size)

        Returns:
            weighted_vars: Weighted variables (batch, ..., hidden_size)
            weights: Variable importance weights (batch, ..., num_variables)
        """
        batch_shape = variables.shape[:-2]
        num_vars = variables.shape[-2]

        # Flatten for attention computation
        flattened = variables.flatten(start_dim=-2)  # (batch, ..., num_vars * input_size)

        # Compute variable weights
        weights = self.flattened_grn(flattened, context)
        weights = F.softmax(weights, dim=-1)  # (batch, ..., num_variables)

        # Process each variable through its GRN
        processed = []
        for i in range(num_vars):
            var = variables[..., i, :]  # (batch, ..., input_size)
            processed_var = self.variable_grns[i](var, context)
            processed.append(processed_var)

        processed = torch.stack(processed, dim=-2)  # (batch, ..., num_vars, hidden_size)

        # Weight variables
        weights_expanded = weights.unsqueeze(-1)  # (batch, ..., num_vars, 1)
        weighted = (processed * weights_expanded).sum(dim=-2)  # (batch, ..., hidden_size)

        return weighted, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretability.

    Standard transformer attention with additive aggregation
    for better interpretation of attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_q, embed_dim)
            key: (batch, seq_k, embed_dim)
            value: (batch, seq_v, embed_dim)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_q, embed_dim)
            attention_weights: (batch, num_heads, seq_q, seq_k)
        """
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]

        # Project and reshape
        q = self.query(query).view(batch_size, seq_q, self.num_heads, self.head_dim)
        k = self.key(key).view(batch_size, seq_k, self.num_heads, self.head_dim)
        v = self.value(value).view(batch_size, seq_k, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, num_heads, seq_q, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Transpose and reshape
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_q, self.embed_dim)

        # Output projection
        output = self.out_proj(context)

        # Average attention weights across heads for interpretability
        avg_attn = attn_weights.mean(dim=1)

        return output, avg_attn


class TemporalFusionTransformer(nn.Module):
    """
    Complete Temporal Fusion Transformer.

    Multi-horizon forecasting with:
    - Variable selection
    - Temporal processing with LSTM
    - Multi-head attention
    - Quantile predictions
    - Interpretability
    """

    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config

        # Static variable selection
        if config.static_input_size > 0 and config.use_variable_selection:
            self.static_selection = VariableSelectionNetwork(
                input_size=config.static_input_size,
                num_variables=1,  # Simplified
                hidden_size=config.hidden_size,
                dropout=config.dropout
            )
            self.static_encoder = GatedResidualNetwork(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                output_size=config.hidden_size,
                dropout=config.dropout
            )

        # Historical variable selection
        historical_size = config.temporal_observed_size + config.target_size
        if config.use_variable_selection:
            self.historical_selection = VariableSelectionNetwork(
                input_size=1,  # Per variable
                num_variables=historical_size,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
                context_size=config.hidden_size if config.static_input_size > 0 else None
            )

        # Future variable selection
        if config.temporal_known_size > 0 and config.use_variable_selection:
            self.future_selection = VariableSelectionNetwork(
                input_size=1,
                num_variables=config.temporal_known_size,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
                context_size=config.hidden_size if config.static_input_size > 0 else None
            )

        # LSTM encoder/decoder
        self.encoder_lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0,
            batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0,
            batch_first=True
        )

        # Post-LSTM gating
        self.post_lstm_gate = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )

        # Self-attention
        self.self_attention = InterpretableMultiHeadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )

        self.post_attention_gate = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )

        # Output layer (quantile predictions)
        self.output_layer = nn.Linear(
            config.hidden_size,
            len(config.quantiles) * config.target_size
        )

    def forward(
        self,
        static_inputs: Optional[torch.Tensor] = None,
        historical_inputs: Optional[torch.Tensor] = None,
        future_inputs: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            static_inputs: Static features (batch, static_size)
            historical_inputs: Historical features (batch, encoder_len, temporal_size)
            future_inputs: Known future features (batch, decoder_len, known_size)
            return_attention: Return attention weights for interpretability

        Returns:
            Dictionary with:
            - predictions: Quantile forecasts (batch, decoder_len, num_quantiles * target_size)
            - attention_weights: Optional attention weights
            - variable_importance: Optional variable selection weights
        """
        batch_size = historical_inputs.shape[0] if historical_inputs is not None else 1

        # Static encoding
        static_context = None
        if static_inputs is not None and hasattr(self, 'static_selection'):
            # Add variable dimension
            static_inputs_expanded = static_inputs.unsqueeze(-2)
            static_selected, static_weights = self.static_selection(static_inputs_expanded)
            static_context = self.static_encoder(static_selected)

        # Historical variable selection
        if historical_inputs is not None:
            # Reshape for variable selection: (batch, seq, num_vars, 1)
            hist_reshaped = historical_inputs.unsqueeze(-1)
            hist_selected, hist_weights = self.historical_selection(
                hist_reshaped,
                context=static_context
            )
        else:
            hist_selected = torch.zeros(
                batch_size,
                self.config.encoder_length,
                self.config.hidden_size
            )

        # Future variable selection
        if future_inputs is not None and hasattr(self, 'future_selection'):
            future_reshaped = future_inputs.unsqueeze(-1)
            future_selected, future_weights = self.future_selection(
                future_reshaped,
                context=static_context
            )
        else:
            future_selected = torch.zeros(
                batch_size,
                self.config.decoder_length,
                self.config.hidden_size
            )

        # LSTM encoder
        encoder_output, (hidden, cell) = self.encoder_lstm(hist_selected)

        # LSTM decoder
        decoder_output, _ = self.decoder_lstm(future_selected, (hidden, cell))

        # Concatenate encoder and decoder outputs
        temporal_output = torch.cat([encoder_output, decoder_output], dim=1)

        # Post-LSTM gating
        gated_output = self.post_lstm_gate(temporal_output)

        # Self-attention
        attn_output, attention_weights = self.self_attention(
            gated_output, gated_output, gated_output
        )

        # Post-attention gating with skip connection
        attn_output = self.post_attention_gate(attn_output) + gated_output

        # Extract decoder portion for prediction
        decoder_portion = attn_output[:, -self.config.decoder_length:, :]

        # Output layer
        predictions = self.output_layer(decoder_portion)

        # Reshape predictions: (batch, decoder_len, num_quantiles, target_size)
        predictions = predictions.view(
            batch_size,
            self.config.decoder_length,
            len(self.config.quantiles),
            self.config.target_size
        )

        outputs = {'predictions': predictions}

        if return_attention:
            outputs['attention_weights'] = attention_weights
            if hasattr(self, 'historical_selection'):
                outputs['historical_importance'] = hist_weights
            if hasattr(self, 'future_selection'):
                outputs['future_importance'] = future_weights

        return outputs

    def predict(
        self,
        static_inputs: Optional[torch.Tensor] = None,
        historical_inputs: Optional[torch.Tensor] = None,
        future_inputs: Optional[torch.Tensor] = None,
        quantile: float = 0.5
    ) -> torch.Tensor:
        """
        Make predictions at a specific quantile.

        Args:
            static_inputs: Static features
            historical_inputs: Historical features
            future_inputs: Known future features
            quantile: Which quantile to return (default: 0.5 = median)

        Returns:
            Predictions (batch, decoder_len, target_size)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                static_inputs=static_inputs,
                historical_inputs=historical_inputs,
                future_inputs=future_inputs
            )

            predictions = outputs['predictions']

            # Find closest quantile
            quantile_idx = min(
                range(len(self.config.quantiles)),
                key=lambda i: abs(self.config.quantiles[i] - quantile)
            )

            # Extract predictions for this quantile
            return predictions[:, :, quantile_idx, :]


class QuantileLoss(nn.Module):
    """
    Quantile loss for TFT.

    Computes pinball loss for quantile regression.
    """

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (batch, seq, num_quantiles, target_size)
            targets: (batch, seq, target_size)

        Returns:
            Loss scalar
        """
        targets = targets.unsqueeze(2)  # (batch, seq, 1, target_size)

        losses = []
        for i, q in enumerate(self.quantiles):
            pred = predictions[:, :, i:i+1, :]  # (batch, seq, 1, target_size)
            errors = targets - pred

            loss = torch.max(
                q * errors,
                (q - 1) * errors
            )
            losses.append(loss)

        total_loss = torch.stack(losses, dim=0).mean()
        return total_loss


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("TFT - Temporal Fusion Transformer")
    print("="*80)

    # Create TFT model
    config = TFTConfig(
        static_input_size=4,
        temporal_observed_size=3,
        temporal_known_size=2,
        target_size=1,
        encoder_length=24,
        decoder_length=12,
        hidden_size=160,
        num_heads=4
    )

    model = TemporalFusionTransformer(config)

    # Test forward pass
    batch_size = 8

    static_inputs = torch.randn(batch_size, config.static_input_size)
    historical_inputs = torch.randn(
        batch_size,
        config.encoder_length,
        config.temporal_observed_size + config.target_size
    )
    future_inputs = torch.randn(
        batch_size,
        config.decoder_length,
        config.temporal_known_size
    )

    print(f"\nInput shapes:")
    print(f"  Static: {static_inputs.shape}")
    print(f"  Historical: {historical_inputs.shape}")
    print(f"  Future: {future_inputs.shape}")

    # Forward pass
    outputs = model(
        static_inputs=static_inputs,
        historical_inputs=historical_inputs,
        future_inputs=future_inputs,
        return_attention=True
    )

    print(f"\nOutputs:")
    print(f"  Predictions: {outputs['predictions'].shape}")
    print(f"  Quantiles: {config.quantiles}")

    if 'attention_weights' in outputs:
        print(f"  Attention weights: {outputs['attention_weights'].shape}")

    # Prediction at median
    median_pred = model.predict(
        static_inputs=static_inputs,
        historical_inputs=historical_inputs,
        future_inputs=future_inputs,
        quantile=0.5
    )
    print(f"\nMedian prediction: {median_pred.shape}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {num_params:,}")

    print("\n" + "="*80)
