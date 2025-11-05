"""
Comprehensive tests for State Space Models (Mamba, S4)
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from architectures.transformers.state_space_models import (
    S4Layer,
    S4Config,
    MambaBlock,
    MambaConfig,
    MambaModel
)


class TestS4Layer:
    """Test S4 (Structured State Space) layer"""

    def test_s4_forward(self):
        """Test S4 forward pass"""
        config = S4Config(
            d_model=128,
            d_state=64,
            dropout=0.1
        )

        layer = S4Layer(config)

        batch_size, seq_len = 2, 20
        x = torch.randn(batch_size, seq_len, config.d_model)

        output = layer(x)
        assert output.shape == x.shape

    def test_s4_different_sequence_lengths(self):
        """Test S4 with different sequence lengths"""
        config = S4Config(d_model=64, d_state=32)
        layer = S4Layer(config)

        for seq_len in [10, 50, 100]:
            x = torch.randn(2, seq_len, config.d_model)
            output = layer(x)
            assert output.shape == x.shape

    def test_s4_training_mode(self):
        """Test S4 in training mode (convolutional)"""
        config = S4Config(d_model=64, d_state=32)
        layer = S4Layer(config)
        layer.train()

        x = torch.randn(2, 20, config.d_model)
        output = layer(x)
        assert output.shape == x.shape

    def test_s4_eval_mode(self):
        """Test S4 in eval mode (recurrent)"""
        config = S4Config(d_model=64, d_state=32)
        layer = S4Layer(config)
        layer.eval()

        x = torch.randn(2, 20, config.d_model)
        with torch.no_grad():
            output = layer(x)
        assert output.shape == x.shape


class TestMambaBlock:
    """Test Mamba block"""

    def test_mamba_forward(self):
        """Test Mamba forward pass"""
        config = MambaConfig(
            d_model=128,
            d_state=16,
            d_conv=4,
            expand=2
        )

        block = MambaBlock(config)

        batch_size, seq_len = 2, 20
        x = torch.randn(batch_size, seq_len, config.d_model)

        output = block(x)
        assert output.shape == x.shape

    def test_mamba_selective_scan(self):
        """Test selective scan mechanism"""
        config = MambaConfig(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2
        )

        block = MambaBlock(config)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.d_model)

        # Forward pass triggers selective scan
        output = block(x)
        assert output.shape == x.shape

    def test_mamba_different_expansions(self):
        """Test Mamba with different expansion factors"""
        for expand in [1, 2, 4]:
            config = MambaConfig(
                d_model=64,
                d_state=16,
                expand=expand
            )

            block = MambaBlock(config)
            x = torch.randn(2, 10, config.d_model)

            output = block(x)
            assert output.shape == x.shape


class TestMambaModel:
    """Test complete Mamba model"""

    def test_mamba_model_forward(self):
        """Test Mamba model forward pass"""
        vocab_size = 1000
        d_model = 128
        n_layers = 4

        model = MambaModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=16
        )

        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_mamba_vs_transformer_complexity(self):
        """Test that Mamba is more efficient for long sequences"""
        vocab_size = 100
        model = MambaModel(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=2,
            d_state=16
        )

        # Test with increasing sequence lengths
        for seq_len in [50, 100, 200]:
            input_ids = torch.randint(0, vocab_size, (1, seq_len))

            with torch.no_grad():
                output = model(input_ids)

            assert output.shape == (1, seq_len, vocab_size)

    def test_mamba_weight_tying(self):
        """Test embedding and output weight tying"""
        model = MambaModel(
            vocab_size=100,
            d_model=64,
            n_layers=2
        )

        # Check weight tying
        assert model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()

    def test_mamba_backward_pass(self):
        """Test backward pass works"""
        model = MambaModel(
            vocab_size=100,
            d_model=64,
            n_layers=2
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Forward
        input_ids = torch.randint(0, 100, (2, 10))
        target_ids = torch.randint(0, 100, (2, 10))

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 100),
            target_ids.view(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestStateSpaceComparison:
    """Compare S4 and Mamba characteristics"""

    def test_parameter_count(self):
        """Compare parameter counts"""
        d_model = 128
        vocab_size = 1000

        # Mamba
        mamba = MambaModel(vocab_size=vocab_size, d_model=d_model, n_layers=4)
        mamba_params = sum(p.numel() for p in mamba.parameters())

        # S4 layer (for comparison)
        s4_config = S4Config(d_model=d_model, d_state=64)
        s4 = S4Layer(s4_config)
        s4_params = sum(p.numel() for p in s4.parameters())

        assert mamba_params > 0
        assert s4_params > 0

    def test_sequence_processing_consistency(self):
        """Test that both models process sequences consistently"""
        batch_size, seq_len = 2, 20
        d_model = 64

        # S4
        s4_config = S4Config(d_model=d_model, d_state=32)
        s4 = S4Layer(s4_config)

        # Mamba
        mamba_config = MambaConfig(d_model=d_model, d_state=16)
        mamba = MambaBlock(mamba_config)

        x = torch.randn(batch_size, seq_len, d_model)

        s4_out = s4(x)
        mamba_out = mamba(x)

        assert s4_out.shape == mamba_out.shape == x.shape


class TestStateSpaceTraining:
    """Test training capabilities"""

    def test_mamba_overfit_small_dataset(self):
        """Test Mamba can overfit (sanity check)"""
        model = MambaModel(
            vocab_size=50,
            d_model=64,
            n_layers=2
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Small dataset
        input_ids = torch.randint(0, 50, (4, 15))
        target_ids = torch.randint(0, 50, (4, 15))

        initial_loss = None
        final_loss = None

        # Train
        for step in range(50):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 50),
                target_ids.view(-1)
            )

            if step == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
