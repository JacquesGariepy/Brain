"""
Integration tests for SOTA Brain
Tests that all components work together
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sota_brain import SOTABrain, SOTABrainConfig


class TestSOTABrainInitialization:
    """Test SOTA Brain initialization"""

    def test_basic_initialization(self):
        """Test basic brain initialization"""
        config = SOTABrainConfig(
            use_language=True,
            use_vision=False,
            language_model="transformer",
            d_model=128,
            num_layers=2,
            num_heads=4,
            vocab_size=100
        )

        brain = SOTABrain(config)
        assert brain is not None

    def test_multimodal_initialization(self):
        """Test multimodal brain"""
        config = SOTABrainConfig(
            use_language=True,
            use_vision=True,
            language_model="transformer",
            vision_model="vit",
            d_model=128,
            num_layers=2
        )

        brain = SOTABrain(config)
        assert hasattr(brain, 'language_model')
        assert hasattr(brain, 'vision_model')
        assert hasattr(brain, 'multimodal_fusion')

    def test_mamba_initialization(self):
        """Test with Mamba language model"""
        config = SOTABrainConfig(
            use_language=True,
            language_model="mamba",
            d_model=128,
            num_layers=2
        )

        brain = SOTABrain(config)
        assert hasattr(brain, 'language_model')


class TestSOTABrainForward:
    """Test forward passes"""

    def test_text_only_forward(self):
        """Test text-only forward pass"""
        config = SOTABrainConfig(
            use_language=True,
            use_vision=False,
            language_model="transformer",
            d_model=128,
            num_layers=2,
            num_heads=4,
            vocab_size=100,
            use_flash_attention=False
        )

        brain = SOTABrain(config)

        batch_size, seq_len = 2, 10
        text_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = brain(text_input=text_input)
        assert output.shape == (batch_size, seq_len, config.vocab_size)

    def test_vision_only_forward(self):
        """Test vision-only forward pass"""
        config = SOTABrainConfig(
            use_language=False,
            use_vision=True,
            vision_model="vit",
            d_model=128,
            num_layers=2
        )

        brain = SOTABrain(config)

        batch_size = 2
        image_input = torch.randn(batch_size, 3, 224, 224)

        output = brain(image_input=image_input)
        assert output.shape[0] == batch_size

    def test_multimodal_forward(self):
        """Test multimodal forward pass"""
        config = SOTABrainConfig(
            use_language=True,
            use_vision=True,
            language_model="transformer",
            vision_model="vit",
            d_model=128,
            num_layers=2,
            num_heads=4,
            use_flash_attention=False
        )

        brain = SOTABrain(config)

        batch_size = 2
        text_input = torch.randint(0, config.vocab_size, (batch_size, 10))
        image_input = torch.randn(batch_size, 3, 224, 224)

        output = brain(text_input=text_input, image_input=image_input)
        assert output.shape[0] == batch_size


class TestSOTABrainOptimizers:
    """Test optimizer configuration"""

    def test_adamw_optimizer(self):
        """Test AdamW optimizer"""
        config = SOTABrainConfig(
            use_language=True,
            optimizer="adamw",
            d_model=64,
            num_layers=2
        )

        brain = SOTABrain(config)
        optimizer = brain.configure_optimizer()

        assert optimizer is not None
        assert 'AdamW' in str(type(optimizer))

    def test_lion_optimizer(self):
        """Test Lion optimizer"""
        config = SOTABrainConfig(
            use_language=True,
            optimizer="lion",
            d_model=64,
            num_layers=2
        )

        brain = SOTABrain(config)
        optimizer = brain.configure_optimizer()

        assert optimizer is not None
        assert 'Lion' in str(type(optimizer))

    def test_sophia_optimizer(self):
        """Test Sophia optimizer"""
        config = SOTABrainConfig(
            use_language=True,
            optimizer="sophia",
            d_model=64,
            num_layers=2
        )

        brain = SOTABrain(config)
        optimizer = brain.configure_optimizer()

        assert optimizer is not None
        assert 'Sophia' in str(type(optimizer))


class TestSOTABrainTraining:
    """Test training capabilities"""

    def test_backward_pass(self):
        """Test backward pass works"""
        config = SOTABrainConfig(
            use_language=True,
            use_vision=False,
            language_model="transformer",
            d_model=64,
            num_layers=2,
            num_heads=4,
            vocab_size=50,
            use_flash_attention=False
        )

        brain = SOTABrain(config)
        optimizer = brain.configure_optimizer()

        # Forward
        text_input = torch.randint(0, config.vocab_size, (2, 10))
        target = torch.randint(0, config.vocab_size, (2, 10))

        output = brain(text_input=text_input)

        # Loss
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, config.vocab_size),
            target.view(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients
        for param in brain.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_training_step(self):
        """Test a full training step"""
        config = SOTABrainConfig(
            use_language=True,
            language_model="transformer",
            d_model=64,
            num_layers=2,
            vocab_size=50,
            use_flash_attention=False
        )

        brain = SOTABrain(config)
        optimizer = brain.configure_optimizer()

        # Create small dataset
        num_batches = 5
        losses = []

        for _ in range(num_batches):
            text_input = torch.randint(0, config.vocab_size, (4, 10))
            target = torch.randint(0, config.vocab_size, (4, 10))

            optimizer.zero_grad()
            output = brain(text_input=text_input)
            loss = torch.nn.functional.cross_entropy(
                output.view(-1, config.vocab_size),
                target.view(-1)
            )
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Should complete without errors
        assert len(losses) == num_batches


class TestComponentIntegration:
    """Test that different components work together"""

    def test_transformer_with_different_configs(self):
        """Test transformer with various configurations"""
        configs = [
            {"language_model": "transformer", "use_flash_attention": False},
            {"language_model": "transformer", "use_flash_attention": False, "use_lora": True},
        ]

        for config_dict in configs:
            config = SOTABrainConfig(
                use_language=True,
                d_model=64,
                num_layers=2,
                vocab_size=50,
                **config_dict
            )

            brain = SOTABrain(config)
            text_input = torch.randint(0, 50, (2, 10))

            with torch.no_grad():
                output = brain(text_input=text_input)

            assert output.shape == (2, 10, 50)

    def test_memory_integration(self):
        """Test memory system integration"""
        for memory_type in ["ntm", "dnc"]:
            config = SOTABrainConfig(
                use_language=True,
                memory_type=memory_type,
                d_model=64,
                num_layers=2,
                memory_size=20,
                memory_dim=8
            )

            brain = SOTABrain(config)
            assert hasattr(brain, 'memory')


class TestModelSaving:
    """Test model saving and loading"""

    def test_state_dict(self):
        """Test getting state dict"""
        config = SOTABrainConfig(
            use_language=True,
            d_model=64,
            num_layers=2,
            use_flash_attention=False
        )

        brain = SOTABrain(config)
        state_dict = brain.state_dict()

        assert state_dict is not None
        assert len(state_dict) > 0

    def test_parameter_count(self):
        """Test parameter counting"""
        config = SOTABrainConfig(
            use_language=True,
            d_model=64,
            num_layers=2,
            vocab_size=100,
            use_flash_attention=False
        )

        brain = SOTABrain(config)

        total_params = sum(p.numel() for p in brain.parameters())
        trainable_params = sum(p.numel() for p in brain.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All should be trainable by default


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
