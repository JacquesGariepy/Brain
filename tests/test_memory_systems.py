"""
Tests for Memory Systems (NTM, DNC, Memory Networks)
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from architectures.memory.neural_memory import (
    NTMMemory,
    NTMController,
    NeuralTuringMachine,
    NTMConfig,
    DifferentiableNeuralComputer,
    DNCConfig,
    MemoryNetwork
)


class TestNTMMemory:
    """Test NTM memory operations"""

    def test_content_addressing(self):
        """Test content-based addressing"""
        memory_module = NTMMemory(memory_size=10, memory_dim=8)

        batch_size = 2
        memory = torch.randn(batch_size, 10, 8)
        key = torch.randn(batch_size, 8)
        strength = torch.ones(batch_size, 1)

        weights = memory_module.content_addressing(memory, key, strength)

        assert weights.shape == (batch_size, 10)
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    def test_read_operation(self):
        """Test memory read"""
        memory_module = NTMMemory(memory_size=10, memory_dim=8)

        batch_size = 2
        memory = torch.randn(batch_size, 10, 8)
        weights = torch.softmax(torch.randn(batch_size, 10), dim=-1)

        read_vector = memory_module.read(memory, weights)

        assert read_vector.shape == (batch_size, 8)

    def test_write_operation(self):
        """Test memory write with erase and add"""
        memory_module = NTMMemory(memory_size=10, memory_dim=8)

        batch_size = 2
        memory = torch.randn(batch_size, 10, 8)
        weights = torch.softmax(torch.randn(batch_size, 10), dim=-1)
        erase = torch.sigmoid(torch.randn(batch_size, 8))
        add = torch.randn(batch_size, 8)

        new_memory = memory_module.write(memory, weights, erase, add)

        assert new_memory.shape == memory.shape


class TestNeuralTuringMachine:
    """Test complete NTM"""

    def test_ntm_forward(self):
        """Test NTM forward pass"""
        config = NTMConfig(
            input_size=10,
            output_size=10,
            controller_size=64,
            memory_size=20,
            memory_dim=8,
            num_heads=1
        )

        ntm = NeuralTuringMachine(config)

        batch_size = 2
        x = torch.randn(batch_size, config.input_size)

        output, memory, weights, hidden = ntm(x)

        assert output.shape == (batch_size, config.output_size)
        assert memory.shape == (batch_size, config.memory_size, config.memory_dim)
        assert weights.shape == (batch_size, config.num_heads, config.memory_size)

    def test_ntm_sequence_processing(self):
        """Test NTM processing sequence"""
        config = NTMConfig(
            input_size=10,
            output_size=10,
            controller_size=64,
            memory_size=20,
            memory_dim=8
        )

        ntm = NeuralTuringMachine(config)

        batch_size, seq_len = 2, 5
        outputs = []

        memory = None
        weights = None
        hidden = None

        for t in range(seq_len):
            x = torch.randn(batch_size, config.input_size)
            output, memory, weights, hidden = ntm(x, memory, weights, hidden)
            outputs.append(output)

        assert len(outputs) == seq_len


class TestDifferentiableNeuralComputer:
    """Test DNC"""

    def test_dnc_forward(self):
        """Test DNC forward pass"""
        config = DNCConfig(
            input_size=10,
            output_size=10,
            controller_size=64,
            memory_size=30,
            memory_dim=8,
            num_read_heads=2,
            num_write_heads=1
        )

        dnc = DifferentiableNeuralComputer(config)

        batch_size = 2
        x = torch.randn(batch_size, config.input_size)

        output = dnc(x)
        assert output.shape == (batch_size, config.output_size)


class TestMemoryNetwork:
    """Test Memory Networks"""

    def test_memory_network_forward(self):
        """Test Memory Network forward pass"""
        vocab_size = 100
        embedding_dim = 64
        num_hops = 3
        memory_size = 10

        model = MemoryNetwork(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_hops=num_hops,
            memory_size=memory_size
        )

        batch_size = 2
        memory_len = 5
        question_len = 3

        memories = torch.randint(0, vocab_size, (batch_size, memory_size, memory_len))
        question = torch.randint(0, vocab_size, (batch_size, question_len))

        logits = model(memories, question)
        assert logits.shape == (batch_size, vocab_size)

    def test_memory_network_multiple_hops(self):
        """Test multi-hop reasoning"""
        for num_hops in [1, 2, 3, 5]:
            model = MemoryNetwork(
                vocab_size=50,
                embedding_dim=32,
                num_hops=num_hops,
                memory_size=5
            )

            memories = torch.randint(0, 50, (2, 5, 3))
            question = torch.randint(0, 50, (2, 2))

            logits = model(memories, question)
            assert logits.shape == (2, 50)


class TestMemoryTraining:
    """Test memory systems can be trained"""

    def test_ntm_backward_pass(self):
        """Test NTM backward pass"""
        config = NTMConfig(
            input_size=8,
            output_size=8,
            controller_size=32,
            memory_size=10,
            memory_dim=4
        )

        ntm = NeuralTuringMachine(config)
        optimizer = torch.optim.Adam(ntm.parameters(), lr=1e-3)

        # Forward
        x = torch.randn(2, config.input_size)
        target = torch.randn(2, config.output_size)

        output, _, _, _ = ntm(x)
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients
        for param in ntm.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
