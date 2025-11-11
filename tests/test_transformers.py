"""
Comprehensive tests for Transformer architectures
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from architectures.transformers.multihead_attention import (
    MultiHeadAttention,
    AttentionConfig,
    GroupedQueryAttention,
    MultiQueryAttention
)
from architectures.transformers.transformer import (
    Transformer,
    TransformerConfig,
    TransformerBlock,
    RMSNorm,
    SwiGLU,
    GeGLU,
    RotaryPositionEmbedding
)


class TestMultiHeadAttention:
    """Test Multi-Head Attention implementations"""

    def test_multihead_attention_forward(self):
        """Test basic forward pass"""
        config = AttentionConfig(
            d_model=256,
            num_heads=8,
            dropout=0.1,
            causal=False,
            use_flash=False
        )

        mha = MultiHeadAttention(config)
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.d_model)

        output, cache = mha(x)

        assert output.shape == (batch_size, seq_len, config.d_model)
        assert cache is None  # No cache when use_cache=False

    def test_multihead_attention_with_mask(self):
        """Test attention with mask"""
        config = AttentionConfig(d_model=128, num_heads=4)
        mha = MultiHeadAttention(config)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.d_model)
        mask = torch.ones(seq_len, seq_len)
        mask[:, 5:] = 0  # Mask out second half

        output, _ = mha(x, mask=mask)
        assert output.shape == (batch_size, seq_len, config.d_model)

    def test_multihead_attention_causal(self):
        """Test causal (autoregressive) attention"""
        config = AttentionConfig(
            d_model=128,
            num_heads=4,
            causal=True,
            use_flash=False
        )
        mha = MultiHeadAttention(config)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.d_model)

        output, _ = mha(x)
        assert output.shape == (batch_size, seq_len, config.d_model)

    def test_multihead_attention_kv_cache(self):
        """Test KV caching for autoregressive generation"""
        config = AttentionConfig(d_model=128, num_heads=4)
        mha = MultiHeadAttention(config)

        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, config.d_model)

        # First forward pass with cache
        output1, cache = mha(x, use_cache=True)
        assert cache is not None
        assert len(cache) == 2  # (k, v)

        # Second forward pass reusing cache
        new_x = torch.randn(batch_size, 1, config.d_model)
        output2, new_cache = mha(new_x, kv_cache=cache, use_cache=True)
        assert output2.shape == (batch_size, 1, config.d_model)

    def test_grouped_query_attention(self):
        """Test Grouped-Query Attention (GQA)"""
        gqa = GroupedQueryAttention(
            d_model=256,
            num_heads=8,
            num_kv_heads=2,  # 4 queries per KV head
            dropout=0.1
        )

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 256)

        output = gqa(x)
        assert output.shape == (batch_size, seq_len, 256)

    def test_multi_query_attention(self):
        """Test Multi-Query Attention (MQA)"""
        mqa = MultiQueryAttention(
            d_model=256,
            num_heads=8,
            dropout=0.1
        )

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 256)

        output = mqa(x)
        assert output.shape == (batch_size, seq_len, 256)

    def test_attention_maps(self):
        """Test getting attention maps for visualization"""
        config = AttentionConfig(d_model=128, num_heads=4, use_flash=False)
        mha = MultiHeadAttention(config)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.d_model)

        attn_maps = mha.get_attention_maps(x)
        assert attn_maps.shape == (batch_size, config.num_heads, seq_len, seq_len)

        # Check attention weights sum to 1
        assert torch.allclose(attn_maps.sum(dim=-1), torch.ones_like(attn_maps.sum(dim=-1)), atol=1e-5)


class TestTransformerComponents:
    """Test individual Transformer components"""

    def test_rmsnorm(self):
        """Test RMSNorm layer"""
        d_model = 256
        norm = RMSNorm(d_model)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        output = norm(x)
        assert output.shape == x.shape

    def test_swiglu(self):
        """Test SwiGLU activation"""
        d_model, d_ff = 256, 1024
        swiglu = SwiGLU(d_model, d_ff)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        output = swiglu(x)
        assert output.shape == x.shape

    def test_geglu(self):
        """Test GeGLU activation"""
        d_model, d_ff = 256, 1024
        geglu = GeGLU(d_model, d_ff)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        output = geglu(x)
        assert output.shape == x.shape

    def test_rotary_position_embedding(self):
        """Test RoPE (Rotary Position Embeddings)"""
        dim = 64
        rope = RotaryPositionEmbedding(dim)

        seq_len = 20
        x = torch.randn(2, seq_len, dim)

        cos, sin = rope(x, seq_len)
        assert cos.shape == (seq_len, dim)
        assert sin.shape == (seq_len, dim)


class TestTransformerBlock:
    """Test Transformer block"""

    def test_transformer_block_forward(self):
        """Test transformer block forward pass"""
        config = TransformerConfig(
            d_model=256,
            num_heads=8,
            d_ff=1024,
            dropout=0.1,
            use_flash=False
        )

        block = TransformerBlock(config)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.d_model)

        output = block(x)
        assert output.shape == x.shape

    def test_transformer_block_parallel(self):
        """Test parallel attention+FFN (GPT-J style)"""
        config = TransformerConfig(
            d_model=256,
            num_heads=8,
            parallel_attn_ffn=True,
            use_flash=False
        )

        block = TransformerBlock(config)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.d_model)

        output = block(x)
        assert output.shape == x.shape


class TestTransformer:
    """Test complete Transformer model"""

    def test_transformer_forward(self):
        """Test transformer forward pass"""
        config = TransformerConfig(
            d_model=256,
            num_layers=4,
            num_heads=8,
            vocab_size=1000,
            max_seq_len=128,
            use_flash=False,
            use_rope=False,
            causal=True
        )

        model = Transformer(config)

        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_transformer_with_rope(self):
        """Test transformer with RoPE"""
        config = TransformerConfig(
            d_model=256,
            num_layers=4,
            num_heads=8,
            vocab_size=1000,
            use_rope=True,
            use_flash=False
        )

        model = Transformer(config)

        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_transformer_generation(self):
        """Test autoregressive generation"""
        config = TransformerConfig(
            d_model=128,
            num_layers=2,
            num_heads=4,
            vocab_size=100,
            max_seq_len=50,
            use_flash=False,
            causal=True
        )

        model = Transformer(config)
        model.eval()

        # Start with single token
        input_ids = torch.randint(0, config.vocab_size, (1, 1))

        # Generate 10 tokens
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0
            )

        assert generated.shape[1] == 11  # Original + 10 new tokens

    def test_transformer_with_gqa(self):
        """Test transformer with Grouped-Query Attention"""
        config = TransformerConfig(
            d_model=256,
            num_layers=4,
            num_heads=8,
            vocab_size=1000,
            use_gqa=True,
            num_kv_heads=2,
            use_flash=False
        )

        model = Transformer(config)

        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_transformer_weight_tying(self):
        """Test that embedding and output weights are tied"""
        config = TransformerConfig(
            d_model=256,
            num_layers=2,
            num_heads=8,
            vocab_size=1000,
            use_flash=False
        )

        model = Transformer(config)

        # Check weight tying
        assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()


class TestTransformerTraining:
    """Test transformer can be trained"""

    def test_transformer_backward_pass(self):
        """Test backward pass and gradient computation"""
        config = TransformerConfig(
            d_model=128,
            num_layers=2,
            num_heads=4,
            vocab_size=100,
            use_flash=False
        )

        model = Transformer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)

        # Loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            target_ids.view(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_transformer_overfitting_small_data(self):
        """Test model can overfit small dataset (sanity check)"""
        config = TransformerConfig(
            d_model=64,
            num_layers=2,
            num_heads=4,
            vocab_size=50,
            use_flash=False
        )

        model = Transformer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Small dataset
        input_ids = torch.randint(0, config.vocab_size, (4, 10))
        target_ids = torch.randint(0, config.vocab_size, (4, 10))

        initial_loss = None
        final_loss = None

        # Train for a few steps
        for step in range(50):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, config.vocab_size),
                target_ids.view(-1)
            )

            if step == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
