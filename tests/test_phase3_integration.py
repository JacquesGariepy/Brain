"""
Integration Tests for Phase 3 - Unified Architecture Interface

Tests:
1. BrainArchitecture interface compliance
2. Orchestrator execution with different model types
3. ModelOutput consistency
4. API endpoint functionality (without server)
5. CLI predict functionality (without actual CLI call)

Run with:
    pytest tests/test_phase3_integration.py -v
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestBrainArchitectureInterface:
    """Test that key architectures implement BrainArchitecture correctly"""

    def test_transformer_implements_interface(self):
        """Test Transformer implements BrainArchitecture"""
        from architectures.transformers.transformer import Transformer, TransformerConfig
        from architectures.base import LanguageArchitecture, ModelOutput

        config = TransformerConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=512,
            max_seq_len=64,
        )

        model = Transformer(config)

        # Check inheritance
        assert isinstance(model, LanguageArchitecture), "Transformer should inherit from LanguageArchitecture"

        # Check methods exist
        assert hasattr(model, 'forward'), "Missing forward method"
        assert hasattr(model, 'predict'), "Missing predict method"
        assert hasattr(model, 'train_step'), "Missing train_step method"
        assert hasattr(model, 'save_pretrained'), "Missing save_pretrained method"
        assert hasattr(model, 'count_parameters'), "Missing count_parameters method"

        # Test forward returns ModelOutput
        input_ids = torch.randint(0, 1000, (2, 32))
        output = model(input_ids)

        assert isinstance(output, ModelOutput), "forward() should return ModelOutput"
        assert output.logits is not None, "ModelOutput should contain logits"
        assert output.logits.shape == (2, 32, 1000), f"Unexpected logits shape: {output.logits.shape}"

    def test_vision_transformer_implements_interface(self):
        """Test VisionTransformer implements BrainArchitecture"""
        from architectures.vision.vision_transformer import VisionTransformer, ViTConfig
        from architectures.base import VisionArchitecture, ModelOutput

        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_classes=10,
            d_model=192,
            num_layers=2,
            num_heads=3,
        )

        model = VisionTransformer(config)

        # Check inheritance
        assert isinstance(model, VisionArchitecture), "ViT should inherit from VisionArchitecture"

        # Test forward returns ModelOutput
        images = torch.randn(2, 3, 224, 224)
        output = model(images)

        assert isinstance(output, ModelOutput), "forward() should return ModelOutput"
        assert output.logits is not None, "ModelOutput should contain logits"
        assert output.logits.shape == (2, 10), f"Unexpected logits shape: {output.logits.shape}"

    def test_clip_implements_interface(self):
        """Test CLIP implements BrainArchitecture"""
        from architectures.multimodal.clip import CLIP, CLIPConfig
        from architectures.base import MultimodalArchitecture, ModelOutput

        config = CLIPConfig(
            image_size=224,
            patch_size=16,
            vision_width=192,
            vision_layers=2,
            vision_heads=3,
            text_width=128,
            text_layers=2,
            text_heads=2,
            embed_dim=128,
        )

        model = CLIP(config)

        # Check inheritance
        assert isinstance(model, MultimodalArchitecture), "CLIP should inherit from MultimodalArchitecture"

        # Test forward returns ModelOutput
        images = torch.randn(2, 3, 224, 224)
        text = torch.randint(0, 49408, (2, 77))

        output = model(images, text, return_loss=True)

        assert isinstance(output, ModelOutput), "forward() should return ModelOutput"
        assert output.embeddings is not None, "ModelOutput should contain embeddings"
        assert output.metadata is not None, "ModelOutput should contain metadata"
        assert 'image_embeds' in output.metadata, "Metadata should contain image embeddings"
        assert 'text_embeds' in output.metadata, "Metadata should contain text embeddings"

    def test_model_output_with_loss(self):
        """Test models compute loss when labels provided"""
        from architectures.transformers.transformer import Transformer, TransformerConfig
        from architectures.vision.vision_transformer import VisionTransformer, ViTConfig

        # Test language model loss
        config = TransformerConfig(vocab_size=1000, d_model=128, num_layers=2, num_heads=4)
        model = Transformer(config)

        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))

        output = model(input_ids, labels=labels)
        assert output.loss is not None, "Model should compute loss when labels provided"
        assert output.loss.requires_grad, "Loss should require gradients"

        # Test vision model loss
        config = ViTConfig(image_size=224, patch_size=16, num_classes=10, d_model=192, num_layers=2, num_heads=3)
        model = VisionTransformer(config)

        images = torch.randn(2, 3, 224, 224)
        labels = torch.randint(0, 10, (2,))

        output = model(images, labels=labels)
        assert output.loss is not None, "Model should compute loss when labels provided"


class TestOrchestratorExecution:
    """Test orchestrator can execute models properly"""

    def test_orchestrator_loads_transformer(self):
        """Test orchestrator can load Transformer"""
        from core.orchestrator import BrainOrchestrator

        orchestrator = BrainOrchestrator()
        model = orchestrator._load_transformer()

        assert model is not None, "Transformer should load successfully"

    def test_orchestrator_loads_vit(self):
        """Test orchestrator can load ViT"""
        from core.orchestrator import BrainOrchestrator

        orchestrator = BrainOrchestrator()
        model = orchestrator._load_vit()

        assert model is not None, "ViT should load successfully"

    def test_orchestrator_loads_clip(self):
        """Test orchestrator can load CLIP"""
        from core.orchestrator import BrainOrchestrator

        orchestrator = BrainOrchestrator()
        model = orchestrator._load_clip()

        assert model is not None, "CLIP should load successfully"

    def test_orchestrator_execute_vision(self):
        """Test orchestrator executes vision model"""
        from core.orchestrator import BrainOrchestrator

        orchestrator = BrainOrchestrator()

        # Load ViT
        model = orchestrator._load_vit()
        model.eval()

        # Prepare inputs
        inputs = {'image': torch.randn(1, 3, 224, 224)}

        # Execute
        output = orchestrator._execute_vision(model, inputs, 'cpu')

        from architectures.base import ModelOutput
        assert isinstance(output, ModelOutput), "Should return ModelOutput"

    def test_orchestrator_execute_language(self):
        """Test orchestrator executes language model"""
        from core.orchestrator import BrainOrchestrator

        orchestrator = BrainOrchestrator()

        # Load Transformer
        model = orchestrator._load_transformer()
        model.eval()

        # Prepare inputs
        inputs = {'input_ids': torch.randint(0, 50000, (1, 32))}

        # Execute
        output = orchestrator._execute_language(model, inputs, 'cpu')

        from architectures.base import ModelOutput
        assert isinstance(output, ModelOutput), "Should return ModelOutput"

    def test_orchestrator_execute_multimodal(self):
        """Test orchestrator executes multimodal model"""
        from core.orchestrator import BrainOrchestrator

        orchestrator = BrainOrchestrator()

        # Load CLIP
        model = orchestrator._load_clip()
        model.eval()

        # Prepare inputs
        inputs = {
            'image': torch.randn(1, 3, 224, 224),
            'input_ids': torch.randint(0, 49408, (1, 77)),
        }

        # Execute
        output = orchestrator._execute_multimodal(model, inputs, 'cpu')

        from architectures.base import ModelOutput
        assert isinstance(output, ModelOutput), "Should return ModelOutput"


class TestModelManager:
    """Test API ModelManager functionality"""

    def test_model_manager_loads_transformer(self):
        """Test ModelManager loads transformer"""
        # Import from API
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

        try:
            from app import ModelManager
        except ImportError:
            pytest.skip("FastAPI not available")

        manager = ModelManager()

        # Load model
        model = manager.load_model('transformer')

        assert model is not None, "Model should load"
        assert 'transformer' in manager.loaded_models, "Model should be cached"

    def test_model_manager_caches_models(self):
        """Test ModelManager caches loaded models"""
        try:
            from api.app import ModelManager
        except ImportError:
            pytest.skip("FastAPI not available")

        manager = ModelManager()

        # Load same model twice
        model1 = manager.load_model('transformer')
        model2 = manager.load_model('transformer')

        assert model1 is model2, "Should return cached model"


class TestParameterCounting:
    """Test parameter counting functionality"""

    def test_transformer_parameter_count(self):
        """Test Transformer parameter counting"""
        from architectures.transformers.transformer import Transformer, TransformerConfig

        config = TransformerConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
        )

        model = Transformer(config)
        params = model.count_parameters()

        assert 'total' in params, "Should return total parameters"
        assert 'trainable' in params, "Should return trainable parameters"
        assert 'non_trainable' in params, "Should return non-trainable parameters"

        assert params['total'] > 0, "Should have parameters"
        assert params['trainable'] == params['total'], "All parameters should be trainable by default"

    def test_freeze_unfreeze(self):
        """Test freeze/unfreeze functionality"""
        from architectures.transformers.transformer import Transformer, TransformerConfig

        config = TransformerConfig(vocab_size=1000, d_model=128, num_layers=2, num_heads=4)
        model = Transformer(config)

        # Initially all trainable
        params_before = model.count_parameters()
        assert params_before['trainable'] > 0

        # Freeze
        model.freeze()
        params_frozen = model.count_parameters()
        assert params_frozen['trainable'] == 0, "All parameters should be frozen"

        # Unfreeze
        model.unfreeze()
        params_after = model.count_parameters()
        assert params_after['trainable'] == params_before['trainable'], "Parameters should be unfrozen"


def test_end_to_end_inference():
    """End-to-end test: load model, run inference, get predictions"""
    from core.orchestrator import BrainOrchestrator
    from architectures.base import ModelOutput

    orchestrator = BrainOrchestrator()

    # Test with Transformer
    print("\nTesting Transformer end-to-end...")
    model = orchestrator._load_transformer()
    model.eval()

    input_ids = torch.randint(0, 50000, (2, 32))

    with torch.no_grad():
        output = model(input_ids=input_ids)

    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.logits.shape[0] == 2  # Batch size
    print(f"✓ Transformer inference successful: {output.logits.shape}")

    # Test with ViT
    print("Testing VisionTransformer end-to-end...")
    model = orchestrator._load_vit()
    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(x=images)

    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.logits.shape[0] == 2  # Batch size
    print(f"✓ ViT inference successful: {output.logits.shape}")

    # Test with CLIP
    print("Testing CLIP end-to-end...")
    model = orchestrator._load_clip()
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    text = torch.randint(0, 49408, (2, 77))

    with torch.no_grad():
        output = model(image=images, text=text)

    assert isinstance(output, ModelOutput)
    assert output.embeddings is not None
    print(f"✓ CLIP inference successful: {output.embeddings.shape}")

    print("\n" + "=" * 70)
    print("All Phase 3 integration tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
