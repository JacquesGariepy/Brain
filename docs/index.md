# Brain Framework Documentation

Welcome to the Brain Framework documentation! Brain is a comprehensive AI toolkit with 46+ state-of-the-art (SOTA) architectures for machine learning research and production.

## 🚀 Quick Links

- **[Installation Guide](installation.md)** - Get started in 5 minutes
- **[Quickstart Tutorial](quickstart.md)** - Your first model in minutes
- **[API Reference](api/index.md)** - Complete API documentation
- **[Examples](../examples/)** - Practical code examples
- **[User Guide](user_guide.md)** - In-depth tutorials

## 📦 What's Included

Brain Framework includes:

### Core Components
- **46+ SOTA Architectures** - Transformers, vision models, audio models, and more
- **Data Pipeline** - Comprehensive data loading and preprocessing
- **Training Infrastructure** - Distributed training, mixed precision, optimization
- **Monitoring & Logging** - WandB, MLflow, TensorBoard integration
- **REST API** - FastAPI-based model serving
- **CLI Tools** - Command-line interface for all operations

### Architecture Categories

| Category | Models | Count |
|----------|--------|-------|
| **Transformers** | BERT, GPT, T5, Llama | 8+ |
| **Vision** | ViT, CLIP, SAM, YOLO | 10+ |
| **Audio** | Whisper, Wav2Vec2, MusicGen | 5+ |
| **Multimodal** | BLIP-2, LLaVA, Flamingo | 4+ |
| **Scientific** | AlphaFold, MoleculeVAE | 3+ |
| **Time Series** | N-BEATS, TFT, PatchTST | 4+ |
| **Training** | LoRA, QLoRA, DeepSpeed | 6+ |
| **Inference** | Quantization, Pruning, ONNX | 6+ |

## 🎯 Key Features

### 1. Production-Ready
```bash
# Install
pip install brain-framework

# Train a model
brain train --model bert-base --dataset glue/sst2

# Serve via API
brain serve --port 8000
```

### 2. Scientific Research
```python
from brain import BrainModel
from utils.data import get_cifar10_loaders
from utils.logging import WandBLogger

# Load data
train_loader, test_loader = get_cifar10_loaders(batch_size=32)

# Initialize model and logging
model = BrainModel("resnet50")
logger = WandBLogger(project="my-research")

# Train with full observability
model.train(train_loader, logger=logger)
```

### 3. Distributed Training
```bash
# Multi-GPU training with DeepSpeed
brain train \
  --model gpt-2 \
  --dataset wikitext \
  --distributed \
  --gpus 8 \
  --deepspeed
```

### 4. Easy Deployment
```bash
# Build Docker image
docker build -t brain-api .

# Start services
docker-compose up -d

# API available at http://localhost:8000
```

## 📚 Documentation Structure

### Getting Started
1. [Installation](installation.md) - Installation and setup
2. [Quickstart](quickstart.md) - First steps with Brain
3. [Basic Concepts](concepts.md) - Core concepts and terminology

### User Guide
4. [Data Handling](user_guide/data.md) - Loading and preprocessing data
5. [Training Models](user_guide/training.md) - Training and fine-tuning
6. [Evaluation](user_guide/evaluation.md) - Model evaluation and metrics
7. [Inference](user_guide/inference.md) - Running inference
8. [Distributed Training](user_guide/distributed.md) - Multi-GPU and multi-node

### Advanced Topics
9. [Model Optimization](advanced/optimization.md) - Quantization, pruning, distillation
10. [Custom Architectures](advanced/custom_models.md) - Building custom models
11. [Production Deployment](advanced/deployment.md) - Deploying to production
12. [Monitoring](advanced/monitoring.md) - Logging and monitoring

### API Reference
13. [Core API](api/core.md) - Core framework APIs
14. [Architectures](api/architectures.md) - Model architectures
15. [Utils](api/utils.md) - Utility functions
16. [CLI](api/cli.md) - Command-line interface

## 🌟 Quick Examples

### Text Classification
```python
from brain import BrainModel

model = BrainModel("bert-base-uncased")
prediction = model.predict("This movie is amazing!")
print(prediction)  # {"label": "positive", "confidence": 0.95}
```

### Image Classification
```python
from brain import BrainModel
from PIL import Image

model = BrainModel("resnet50")
image = Image.open("cat.jpg")
prediction = model.predict(image)
print(prediction)  # {"label": "cat", "confidence": 0.98}
```

### Multimodal
```python
from architectures.multimodal.clip import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base")
similarity = model.compute_similarity(
    image="image.jpg",
    texts=["a cat", "a dog", "a bird"]
)
print(similarity)  # [0.85, 0.12, 0.03]
```

## 💡 Use Cases

Brain Framework is perfect for:

- **Research** - Experiment with SOTA architectures
- **Production** - Deploy models at scale
- **Education** - Learn modern ML techniques
- **Prototyping** - Quickly test ideas
- **Competition** - Kaggle, research competitions

## 🤝 Community & Support

- **GitHub Issues** - Report bugs and request features
- **Discussions** - Ask questions and share ideas
- **Examples** - Check out our [examples directory](../examples/)
- **Contributing** - See our [contributing guide](../CONTRIBUTING.md)

## 📄 License

Brain Framework is released under the MIT License. See [LICENSE](../LICENSE) for details.

## 🚦 Next Steps

Ready to get started?

1. **[Install Brain](installation.md)** - Set up your environment
2. **[Try the Quickstart](quickstart.md)** - Train your first model
3. **[Explore Examples](../examples/)** - Learn from practical examples
4. **[Read the User Guide](user_guide.md)** - Deep dive into features

---

**Happy Building! 🧠✨**
