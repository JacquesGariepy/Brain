# Quickstart Guide

Get started with Brain Framework in 5 minutes! This guide will walk you through training your first model.

## Prerequisites

Make sure Brain is installed:

```bash
pip install brain-framework[full]
```

## Your First Model (CLI)

The fastest way to train a model:

```bash
# Train BERT on sentiment analysis
brain train \
  --model bert-base-uncased \
  --dataset glue/sst2 \
  --epochs 3 \
  --batch-size 32 \
  --output-dir ./my_model

# Evaluate the model
brain evaluate \
  --model ./my_model/checkpoint.pt \
  --dataset glue/sst2

# Make predictions
brain predict \
  --model ./my_model/checkpoint.pt \
  --text "This movie is amazing!"
```

## Your First Model (Python)

### 1. Text Classification with BERT

```python
from brain import BrainModel
from utils.data import get_text_dataset
from utils.logging import WandBLogger

# Initialize model
model = BrainModel("bert-base-uncased", num_classes=2)

# Load dataset
train_dataset = get_text_dataset("glue", "sst2", split="train")
val_dataset = get_text_dataset("glue", "sst2", split="validation")

# Initialize logger (optional)
logger = WandBLogger(project="my-first-model", name="bert-sst2")

# Train
model.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=3,
    batch_size=32,
    learning_rate=2e-5,
    logger=logger,
)

# Evaluate
results = model.evaluate(val_dataset)
print(f"Accuracy: {results['accuracy']:.4f}")

# Predict
prediction = model.predict("This movie is amazing!")
print(f"Label: {prediction['label']}")
print(f"Confidence: {prediction['confidence']:.4f}")

# Save model
model.save("./my_model")
```

### 2. Image Classification with ResNet

```python
from brain import BrainModel
from utils.data import get_cifar10_loaders

# Initialize model
model = BrainModel("resnet50", num_classes=10)

# Load data
train_loader, test_loader = get_cifar10_loaders(
    batch_size=64,
    augment=True
)

# Train
model.train(
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=10,
    learning_rate=0.001,
)

# Evaluate
accuracy = model.evaluate(test_loader)
print(f"Test Accuracy: {accuracy:.4f}")
```

### 3. Multimodal with CLIP

```python
from architectures.multimodal.clip import CLIPModel
import torch
from PIL import Image

# Load model
model = CLIPModel(
    image_size=224,
    patch_size=16,
    hidden_size=512,
    num_heads=8,
    num_layers=12,
    vocab_size=49408,
)

# Prepare inputs
image = torch.randn(1, 3, 224, 224)
text_tokens = torch.randint(0, 49408, (1, 77))

# Get embeddings
with torch.no_grad():
    outputs = model(image, text_tokens)

print(f"Image embedding: {outputs['image_embeds'].shape}")
print(f"Text embedding: {outputs['text_embeds'].shape}")
print(f"Similarity: {outputs['similarity'][0, 0]:.4f}")
```

## Training with Monitoring

### WandB Integration

```python
from brain import BrainModel
from utils.logging import WandBLogger

# Initialize logger
logger = WandBLogger(
    project="my-project",
    name="experiment-1",
    config={
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 32,
    }
)

# Train with logging
model = BrainModel("bert-base-uncased")
model.train(
    train_dataset=train_dataset,
    epochs=3,
    logger=logger,
)

# Finish logging
logger.finish()
```

### MLflow Integration

```python
from utils.logging import MLflowLogger

# Initialize logger
logger = MLflowLogger(
    experiment_name="my-experiment",
    run_name="run-1",
)

# Log parameters
logger.log_params({
    "model": "bert-base-uncased",
    "learning_rate": 2e-5,
    "batch_size": 32,
})

# Train with logging
model.train(train_dataset=train_dataset, logger=logger)

# Log metrics
logger.log_metrics({
    "accuracy": 0.95,
    "loss": 0.12,
}, step=1000)

# End run
logger.end_run()
```

### TensorBoard Integration

```python
from utils.logging import TensorBoardLogger

# Initialize logger
logger = TensorBoardLogger(log_dir="./runs/experiment1")

# Train with logging
model.train(train_dataset=train_dataset, logger=logger)

# View logs
# tensorboard --logdir=./runs
```

## API Server

Start the Brain API server:

```bash
# Start server
brain serve --port 8000

# In another terminal, make predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is amazing!",
    "model_name": "bert-base-uncased"
  }'
```

Or use Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "This is amazing!",
        "model_name": "bert-base-uncased"
    }
)

result = response.json()
print(result)
```

## Docker Quickstart

### Using Docker

```bash
# Pull image
docker pull brain-framework:latest

# Run container
docker run -it --gpus all -p 8000:8000 brain-framework:latest

# API available at http://localhost:8000
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Services:
# - Brain API: http://localhost:8000
# - TensorBoard: http://localhost:6006
# - MLflow: http://localhost:5000
# - Jupyter: http://localhost:8888
```

## Data Loading

### Standard Datasets

```python
from utils.data import (
    get_mnist_loaders,
    get_cifar10_loaders,
    get_imagenet_loaders,
)

# MNIST
train_loader, test_loader = get_mnist_loaders(batch_size=64)

# CIFAR-10
train_loader, test_loader = get_cifar10_loaders(
    batch_size=64,
    augment=True
)

# ImageNet (requires manual download)
train_loader, val_loader = get_imagenet_loaders(
    data_dir="/path/to/imagenet",
    batch_size=256
)
```

### HuggingFace Datasets

```python
from utils.data import get_huggingface_dataset

# Load dataset
dataset = get_huggingface_dataset(
    "glue",
    "sst2",
    split="train"
)

# With tokenization
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = get_huggingface_dataset(
    "glue",
    "sst2",
    split="train",
    tokenizer=tokenizer,
    max_length=128
)
```

## Fine-Tuning with LoRA

Efficient fine-tuning with LoRA:

```python
from brain import BrainModel
from architectures.lora import add_lora_to_model

# Load pretrained model
model = BrainModel.from_pretrained("bert-base-uncased")

# Add LoRA adapters
model = add_lora_to_model(
    model,
    rank=8,
    alpha=16,
    target_modules=["query", "value"]
)

# Only LoRA parameters are trained
model.train(train_dataset=train_dataset, epochs=3)

# Merge LoRA weights
model.merge_lora_weights()

# Save
model.save("./lora_model")
```

## Distributed Training

Train on multiple GPUs:

```bash
# Using torchrun
torchrun --nproc_per_node=4 examples/training_example.py

# Using Brain CLI
brain train \
  --model gpt-2 \
  --dataset wikitext \
  --distributed \
  --gpus 4
```

## Common Patterns

### Load a Pretrained Model

```python
from brain import BrainModel

# From HuggingFace Hub
model = BrainModel.from_pretrained("bert-base-uncased")

# From local checkpoint
model = BrainModel.from_checkpoint("./my_model/checkpoint.pt")
```

### Custom Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from brain import BrainModel

model = BrainModel("bert-base-uncased")
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(batch['input_ids'])
        loss = criterion(outputs, batch['labels'])

        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
```

### Batch Inference

```python
texts = [
    "This is great!",
    "This is terrible!",
    "This is okay.",
]

predictions = model.predict_batch(texts, batch_size=8)

for text, pred in zip(texts, predictions):
    print(f"'{text}' -> {pred['label']} ({pred['confidence']:.4f})")
```

## Next Steps

Now that you've completed the quickstart:

1. **[Explore Examples](../examples/)** - See more complex examples
2. **[Read the User Guide](user_guide.md)** - Learn all features
3. **[Browse API Docs](api/index.md)** - Detailed API reference
4. **[Join Community](https://github.com/yourusername/Brain/discussions)** - Ask questions

## Need Help?

- **Documentation**: Full docs at [docs/](index.md)
- **Examples**: More examples in [examples/](../examples/)
- **Issues**: Report bugs on [GitHub](https://github.com/yourusername/Brain/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/yourusername/Brain/discussions)

---

**Happy training! 🚀**
