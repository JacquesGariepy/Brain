# Installation Guide

This guide will help you install Brain Framework in various environments.

## Quick Install

### From PyPI (Recommended)

```bash
# Basic installation
pip install brain-framework

# Full installation with all features
pip install brain-framework[full]
```

### From Source

```bash
# Clone repository
git clone https://github.com/yourusername/Brain.git
cd Brain

# Install in editable mode
pip install -e .

# Or with all dependencies
pip install -e ".[full]"
```

## Installation Options

### Minimal Installation

For basic usage with core features:

```bash
pip install brain-framework
```

This includes:
- Core framework
- PyTorch
- NumPy
- Basic utilities

### Custom Installation

Install only what you need:

```bash
# For data handling
pip install brain-framework[data]

# For model architectures
pip install brain-framework[models]

# For training
pip install brain-framework[training]

# For monitoring
pip install brain-framework[monitoring]

# For API serving
pip install brain-framework[api]

# For scientific computing
pip install brain-framework[scientific]

# For development
pip install brain-framework[dev]
```

### Full Installation

For all features:

```bash
pip install brain-framework[full]
```

This includes:
- All model architectures
- Data loading and preprocessing
- Training utilities (DeepSpeed, Accelerate)
- Monitoring (WandB, MLflow, TensorBoard)
- API serving (FastAPI)
- Scientific computing tools
- All dependencies

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **Disk Space**: 5 GB
- **OS**: Linux, macOS, Windows

### Recommended Requirements

- **Python**: 3.10 or higher
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU with 16+ GB VRAM
- **CUDA**: 11.8 or higher
- **Disk Space**: 50 GB (for models and datasets)

## GPU Support

### CUDA Setup

For NVIDIA GPUs:

```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install Brain
pip install brain-framework[full]
```

Verify CUDA installation:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Apple Silicon (MPS)

For M1/M2 Macs:

```bash
# Install with MPS support
pip install torch torchvision torchaudio
pip install brain-framework[full]
```

Verify MPS:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### CPU Only

For CPU-only environments:

```bash
# Install PyTorch CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install Brain
pip install brain-framework
```

## Docker Installation

### Using Pre-built Images

```bash
# Pull image (GPU)
docker pull brain-framework:latest

# Pull image (CPU)
docker pull brain-framework:cpu

# Run container
docker run -it --gpus all -p 8000:8000 brain-framework:latest
```

### Build from Dockerfile

```bash
# Build GPU image
docker build -t brain-framework:latest .

# Build CPU image
docker build -f Dockerfile.cpu -t brain-framework:cpu .
```

### Docker Compose

```bash
# Start all services (API, TensorBoard, MLflow)
docker-compose up -d

# Start CPU-only version
docker-compose --profile cpu up -d

# Start with Jupyter
docker-compose --profile dev up -d
```

## Verify Installation

After installation, verify everything works:

```python
# Test import
import brain
print(f"Brain Framework version: {brain.__version__}")

# Test PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test Brain utilities
from utils.data import get_mnist_loaders
from utils.logging import get_logger

logger = get_logger()
logger.info("Brain Framework installed successfully!")
```

Or use the CLI:

```bash
# Check system info
brain info

# List available models
brain list

# Test with quickstart
brain train --model bert-base-uncased --dataset glue/sst2 --epochs 1
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Use smaller batch size
brain train --batch-size 16

# Enable gradient checkpointing
brain train --gradient-checkpointing
```

#### 2. Import Errors

```bash
# Reinstall with all dependencies
pip uninstall brain-framework
pip install brain-framework[full]
```

#### 3. Permission Errors

```bash
# Install for user only
pip install --user brain-framework
```

#### 4. Version Conflicts

```bash
# Create fresh virtual environment
python -m venv brain_env
source brain_env/bin/activate  # On Windows: brain_env\Scripts\activate
pip install brain-framework[full]
```

### Getting Help

If you encounter issues:

1. Check the [FAQ](faq.md)
2. Search [GitHub Issues](https://github.com/yourusername/Brain/issues)
3. Ask on [Discussions](https://github.com/yourusername/Brain/discussions)
4. Report a bug with `brain info` output

## Development Installation

For contributing to Brain:

```bash
# Clone repository
git clone https://github.com/yourusername/Brain.git
cd Brain

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Updating Brain

### Update from PyPI

```bash
# Update to latest version
pip install --upgrade brain-framework

# Update specific component
pip install --upgrade brain-framework[monitoring]
```

### Update from Source

```bash
cd Brain
git pull
pip install -e ".[full]"
```

## Uninstallation

```bash
# Uninstall Brain
pip uninstall brain-framework

# Remove cache
rm -rf ~/.cache/brain

# Remove data directory (optional)
rm -rf ./data
```

## Next Steps

Now that Brain is installed:

1. **[Try the Quickstart](quickstart.md)** - Train your first model
2. **[Explore Examples](../examples/)** - See practical examples
3. **[Read the User Guide](user_guide.md)** - Learn all features

---

**Need help?** Check our [troubleshooting guide](troubleshooting.md) or open an [issue](https://github.com/yourusername/Brain/issues).
