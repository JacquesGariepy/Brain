"""
Brain CLI - Info Command
"""

import sys
import platform


def info_command(args):
    """Display system and framework information"""
    print("=" * 70)
    print("Brain Framework - System Information")
    print("=" * 70)

    # Python info
    print("\nPython:")
    print(f"  Version: {sys.version.split()[0]}")
    print(f"  Implementation: {platform.python_implementation()}")
    print(f"  Executable: {sys.executable}")

    # System info
    print("\nSystem:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Processor: {platform.processor()}")

    # PyTorch info
    print("\nPyTorch:")
    try:
        import torch
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  Status: Not installed")

    # Brain framework
    print("\nBrain Framework:")
    print(f"  Version: 1.0.0")
    print(f"  Components: 46+ SOTA architectures")
    print(f"  Installation: {__file__}")

    # Dependencies
    print("\nKey Dependencies:")
    dependencies = {
        "transformers": "HuggingFace Transformers",
        "datasets": "HuggingFace Datasets",
        "accelerate": "HuggingFace Accelerate",
        "wandb": "Weights & Biases",
        "mlflow": "MLflow",
        "fastapi": "FastAPI",
    }

    for package, name in dependencies.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"  {name}: {version}")
        except ImportError:
            print(f"  {name}: Not installed")

    print("\n" + "=" * 70)
