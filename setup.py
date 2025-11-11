"""
Brain Framework Setup

Installation script for the Brain framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="brain-framework",
    version="1.0.0",
    author="Brain Team",
    author_email="brain@example.com",
    description="Comprehensive AI framework with 46+ SOTA architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Brain",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/Brain/issues",
        "Documentation": "https://github.com/yourusername/Brain/docs",
        "Source Code": "https://github.com/yourusername/Brain",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies (minimal)
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        # Full installation with all features
        "full": requirements,

        # Data handling
        "data": [
            "datasets>=2.0.0",
            "torchvision>=0.13.0",
            "torchaudio>=0.12.0",
            "Pillow>=9.0.0",
        ],

        # Model architectures
        "models": [
            "transformers>=4.20.0",
            "timm>=0.6.0",
            "einops>=0.6.0",
        ],

        # Training
        "training": [
            "accelerate>=0.20.0",
            "deepspeed>=0.9.0",
            "lightning>=2.0.0",
        ],

        # Monitoring
        "monitoring": [
            "wandb>=0.13.0",
            "tensorboard>=2.10.0",
            "mlflow>=2.0.0",
        ],

        # API
        "api": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "pydantic>=2.0.0",
        ],

        # Scientific
        "scientific": [
            "scipy>=1.9.0",
            "scikit-learn>=1.1.0",
            "matplotlib>=3.5.0",
        ],

        # Development
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "brain=cli.main:main",
            "brain-train=cli.commands.train:main",
            "brain-eval=cli.commands.evaluate:main",
            "brain-serve=cli.commands.serve:main",
        ],
    },
    include_package_data=True,
    package_data={
        "brain": ["py.typed"],
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "transformers",
        "neural networks",
        "computer vision",
        "natural language processing",
        "multimodal",
        "AGI",
        "SOTA",
    ],
)
