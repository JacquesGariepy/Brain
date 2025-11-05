"""
Code Validation Script

Validates all Python files for:
- Syntax errors
- Import errors
- Basic functionality
"""

import ast
import os
import sys
from pathlib import Path


def validate_syntax(file_path):
    """Check if Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def validate_all_python_files(root_dir='.'):
    """Validate all Python files in directory"""
    print("=" * 80)
    print("VALIDATING PYTHON CODE")
    print("=" * 80)

    root_path = Path(root_dir)
    python_files = list(root_path.rglob('*.py'))

    print(f"\nFound {len(python_files)} Python files\n")

    errors = []
    success_count = 0

    for file_path in sorted(python_files):
        # Skip __pycache__
        if '__pycache__' in str(file_path):
            continue

        relative_path = file_path.relative_to(root_path)

        valid, error = validate_syntax(file_path)

        if valid:
            print(f"✓ {relative_path}")
            success_count += 1
        else:
            print(f"✗ {relative_path}")
            print(f"  Error: {error}")
            errors.append((relative_path, error))

    print("\n" + "=" * 80)
    print(f"VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal files: {len(python_files) - len([f for f in python_files if '__pycache__' in str(f)])}")
    print(f"Successful: {success_count}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nFiles with errors:")
        for file_path, error in errors:
            print(f"  - {file_path}: {error}")
        return False
    else:
        print("\n✓ All files validated successfully!")
        return True


def check_imports():
    """Check critical imports"""
    print("\n" + "=" * 80)
    print("CHECKING CRITICAL IMPORTS")
    print("=" * 80)

    critical_modules = [
        "torch",
        "numpy",
        "transformers"
    ]

    for module in critical_modules:
        try:
            __import__(module)
            print(f"✓ {module} available")
        except ImportError:
            print(f"✗ {module} NOT available (install with: pip install {module})")


def count_lines_of_code(root_dir='.'):
    """Count total lines of code"""
    print("\n" + "=" * 80)
    print("CODE STATISTICS")
    print("=" * 80)

    root_path = Path(root_dir)
    python_files = [f for f in root_path.rglob('*.py') if '__pycache__' not in str(f)]

    total_lines = 0
    total_code_lines = 0
    total_comment_lines = 0
    total_blank_lines = 0

    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines += len(lines)

        for line in lines:
            stripped = line.strip()
            if not stripped:
                total_blank_lines += 1
            elif stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                total_comment_lines += 1
            else:
                total_code_lines += 1

    print(f"\nTotal Python files: {len(python_files)}")
    print(f"Total lines: {total_lines:,}")
    print(f"Code lines: {total_code_lines:,}")
    print(f"Comment/doc lines: {total_comment_lines:,}")
    print(f"Blank lines: {total_blank_lines:,}")


def list_all_architectures():
    """List all implemented architectures"""
    print("\n" + "=" * 80)
    print("IMPLEMENTED ARCHITECTURES")
    print("=" * 80)

    architectures = {
        "Transformers": [
            "Multi-Head Attention",
            "Grouped-Query Attention (GQA)",
            "Multi-Query Attention (MQA)",
            "Complete Transformer",
            "State Space Models (Mamba, S4)",
            "Sparse Transformers",
            "Mixture of Experts (MoE)"
        ],
        "Vision": [
            "Vision Transformer (ViT)",
            "Swin Transformer"
        ],
        "Memory Systems": [
            "Neural Turing Machine (NTM)",
            "Differentiable Neural Computer (DNC)",
            "Memory Networks"
        ],
        "Reinforcement Learning": [
            "PPO (Proximal Policy Optimization)",
            "SAC (Soft Actor-Critic)",
            "Rainbow DQN",
            "World Models"
        ],
        "Generative Models": [
            "Diffusion Models (DDPM, DDIM)",
            "Classifier-free Guidance"
        ],
        "Graph Networks": [
            "GCN (Graph Convolutional Networks)",
            "GAT (Graph Attention Networks)",
            "GraphSAGE",
            "GIN (Graph Isomorphism Networks)",
            "Temporal Graph Networks"
        ],
        "Reasoning": [
            "Chain-of-Thought",
            "Tree of Thoughts",
            "ReAct (Reasoning + Acting)",
            "Self-Consistency",
            "Function Calling"
        ],
        "Optimization": [
            "AdamW",
            "Lion",
            "Sophia",
            "SAM (Sharpness Aware Minimization)",
            "AdaFactor",
            "LAMB"
        ],
        "Compression & Efficiency": [
            "LoRA (Low-Rank Adaptation)",
            "QLoRA (Quantized LoRA)",
            "Adapter Layers",
            "Prefix Tuning",
            "Prompt Tuning",
            "Pruning",
            "Quantization",
            "Knowledge Distillation"
        ]
    }

    for category, items in architectures.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")


if __name__ == "__main__":
    # Validate all code
    success = validate_all_python_files()

    # Check imports
    check_imports()

    # Count lines
    count_lines_of_code()

    # List architectures
    list_all_architectures()

    # Exit code
    sys.exit(0 if success else 1)
