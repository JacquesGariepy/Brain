"""
Evaluation Framework - Comprehensive LLM Benchmarking

Implements all major LLM evaluation benchmarks.
"""

from .eval_framework import (
    # Base
    TaskType,
    Example,
    BenchmarkResult,
    BaseBenchmark,

    # MMLU
    MMLUConfig,
    MMLU,

    # Other benchmarks
    HellaSwag,
    TruthfulQA,
    GSM8K,
    HumanEval,

    # Suite
    EvaluationSuite
)

__all__ = [
    # Base
    'TaskType',
    'Example',
    'BenchmarkResult',
    'BaseBenchmark',

    # MMLU
    'MMLUConfig',
    'MMLU',

    # Other benchmarks
    'HellaSwag',
    'TruthfulQA',
    'GSM8K',
    'HumanEval',

    # Suite
    'EvaluationSuite'
]
