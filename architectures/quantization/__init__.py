"""
Quantization - Model Compression for Efficient Inference

Reduces model size and memory by 4-8x with minimal quality loss.
"""

from .quantization import (
    # Base
    QuantizationType,
    QuantizationConfig,
    QuantizedLinear,

    # GPTQ
    GPTQConfig,
    GPTQLinear,

    # AWQ
    AWQConfig,
    AWQLinear,

    # bitsandbytes
    BitsAndBytesConfig,
    Int8Linear,
    NF4Quantizer,

    # GGML/GGUF
    GGMLConfig,
    GGMLQuantizer
)

__all__ = [
    # Base
    'QuantizationType',
    'QuantizationConfig',
    'QuantizedLinear',

    # GPTQ
    'GPTQConfig',
    'GPTQLinear',

    # AWQ
    'AWQConfig',
    'AWQLinear',

    # bitsandbytes
    'BitsAndBytesConfig',
    'Int8Linear',
    'NF4Quantizer',

    # GGML/GGUF
    'GGMLConfig',
    'GGMLQuantizer'
]
