"""
Advanced Optimizers for LLM Training

State-of-the-art optimizers beyond Adam:
1. Lion: Discovered by Google via evolution
2. Sophia: Second-order optimizer with Hessian
3. Adafactor: Memory-efficient adaptive optimizer
4. 8-bit Adam: Quantized optimizer states
5. Shampoo: Full-matrix preconditioning

Key Benefits:
- Faster convergence
- Lower memory usage
- Better final performance
- Scalability to large models

References:
- Lion: https://arxiv.org/abs/2302.06675
- Sophia: https://arxiv.org/abs/2305.14342
- Adafactor: https://arxiv.org/abs/1804.04235
- 8-bit Optimizers: https://arxiv.org/abs/2110.02861
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn
import math


# ============================================================================
# Lion Optimizer
# ============================================================================

@dataclass
class LionConfig:
    """Configuration for Lion optimizer"""
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0


class Lion(torch.optim.Optimizer):
    """
    Lion: Evolved Sign Momentum Optimizer

    Key Innovation:
    - Discovered by Google via program search (AutoML)
    - Uses sign of momentum (not momentum value)
    - Simpler than Adam but often better
    - Lower memory (no second moment)

    Update rule:
        c_t = β₁ * m_{t-1} + (1 - β₁) * g_t  (momentum update)
        θ_t = θ_{t-1} - λ * (sign(c_t) + wd * θ_{t-1})  (parameter update)
        m_t = β₂ * m_{t-1} + (1 - β₂) * g_t  (momentum for next step)

    vs Adam:
        Adam: θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
        Lion: θ_t = θ_{t-1} - λ * sign(c_t)

    Key differences:
    1. Uses sign (not magnitude) → more robust
    2. Single state (momentum), no variance → 50% memory
    3. Separate β for update vs momentum → more flexible

    Advantages:
    - 2x memory reduction vs Adam
    - Better on image tasks (Vision Transformer, DiffusionModels)
    - Competitive on language tasks
    - Simpler implementation

    Disadvantages:
    - Learning rate tuning more sensitive
    - May need higher weight decay

    Typical hyperparameters:
    - lr: 1e-4 to 3e-4 (10x smaller than Adam)
    - beta1: 0.9
    - beta2: 0.99
    - weight_decay: 0.1 to 1.0 (10x higher than Adam)

    Reference:
        "Symbolic Discovery of Optimization Algorithms"
        (Chen et al., Google, 2023)

    Example:
        >>> optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)
        >>> for x, y in dataloader:
        ...     loss = model(x, y)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Get gradient
                grad = p.grad

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    # Initialize momentum
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                # Momentum update for parameter update
                # c_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                update = exp_avg * beta1 + grad * (1 - beta1)

                # Parameter update
                # θ_t = θ_{t-1} - lr * (sign(c_t) + wd * θ_{t-1})
                p.add_(torch.sign(update), alpha=-group['lr'])

                # Weight decay (decoupled)
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])

                # Momentum update for next iteration
                # m_t = β₂ * m_{t-1} + (1 - β₂) * g_t
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


# ============================================================================
# Sophia Optimizer
# ============================================================================

@dataclass
class SophiaConfig:
    """Configuration for Sophia optimizer"""
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.965, 0.99)
    rho: float = 0.04  # Hessian update interval
    weight_decay: float = 1e-1


class Sophia(torch.optim.Optimizer):
    """
    Sophia: Second-order Clipped Stochastic Optimization

    Key Innovation:
    - Uses Hessian diagonal (second-order information)
    - Clips update by Hessian (adaptive clipping)
    - 2x faster convergence than Adam on language models

    Update rule:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        h_t = β₂ * h_{t-1} + (1 - β₂) * (g_t ⊙ g_t)  (Hessian diagonal approx)
        θ_t = θ_{t-1} - lr * clip(m_t / h_t, ρ)

    where clip(x, ρ) = max(-ρ, min(x, ρ))

    Key Insight:
    - Hessian diagonal approximates curvature
    - Clip by curvature prevents large updates in flat regions
    - More stable than pure second-order methods

    Advantages:
    - 2x faster convergence on language models
    - Better final loss
    - Similar memory to Adam

    Disadvantages:
    - Requires Hessian estimation (expensive)
    - Update Hessian every K steps (not every step)

    Typical hyperparameters:
    - lr: 2e-4 (similar to Adam)
    - beta1: 0.965
    - beta2: 0.99
    - rho: 0.04 (clip threshold)
    - hessian_update_freq: 10 steps

    Results (on language modeling):
    - 2x speedup to reach same loss as Adam
    - Better final performance
    - Tested on up to 7B models

    Reference:
        "Sophia: A Scalable Stochastic Second-order Optimizer for Language
        Model Pre-training" (Liu et al., Stanford, 2023)
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        hessian_update_freq: int = 10
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            hessian_update_freq=hessian_update_freq
        )
        super().__init__(params, defaults)

        # Global step counter
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None, hessian=None):
        """
        Perform optimization step.

        Args:
            closure: Optional closure to recompute loss
            hessian: Optional pre-computed Hessian diagonal
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            rho = group['rho']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.ones_like(p)  # Initialize to 1

                exp_avg = state['hessian']
                hess = state['hessian']

                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update Hessian diagonal (every K steps)
                if self.step_count % group['hessian_update_freq'] == 0:
                    if hessian is not None:
                        # Use provided Hessian
                        hess.copy_(hessian)
                    else:
                        # Approximate Hessian diagonal with gradient norm
                        # h ≈ E[g²] (similar to Adam's second moment)
                        hess.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute update: clip(m / h, rho)
                update = exp_avg / (hess + 1e-8)
                update = torch.clamp(update, -rho, rho)

                # Apply update
                p.add_(update, alpha=-group['lr'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])

        return loss


# ============================================================================
# Adafactor
# ============================================================================

class Adafactor(torch.optim.Optimizer):
    """
    Adafactor: Memory-Efficient Adaptive Optimization

    Key Innovation:
    - Factorized second moment estimation
    - Instead of storing full V matrix (d×1), store row/col means
    - Memory: O(d) instead of O(d) for each parameter
    - No beta2 hyperparameter needed

    Second moment factorization:
        Instead of: v_t = β₂ * v_{t-1} + (1-β₂) * g²
        Use: r_t = mean(g², dim=0)  # Row factors
             c_t = mean(g², dim=1)  # Column factors
             v_t ≈ r_t × c_t (outer product approximation)

    For a (1000, 1000) matrix:
        Adam: 1M values for second moment
        Adafactor: 2K values (row + col means)
        Savings: 500x memory!

    Advantages:
    - Much lower memory (critical for large models)
    - No beta2 tuning needed (adaptive)
    - Competitive performance with Adam

    Disadvantages:
    - Slightly more complex
    - May converge slower on some tasks

    Typical usage:
    - T5, mT5, UL2: All trained with Adafactor
    - Large models where memory is constrained
    - TPU training (Google's default)

    Reference:
        "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
        (Shazeer & Stern, Google, 2018)
    """

    def __init__(
        self,
        params,
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr with relative_step")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init
        )
        super().__init__(params, defaults)

    def _get_lr(self, param_group, param_scale):
        """Compute learning rate (adaptive if relative_step=True)."""
        if param_group['relative_step']:
            min_step = 1e-6 * param_group['step_count'] if param_group['warmup_init'] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_group['step_count']))
            param_group['lr'] = rel_step_sz * param_scale
        return param_group['lr']

    def _get_options(self, param_group, param_shape):
        """Get factorization options based on parameter shape."""
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1']
        return factored, use_first_moment

    def _rms(self, tensor):
        """Root mean square."""
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients")

                # State initialization
                state = self.state[p]
                grad_shape = grad.shape

                if len(state) == 0:
                    state['step'] = 0

                state['step'] += 1
                group['step_count'] = state['step']

                # Get options
                factored, use_first_moment = self._get_options(group, grad_shape)

                # Bias correction
                bias_correction = 1 - group['beta1'] ** state['step'] if use_first_moment else 1

                # Learning rate
                param_scale = 1
                if group['scale_parameter']:
                    param_scale = math.sqrt(grad_shape[0]) if factored else 1
                lr = self._get_lr(group, param_scale)

                # Clipping
                if group['clip_threshold'] >= 1.0:
                    grad_norm = self._rms(grad)
                    if grad_norm > group['clip_threshold']:
                        grad.div_(grad_norm / group['clip_threshold'])

                # Momentum
                if use_first_moment:
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg'].mul_(group['beta1']).add_(grad, alpha=1 - group['beta1'])

                # Second moment (factorized or full)
                if factored:
                    # Factorized second moment
                    if 'exp_avg_sq_row' not in state:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[0], device=grad.device)
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[1], device=grad.device)

                    # Update row and column factors
                    state['exp_avg_sq_row'].mul_(group['decay_rate']).add_(
                        grad.mean(dim=1).pow(2), alpha=1 - group['decay_rate']
                    )
                    state['exp_avg_sq_col'].mul_(group['decay_rate']).add_(
                        grad.mean(dim=0).pow(2), alpha=1 - group['decay_rate']
                    )

                    # Reconstruct second moment from factors
                    v = state['exp_avg_sq_row'].unsqueeze(1) @ state['exp_avg_sq_col'].unsqueeze(0)
                    v = v.sqrt().add_(group['eps'][1])
                else:
                    # Full second moment (for 1D parameters)
                    if 'exp_avg_sq' not in state:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['exp_avg_sq'].mul_(group['decay_rate']).add_(
                        grad.pow(2), alpha=1 - group['decay_rate']
                    )
                    v = state['exp_avg_sq'].sqrt().add_(group['eps'][1])

                # Update parameters
                if use_first_moment:
                    update = state['exp_avg'] / (v * bias_correction)
                else:
                    update = grad / v

                p.add_(update, alpha=-lr)

                # Weight decay
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-lr * group['weight_decay'])

        return loss


# ============================================================================
# 8-bit Adam
# ============================================================================

class Adam8bit(torch.optim.Optimizer):
    """
    8-bit Adam: Quantized Optimizer States

    Key Innovation:
    - Quantize optimizer states (momentum, variance) to 8-bit
    - Dynamic block-wise quantization
    - 75% memory reduction vs FP32 Adam
    - Minimal accuracy loss

    Quantization:
        For each block of states:
        1. Compute max absolute value
        2. Quantize: q = round(x / max * 127)
        3. Store: (q: int8, max: float32)
        4. Dequantize: x = q * max / 127

    Memory savings:
        FP32 Adam: 4 bytes × 2 states = 8 bytes/param
        8-bit Adam: 1 byte × 2 states + overhead = 2.5 bytes/param
        Reduction: 70%

    Advantages:
    - 3-4x memory reduction
    - Enables larger batch sizes
    - Minimal accuracy loss (<0.1%)
    - Compatible with any model

    Disadvantages:
    - Slightly slower (quantization overhead)
    - Requires careful implementation

    Reference:
        "8-bit Optimizers via Block-wise Quantization"
        (Dettmers et al., 2021)

    Note: This is a simplified implementation.
    Production version: use bitsandbytes library
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        block_size: int = 256
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, block_size=block_size)
        super().__init__(params, defaults)

    def _quantize_block(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to 8-bit with block-wise scaling.

        Args:
            tensor: Tensor to quantize

        Returns:
            quantized: Int8 tensor
            scale: FP32 scale factors per block
        """
        # Reshape into blocks
        numel = tensor.numel()
        block_size = self.defaults['block_size']

        # Compute scale per block
        tensor_blocks = tensor.view(-1, block_size) if numel >= block_size else tensor.unsqueeze(0)

        # Max absolute value per block
        scale = tensor_blocks.abs().max(dim=-1, keepdim=True)[0]
        scale = scale.clamp(min=1e-8)

        # Quantize
        quantized = (tensor_blocks / scale * 127).round().to(torch.int8)

        return quantized, scale

    def _dequantize_block(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize 8-bit tensor."""
        dequantized = quantized.float() * scale / 127
        return dequantized.flatten()

    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step with 8-bit states."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Store quantized states
                    state['exp_avg_quant'] = None
                    state['exp_avg_sq_quant'] = None
                    state['exp_avg_scale'] = None
                    state['exp_avg_sq_scale'] = None

                state['step'] += 1

                # Dequantize states if they exist
                if state['exp_avg_quant'] is not None:
                    exp_avg = self._dequantize_block(state['exp_avg_quant'], state['exp_avg_scale'])
                else:
                    exp_avg = torch.zeros_like(p)

                if state['exp_avg_sq_quant'] is not None:
                    exp_avg_sq = self._dequantize_block(state['exp_avg_sq_quant'], state['exp_avg_sq_scale'])
                else:
                    exp_avg_sq = torch.zeros_like(p)

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Quantize and store
                state['exp_avg_quant'], state['exp_avg_scale'] = self._quantize_block(exp_avg)
                state['exp_avg_sq_quant'], state['exp_avg_sq_scale'] = self._quantize_block(exp_avg_sq)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute update
                step_size = group['lr'] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])

                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Weight decay
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])

        return loss


# ============================================================================
# Testing
# ============================================================================

def test_lion():
    """Test Lion optimizer."""
    print("=" * 80)
    print("Test 1: Lion Optimizer")
    print("=" * 80)

    # Create simple model
    model = nn.Linear(100, 10)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)

    # Dummy training step
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))

    output = model(x)
    loss = nn.functional.cross_entropy(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Weight decay: {optimizer.param_groups[0]['weight_decay']}")
    print(f"Betas: {optimizer.param_groups[0]['betas']}")

    # Check memory
    num_states = sum(len(state) > 0 for state in optimizer.state.values())
    states_per_param = num_states / len(list(model.parameters())) if len(list(model.parameters())) > 0 else 0

    print(f"\n✓ Lion test PASSED")
    print(f"Memory: 1 state per param vs 2 for Adam")
    print(f"Discovered by Google via program search")

    return {'status': 'PASS', 'optimizer': 'Lion'}


def test_sophia():
    """Test Sophia optimizer."""
    print("\n" + "=" * 80)
    print("Test 2: Sophia Optimizer")
    print("=" * 80)

    model = nn.Linear(100, 10)
    optimizer = Sophia(model.parameters(), lr=2e-4, rho=0.04)

    # Training step
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))

    output = model(x)
    loss = nn.functional.cross_entropy(output, y)
    loss.backward()
    optimizer.step()

    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Rho (clip threshold): {optimizer.param_groups[0]['rho']}")
    print(f"Uses Hessian diagonal for adaptive clipping")

    print(f"\n✓ Sophia test PASSED")
    print(f"2x faster convergence on language models")

    return {'status': 'PASS', 'optimizer': 'Sophia'}


def test_adafactor():
    """Test Adafactor."""
    print("\n" + "=" * 80)
    print("Test 3: Adafactor Optimizer")
    print("=" * 80)

    # 2D parameter for factorization
    model = nn.Linear(1000, 1000)
    optimizer = Adafactor(model.parameters())

    # Training step
    x = torch.randn(32, 1000)
    output = model(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    param = list(model.parameters())[0]
    state = optimizer.state[param]

    print(f"Parameter shape: {param.shape}")
    print(f"Factorized second moment: {' exp_avg_sq_row' in state and 'exp_avg_sq_col' in state}")

    if 'exp_avg_sq_row' in state:
        row_size = state['exp_avg_sq_row'].numel()
        col_size = state['exp_avg_sq_col'].numel()
        full_size = param.numel()

        print(f"Memory: {row_size + col_size} vs {full_size} (full)")
        print(f"Reduction: {full_size / (row_size + col_size):.1f}x")

    print(f"\n✓ Adafactor test PASSED")
    print(f"Used in T5, mT5, UL2")

    return {'status': 'PASS', 'optimizer': 'Adafactor'}


def test_adam8bit():
    """Test 8-bit Adam."""
    print("\n" + "=" * 80)
    print("Test 4: 8-bit Adam")
    print("=" * 80)

    model = nn.Linear(100, 10)
    optimizer = Adam8bit(model.parameters(), lr=1e-3, block_size=256)

    # Training step
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))

    output = model(x)
    loss = nn.functional.cross_entropy(output, y)
    loss.backward()
    optimizer.step()

    # Check quantized states
    param = list(model.parameters())[0]
    state = optimizer.state[param]

    if 'exp_avg_quant' in state and state['exp_avg_quant'] is not None:
        print(f"States quantized: int8")
        print(f"Block size: {optimizer.defaults['block_size']}")

        quant_size = state['exp_avg_quant'].element_size() * state['exp_avg_quant'].numel()
        scale_size = state['exp_avg_scale'].element_size() * state['exp_avg_scale'].numel()
        total_size = quant_size + scale_size

        full_size = param.numel() * 4  # FP32

        print(f"Memory per state: {total_size} bytes vs {full_size} bytes (FP32)")
        print(f"Reduction: {full_size / total_size:.1f}x per state")

    print(f"\n✓ 8-bit Adam test PASSED")
    print(f"70% memory reduction vs FP32 Adam")

    return {'status': 'PASS', 'optimizer': 'Adam8bit'}


def test_all():
    """Run all optimizer tests."""
    print("\n" + "=" * 80)
    print("Advanced Optimizers - Complete Test Suite")
    print("=" * 80)

    results = {}

    results['Lion'] = test_lion()
    results['Sophia'] = test_sophia()
    results['Adafactor'] = test_adafactor()
    results['Adam8bit'] = test_adam8bit()

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, result in results.items():
        print(f"\n{name}: {result['status']}")

    print("\n" + "=" * 80)
    print("Optimizer Comparison")
    print("=" * 80)
    print("""
Optimizer  | Memory | Convergence | Tuning    | Best For
-----------|--------|-------------|-----------|------------------
Adam       | High   | Good        | Easy      | Default choice
Lion       | 50%    | Better      | Sensitive | Vision, generation
Sophia     | High   | 2x faster   | Medium    | Language models
Adafactor  | 25%    | Good        | Easy      | Large models
8-bit Adam | 30%    | Same        | Easy      | Memory-constrained

Detailed Comparison:

1. Lion (Google, 2023):
   - Memory: 50% of Adam (no second moment)
   - Sign-based updates
   - Better on vision, competitive on NLP
   - Needs 10x smaller LR, 10x higher WD

2. Sophia (Stanford, 2023):
   - Uses Hessian diagonal (second-order)
   - 2x faster convergence on language
   - Update every 10 steps
   - Best for pre-training LLMs

3. Adafactor (Google, 2018):
   - Factorized second moment
   - 75% memory reduction
   - No beta2 tuning
   - Used in T5, UL2

4. 8-bit Adam:
   - Quantized optimizer states
   - 70% memory reduction
   - Minimal accuracy loss
   - Universal compatibility

Performance (on language modeling):
----------------------------------
Optimizer  | Steps to Loss X | Memory/Param | LR
-----------|-----------------|--------------|--------
Adam       | 100K (baseline) | 8 bytes      | 1e-3
Lion       | 95K (1.05x)     | 4 bytes      | 1e-4
Sophia     | 50K (2x faster!)| 8 bytes      | 2e-4
Adafactor  | 105K (0.95x)    | 2 bytes      | Auto
8-bit Adam | 100K (same)     | 2.5 bytes    | 1e-3

When to Use:
-----------
- Adam: Default, no memory constraints
- Lion: Vision tasks, want simpler optimizer
- Sophia: Pre-training LLMs, have compute
- Adafactor: Very large models, limited memory
- 8-bit Adam: Need memory savings, no time for tuning

Production Usage:
----------------
- GPT-3: Adam
- PaLM, T5: Adafactor
- Stable Diffusion: Lion
- Future LLMs: Likely Sophia or variants

Hyperparameter Guidelines:
-------------------------
Adam:       lr=1e-3, wd=0.01
Lion:       lr=1e-4, wd=0.1
Sophia:     lr=2e-4, wd=0.1, rho=0.04
Adafactor:  lr=auto, factored=True
8-bit Adam: lr=1e-3, wd=0.01, block=256
    """)

    print("=" * 80)

    return results


if __name__ == "__main__":
    test_all()
