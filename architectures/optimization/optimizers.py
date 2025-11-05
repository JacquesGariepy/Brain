"""
State-of-the-Art Optimizers

Implementations:
- AdamW (Weight Decay corrected Adam)
- Lion (Evolved Sign Momentum)
- Sophia (Second-order optimizer)
- Sharpness Aware Minimization (SAM)
- AdaFactor
- LAMB (Layer-wise Adaptive Moments)
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Optional, Callable
import math


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    Fixes weight decay in Adam for better generalization.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decoupled weight decay (AdamW)
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class Lion(Optimizer):
    """
    Lion optimizer: Evolved Sign Momentum.

    More memory efficient than Adam, often better performance.
    Discovered through algorithm evolution.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update with sign of interpolated gradient
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.data.add_(torch.sign(update), alpha=-group['lr'])

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class Sophia(Optimizer):
    """
    Sophia: Second-order Clipped Stochastic Optimization.

    Uses Hessian information for better curvature awareness.
    Particularly effective for large language models.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.965, 0.99),
        rho: float = 0.04,  # Clipping threshold
        weight_decay: float = 1e-4,
        eps: float = 1e-12
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            eps=eps
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['hessian'] = torch.zeros_like(p.data)

                exp_avg, hessian = state['exp_avg'], state['hessian']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Hessian diagonal approximation (updated periodically)
                # In practice, this would be computed via Hutchinson's estimator
                # For simplicity, we use gradient magnitude as proxy
                if state['step'] % 10 == 0:
                    hessian.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)

                # Clipped update
                update = exp_avg / (hessian.clamp(min=group['eps']))
                update = torch.clamp(update, -group['rho'], group['rho'])

                p.data.add_(update, alpha=-group['lr'])

        return loss


class SAM(Optimizer):
    """
    Sharpness Aware Minimization.

    Seeks parameters in flat minima for better generalization.
    Requires two forward-backward passes.
    """

    def __init__(
        self,
        params,
        base_optimizer: Optimizer,
        rho: float = 0.05,
        adaptive: bool = False
    ):
        assert rho >= 0.0, f"Invalid rho: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        First step: compute and ascend to worst-case perturbation.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None:
                    continue

                # Save current parameters
                self.state[p]['old_p'] = p.data.clone()

                # Compute perturbation
                e_w = p.grad * scale
                if group['adaptive']:
                    e_w = e_w * p.data.abs()

                # Ascend to worst-case perturbation
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Second step: update with gradient at perturbed point.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Restore original parameters
                p.data = self.state[p]['old_p']

        # Update with base optimizer
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure: Optional[Callable] = None):
        """Not used directly - use first_step and second_step"""
        raise NotImplementedError("SAM requires calling first_step() and second_step()")

    def _grad_norm(self):
        """Compute gradient norm"""
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def zero_grad(self):
        self.base_optimizer.zero_grad()


class AdaFactor(Optimizer):
    """
    AdaFactor: Memory-efficient adaptive learning rate method.

    Reduces memory footprint by factorizing second moments.
    Useful for training very large models.
    """

    def __init__(
        self,
        params,
        lr: Optional[float] = None,
        eps: tuple = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False
    ):
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

    def _get_lr(self, param_group, param_state):
        """Compute learning rate"""
        if param_group['lr'] is None:
            min_step = 1e-6 * param_state['step'] if param_group['warmup_init'] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
            param_group['lr'] = rel_step_sz

        return param_group['lr']

    def _get_options(self, param_group, param_shape):
        """Get factorization options"""
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1']
        return factored, use_first_moment

    def _rms(self, tensor):
        """Root mean square"""
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaFactor does not support sparse gradients')

                state = self.state[p]
                grad_shape = grad.shape

                param_group = group
                factored, use_first_moment = self._get_options(param_group, grad_shape)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    if use_first_moment:
                        state['exp_avg'] = torch.zeros_like(grad)

                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[0])
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[1:]).flatten()
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0

                state['step'] += 1
                lr = self._get_lr(param_group, state)
                group['lr'] = lr

                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = grad

                # Moving average of squared gradient
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=list(range(1, len(grad_shape)))),
                        alpha=1 - beta2t
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=0).flatten(),
                        alpha=1 - beta2t
                    )

                    update = update / (
                        exp_avg_sq_row.view(-1, *([1] * (len(grad_shape) - 1))) *
                        exp_avg_sq_col.view(grad_shape[1:])
                    ).sqrt().add_(group['eps'][0])
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2t).add_(update.pow(2), alpha=1 - beta2t)
                    update = update / exp_avg_sq.sqrt().add_(group['eps'][0])

                # Clipping
                rms = self._rms(update)
                if group['clip_threshold'] > 0:
                    update = update / max(1.0, rms / group['clip_threshold'])

                # First moment
                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(update, alpha=1 - group['beta1'])
                    update = exp_avg

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * lr)

                # Apply update
                p.data.add_(update, alpha=-lr)

        return loss


class LAMB(Optimizer):
    """
    LAMB: Layer-wise Adaptive Moments optimizer for Batch training.

    Enables very large batch training by layer-wise adaptation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        adam: bool = False
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adam=adam)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute Adam step
                step = exp_avg / bias_correction1
                second_moment = exp_avg_sq / bias_correction2

                adam_step = step / (second_moment.sqrt().add_(group['eps']))

                # Weight decay
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                # Layer-wise adaptation
                weight_norm = p.data.pow(2).sum().sqrt()
                adam_norm = adam_step.pow(2).sum().sqrt()

                if weight_norm > 0 and adam_norm > 0 and not group['adam']:
                    trust_ratio = weight_norm / adam_norm
                else:
                    trust_ratio = 1.0

                # Update parameters
                p.data.add_(adam_step, alpha=-group['lr'] * trust_ratio)

        return loss
