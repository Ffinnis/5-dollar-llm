"""
Adafactor Optimizer

Memory-efficient optimizer that uses factored second moments.
Instead of storing full V ∈ R^(n×m), stores only:
- Row means R ∈ R^n
- Column means C ∈ R^m
Reducing memory from O(nm) to O(n+m).

Reference: "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
https://arxiv.org/abs/1804.04235
"""

import torch
from torch.optim import Optimizer
import math
from typing import Optional, Tuple


class Adafactor(Optimizer):
    """
    Adafactor optimizer with memory-efficient factorized second moments.
    
    Key features:
    - Factorized second moment for matrices (O(n+m) vs O(nm))
    - Update clipping for stability
    - Optional relative step size
    - Optional first moment (momentum)
    
    Args:
        params: Parameters to optimize
        lr: Learning rate. If None, uses relative step size.
        eps: Tuple of (eps1, eps2) for numerical stability
        clip_threshold: Threshold for update clipping
        decay_rate: For computing beta2 = 1 - t^decay_rate
        beta1: First moment coefficient. None = no momentum (memory efficient)
        weight_decay: L2 regularization coefficient
        scale_parameter: If True, scale lr by RMS of parameters
        relative_step: If True, compute lr from parameter scale
        warmup_init: If True, use warmup for relative step
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
        warmup_init: bool = False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot use both lr and relative_step=True")
        if lr is None and not relative_step:
            raise ValueError("Must specify lr when relative_step=False")
            
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)
    
    @staticmethod
    def _get_lr(group, state):
        """Compute learning rate based on relative step or fixed lr."""
        rel_step_sz = group['lr']
        
        if group['relative_step']:
            # Compute relative step size
            min_step = 1e-6 if group['warmup_init'] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(state['step']))
            
        if group['scale_parameter']:
            # Scale by RMS of parameters
            param_scale = max(group['eps'][1], state['RMS'])
            rel_step_sz = rel_step_sz * param_scale
            
        return rel_step_sz
    
    @staticmethod
    def _get_beta2(step, decay_rate):
        """Compute beta2 for current step: 1 - t^decay_rate."""
        return 1.0 - math.pow(step, decay_rate)
    
    @staticmethod
    def _rms(tensor):
        """Compute root mean square of tensor."""
        return tensor.pow(2).mean().sqrt()
    
    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        """
        Approximate squared gradient from factorized moments.
        V_hat = (R * C) / sum(R)
        Where * is outer product.
        """
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean()).unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(0)
        return r_factor * c_factor

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
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
                
                state = self.state[p]
                grad_shape = grad.shape
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['RMS'] = 0.0
                    
                    # Factorized second moment for matrices (2D+)
                    if len(grad_shape) >= 2:
                        # For matrices, store row and column averages
                        state['exp_avg_sq_row'] = torch.zeros(
                            grad_shape[:-1], dtype=grad.dtype, device=grad.device
                        )
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad_shape[-1], dtype=grad.dtype, device=grad.device
                        )
                    else:
                        # For vectors, store full second moment
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    
                    # Optional first moment (momentum)
                    if group['beta1'] is not None:
                        state['exp_avg'] = torch.zeros_like(grad)
                
                state['step'] += 1
                state['RMS'] = self._rms(p.data).item()
                
                # Get learning rate
                lr = self._get_lr(group, state)
                
                # Get beta2 (time-dependent)
                beta2 = self._get_beta2(state['step'], group['decay_rate'])
                
                # Compute gradient squared with epsilon for stability
                eps1 = group['eps'][0]
                eps2 = group['eps'][1]
                grad_sqr = grad.pow(2).add_(eps1)
                
                # Update factorized second moment
                if len(grad_shape) >= 2:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    
                    # Compute row and column sums
                    row_mean = grad_sqr.mean(dim=-1)
                    col_mean = grad_sqr.mean(dim=tuple(range(grad.dim() - 1)))
                    
                    # Update EMAs
                    exp_avg_sq_row.mul_(beta2).add_(row_mean, alpha=1 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(col_mean, alpha=1 - beta2)
                    
                    # Approximate full second moment
                    v_hat = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    v_hat = v_hat.add_(eps1)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).add_(grad_sqr, alpha=1 - beta2)
                    v_hat = exp_avg_sq
                
                # Compute normalized update: U = G / sqrt(V_hat)
                update = grad / v_hat.sqrt()
                
                # Update clipping (not gradient clipping!)
                # U_hat = U / max(1, RMS(U) / clip_threshold)
                if group['clip_threshold'] > 0:
                    update_rms = self._rms(update)
                    clip_denom = max(1.0, update_rms.item() / group['clip_threshold'])
                    update.div_(clip_denom)
                
                # Apply first moment (momentum) if enabled
                if group['beta1'] is not None:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(update, alpha=1 - group['beta1'])
                    update = exp_avg
                
                # Apply weight decay (decoupled)
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - lr * group['weight_decay'])
                
                # Apply update
                p.data.add_(update, alpha=-lr)
        
        return loss
