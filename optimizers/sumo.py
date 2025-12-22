"""
SUMO: Subspace-Aware Moment-Orthogonalization Optimizer

Based on the paper: "SUMO: Subspace-Aware Moment-Orthogonalization for 
Accelerating Memory-Efficient LLM Training"

Key features:
- Exact SVD-based orthogonalization (instead of Newton-Schulz approximation)
- Low-rank subspace optimization for memory efficiency  
- Norm-growth limiter for stability
- Perpendicular gradient term for better convergence
"""

import torch
import torch.nn.functional as F
from typing import Optional


def randomized_svd(A: torch.Tensor, rank: int, n_oversamples: int = 5, n_iter: int = 2) -> torch.Tensor:
    """
    Randomized SVD to efficiently compute top-r left singular vectors.
    
    Returns Q: orthonormal basis for the column space of A (m x r).
    """
    m, n = A.shape
    r = min(rank + n_oversamples, min(m, n))
    
    # Random projection
    Omega = torch.randn(n, r, device=A.device, dtype=A.dtype)
    Y = A @ Omega  # m x r
    
    # Power iteration for better approximation
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
    
    # QR decomposition to get orthonormal basis
    Q, _ = torch.linalg.qr(Y)
    
    return Q[:, :rank]


def orth_svd(M: torch.Tensor) -> torch.Tensor:
    """
    Exact orthogonalization via SVD.
    Returns U @ V^T where M = U @ Σ @ V^T
    """
    # Normalize like Newton-Schulz does
    M_norm = M / (M.norm() + 1e-7)
    U, S, Vt = torch.linalg.svd(M_norm, full_matrices=False)
    return U @ Vt


class SUMO(torch.optim.Optimizer):
    """
    SUMO: Subspace-Aware Moment-Orthogonalization Optimizer
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient β (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        rank: Low-rank subspace dimension r (default: 16)
        subspace_update_freq: How often to update subspace Q (default: 200)
        perp_grad_scale: Scale for perpendicular gradient term α (default: 0.1)
        norm_growth_limit: Maximum allowed norm growth ratio γ (default: 1.1)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        rank: int = 16,
        subspace_update_freq: int = 200,
        perp_grad_scale: float = 0.1,
        norm_growth_limit: float = 1.1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            rank=rank,
            subspace_update_freq=subspace_update_freq,
            perp_grad_scale=perp_grad_scale,
            norm_growth_limit=norm_growth_limit,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            rank = group['rank']
            K = group['subspace_update_freq']
            alpha = group['perp_grad_scale']
            gamma = group['norm_growth_limit']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Only apply SUMO to 2D parameters (weight matrices)
                if grad.ndim != 2:
                    # Fallback to Muon-style momentum for non-2D params
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(grad)
                    buf = state['momentum_buffer']
                    buf.lerp_(grad, 1 - momentum)
                    g = grad.lerp_(buf, momentum) if nesterov else buf
                    p.add_(g, alpha=-lr)
                    continue
                
                m, n = grad.shape
                state = self.state[p]
                
                # Initialize state
                if 'step' not in state:
                    state['step'] = 0
                    state['Q'] = None  # Left subspace basis (m x r)
                    state['M_hat'] = None  # Low-rank moment (r x n)
                    state['momentum_buffer'] = torch.zeros_like(grad)  # For Nesterov
                    state['prev_O_norm'] = None
                
                state['step'] += 1
                step = state['step']
                
                # ============================================
                # Nesterov momentum (same as Muon)
                # ============================================
                buf = state['momentum_buffer']
                buf.lerp_(grad, 1 - momentum)
                g = grad.lerp_(buf, momentum) if nesterov else buf
                
                # Ensure rank doesn't exceed matrix dimensions
                effective_rank = min(rank, m, n)
                
                # ============================================
                # Block 1: Adaptive Subspace Selection
                # ============================================
                if step % K == 1 or state['Q'] is None:
                    Q_new = randomized_svd(g, effective_rank)
                    
                    # Block 1.1: Transform moment to new subspace
                    if state['M_hat'] is not None and state['Q'] is not None:
                        Q_old = state['Q']
                        # Transform: M_hat_new = Q_new^T @ Q_old @ M_hat_old  
                        state['M_hat'] = Q_new.T @ Q_old @ state['M_hat']
                    
                    state['Q'] = Q_new
                
                Q = state['Q']
                
                # ============================================
                # Project gradient to low-rank subspace
                # ============================================
                G_hat = Q.T @ g  # r x n
                
                # ============================================
                # Block 2: Low-Rank Moment Update + Orthogonalization
                # ============================================
                if state['M_hat'] is None:
                    state['M_hat'] = G_hat.clone()
                else:
                    # EMA update in subspace
                    state['M_hat'].lerp_(G_hat, 1 - momentum)
                
                M_hat = state['M_hat']
                
                # Exact orthogonalization via SVD (with normalization like NS5)
                O = orth_svd(M_hat)  # r x n
                
                # ============================================
                # Block 3: Norm-Growth Limiter
                # ============================================
                O_norm = O.norm()
                if state['prev_O_norm'] is not None:
                    if O_norm > gamma * state['prev_O_norm']:
                        O = O * (gamma * state['prev_O_norm'] / O_norm)
                        O_norm = gamma * state['prev_O_norm']
                state['prev_O_norm'] = O_norm
                
                # ============================================
                # Block 4: Update in Original Space
                # ============================================
                # Project back: Q @ O gives m x n
                update = Q @ O
                
                # Add perpendicular gradient term
                if alpha > 0:
                    G_proj = Q @ G_hat
                    G_perp = g - G_proj
                    update = update + alpha * G_perp
                
                # Scale like Muon
                scale = max(1, m / n) ** 0.5
                p.add_(update, alpha=-lr * scale)
        
        return loss
