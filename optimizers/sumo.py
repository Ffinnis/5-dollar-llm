"""
SUMO: Subspace-Aware Moment-Orthogonalization Optimizer

Based on the paper: "SUMO: Subspace-Aware Moment-Orthogonalization for 
Accelerating Memory-Efficient LLM Training"

Matches paper algorithm exactly:
1. Project gradient to subspace: Ĝ = Q^T @ G  
2. EMA update: M̂ = β*M̂ + (1-β)*Ĝ
3. Orthogonalize M̂ via SVD
4. Update weights with orthogonalized moment
"""

import torch
from typing import Optional


def randomized_svd(A: torch.Tensor, rank: int, n_oversamples: int = 5, n_iter: int = 2) -> torch.Tensor:
    """Randomized SVD for efficient top-r left singular vectors."""
    m, n = A.shape
    r = min(rank + n_oversamples, min(m, n))
    
    Omega = torch.randn(n, r, device=A.device, dtype=A.dtype)
    Y = A @ Omega
    
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
    
    Q, _ = torch.linalg.qr(Y)
    return Q[:, :rank]


def orth_svd(M: torch.Tensor) -> torch.Tensor:
    """Exact orthogonalization via SVD with normalization."""
    # Normalize like Newton-Schulz
    M_normalized = M / (M.norm() + 1e-7)
    U, S, Vt = torch.linalg.svd(M_normalized, full_matrices=False)
    return U @ Vt


class SUMO(torch.optim.Optimizer):
    """
    SUMO optimizer following paper algorithm exactly.
    
    Key difference from Muon: operates in low-rank subspace with exact SVD.
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        rank: int = 16,
        subspace_update_freq: int = 200,
        perp_grad_scale: float = 0.1,
        norm_growth_limit: float = 1.1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            rank=rank,
            subspace_update_freq=subspace_update_freq,
            perp_grad_scale=perp_grad_scale,
            norm_growth_limit=norm_growth_limit,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']  # β in paper
            rank = group['rank']
            K = group['subspace_update_freq']
            alpha = group['perp_grad_scale']
            gamma = group['norm_growth_limit']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Non-2D: simple momentum SGD
                if grad.ndim != 2:
                    state = self.state[p]
                    if 'buf' not in state:
                        state['buf'] = torch.zeros_like(grad)
                    buf = state['buf']
                    buf.mul_(beta).add_(grad, alpha=1 - beta)
                    p.add_(buf, alpha=-lr)
                    continue
                
                m, n = grad.shape
                state = self.state[p]
                
                if 'step' not in state:
                    state['step'] = 0
                    state['Q'] = None
                    state['M_hat'] = None
                    state['prev_O_norm'] = None
                
                state['step'] += 1
                step = state['step']
                effective_rank = min(rank, m, n)
                
                # ============================================
                # Block 1: Subspace Selection (every K steps)
                # ============================================
                if step % K == 1 or state['Q'] is None:
                    Q_new = randomized_svd(grad, effective_rank)
                    
                    # Transform moment to new subspace
                    if state['M_hat'] is not None and state['Q'] is not None:
                        Q_old = state['Q']
                        state['M_hat'] = Q_new.T @ Q_old @ state['M_hat']
                    
                    state['Q'] = Q_new
                
                Q = state['Q']
                
                # ============================================
                # Project gradient to subspace: Ĝ = Q^T @ G
                # ============================================
                G_hat = Q.T @ grad  # r x n
                
                # ============================================
                # Block 2: EMA moment update (paper eq)
                # M̂ = β * M̂ + (1-β) * Ĝ
                # ============================================
                if state['M_hat'] is None:
                    state['M_hat'] = G_hat.clone()
                else:
                    state['M_hat'].mul_(beta).add_(G_hat, alpha=1 - beta)
                
                M_hat = state['M_hat']
                
                # Orthogonalize via exact SVD
                O = orth_svd(M_hat)
                
                # ============================================
                # Block 3: Norm-Growth Limiter
                # ============================================
                O_norm = O.norm()
                if state['prev_O_norm'] is not None and O_norm > gamma * state['prev_O_norm']:
                    O = O * (gamma * state['prev_O_norm'] / O_norm)
                    O_norm = gamma * state['prev_O_norm']
                state['prev_O_norm'] = O_norm
                
                # ============================================
                # Block 4: Update with perpendicular term
                # ============================================
                update = Q @ O  # m x n
                
                if alpha > 0:
                    G_perp = grad - Q @ G_hat  # Perpendicular to subspace
                    update = update + alpha * G_perp
                
                # Scale like Muon
                scale = max(1, m / n) ** 0.5
                p.add_(update, alpha=-lr * scale)
        
        return loss
