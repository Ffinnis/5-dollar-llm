"""
SUMO: Subspace-Aware Moment-Orthogonalization Optimizer

Exact implementation of Algorithm 1 from the paper.
"""

import torch
from typing import Optional


def randomized_svd(A: torch.Tensor, rank: int, n_oversamples: int = 5, n_iter: int = 2) -> torch.Tensor:
    """Randomized SVD for top-r left singular vectors."""
    m, n = A.shape
    r = min(rank + n_oversamples, min(m, n))
    
    Omega = torch.randn(n, r, device=A.device, dtype=A.dtype)
    Y = A @ Omega
    
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
    
    Q, _ = torch.linalg.qr(Y)
    return Q[:, :rank]


def orth_svd(M: torch.Tensor) -> torch.Tensor:
    """Orthogonalization_SVD(A) = argmin_O ||O - A||_F : O^T O = I or O O^T = I"""
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    return U @ Vt


class SUMO(torch.optim.Optimizer):
    """
    SUMO optimizer - exact Algorithm 1 from paper.
    
    Args:
        params: Parameters to optimize
        lr: Step size η (default: 0.02)
        alpha: Scale factor α for update (default: 1.0)
        mu: Momentum decay μ (default: 0.95)
        weight_decay: Weight decay λ (default: 0.0)
        rank: Low-rank dimension r (default: 16)
        subspace_update_freq: K - how often to update Q (default: 200)
        norm_growth_limit: γ for norm clipping (default: 1.1)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        alpha: float = 1.0,
        mu: float = 0.95,
        weight_decay: float = 0.0,
        rank: int = 16,
        subspace_update_freq: int = 200,
        norm_growth_limit: float = 1.1,
    ):
        defaults = dict(
            lr=lr,
            alpha=alpha,
            mu=mu,
            weight_decay=weight_decay,
            rank=rank,
            subspace_update_freq=subspace_update_freq,
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
            lr = group['lr']           # η
            alpha = group['alpha']     # α  
            mu = group['mu']           # μ
            wd = group['weight_decay'] # λ
            rank = group['rank']       # r
            K = group['subspace_update_freq']
            gamma = group['norm_growth_limit']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                G = p.grad  # G^(t)
                
                # Non-2D: simple momentum SGD with weight decay
                if G.ndim != 2:
                    state = self.state[p]
                    if 'M' not in state:
                        state['M'] = torch.zeros_like(G)
                    M = state['M']
                    M.mul_(mu).add_(G)  # M = μ*M + G
                    p.add_(M, alpha=-lr)
                    if wd > 0:
                        p.add_(p, alpha=-lr * wd)
                    continue
                
                m, n = G.shape
                state = self.state[p]
                
                if 'step' not in state:
                    state['step'] = 0
                    state['Q'] = None      # Q^(t) - subspace basis (m x r)
                    state['M'] = None      # M^(t) - moment in subspace (r x n)
                    state['prev_O_norm'] = None
                
                state['step'] += 1
                t = state['step']
                effective_rank = min(rank, m, n)
                
                # ============================================
                # Block 1: Subspace Selection (if t mod K = 0)
                # ============================================
                if t % K == 1 or state['Q'] is None:
                    Q_new = randomized_svd(G, effective_rank)
                    
                    # Block 1.1: Moment subspaces transformation
                    if state['Q'] is not None and state['M'] is not None:
                        Q_old = state['Q']
                        # R = Q_new^T @ Q_old (r x r rotation matrix)
                        R = Q_new.T @ Q_old
                        # M_new = R @ M_old
                        state['M'] = R @ state['M']
                    else:
                        # Initialize M to zeros
                        state['M'] = torch.zeros(effective_rank, n, device=G.device, dtype=G.dtype)
                    
                    state['Q'] = Q_new
                
                Q = state['Q']
                M = state['M']
                
                # ============================================
                # Project gradient: Ĝ = Q^T @ G
                # ============================================
                G_hat = Q.T @ G  # r x n
                
                # ============================================
                # Block 2: Moment update (SGD-style momentum!)
                # M^(t) = μ * M^(t-1) + Ĝ^(t)
                # ============================================
                M.mul_(mu).add_(G_hat)
                
                # Orthogonalize: O = SVD_orth(M)
                O = orth_svd(M)  # r x n
                
                # ============================================
                # Block 3: Norm-Growth Limiter (Optional)
                # ============================================
                O_norm = O.norm()
                if state['prev_O_norm'] is not None:
                    if O_norm / state['prev_O_norm'] > gamma:
                        O = O / O_norm * gamma * state['prev_O_norm']
                        O_norm = gamma * state['prev_O_norm']
                state['prev_O_norm'] = O_norm
                
                # ============================================
                # Block 4: Update weight in original space
                # W = W - αη * (G - Q @ (Ĝ - O)) - ηλW
                # 
                # Expanding: G - Q @ (Ĝ - O) = G - Q@Ĝ + Q@O
                #          = (G - Q@Q^T@G) + Q@O  
                #          = G_perp + Q@O
                # ============================================
                
                # Compute update: G - Q @ (Ĝ - O) = G - Q@Ĝ + Q@O
                update = G - Q @ G_hat + Q @ O  # m x n
                
                # Apply update: W -= αη * update
                p.add_(update, alpha=-alpha * lr)
                
                # Weight decay: W -= ηλ * W
                if wd > 0:
                    p.add_(p, alpha=-lr * wd)
        
        return loss
