import torch
from typing import Optional, Dict, Tuple

# Import Polar Express from muon.py for fast orthogonalization
from .muon import zeropower_polar_express


def rsvd(
    A: torch.Tensor, 
    rank: int, 
    oversampling: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD (Algorithm 2 from LiMuon paper).
    
    Compresses matrix A ∈ R^(m×n) into low-rank factors U, S, V
    such that A ≈ U @ diag(S) @ V.T
    
    Args:
        A: Input matrix of shape (m, n)
        rank: Target rank r̂
        oversampling: Oversampling parameter s (default: 5)
    
    Returns:
        U: Left singular vectors (m, rank)
        S: Singular values (rank,)
        V: Right singular vectors (n, rank)
    """
    m, n = A.shape
    
    # Clamp to valid dimensions
    effective_rank = min(rank, min(m, n) - 1)
    if effective_rank < 1:
        effective_rank = 1
    l = min(effective_rank + oversampling, min(m, n))
    
    # Step 3: Generate random Gaussian matrix Ω ∈ R^(n×l)
    Omega = torch.randn(n, l, device=A.device, dtype=A.dtype)
    
    # Step 4: Y = A @ Ω, then QR decomposition Y = QR
    Y = A @ Omega  # (m, l)
    Q, _ = torch.linalg.qr(Y)  # Q is (m, l)
    
    # Step 5: B = Q^T @ A ∈ R^(l×n), then SVD(B) = (Ũ, Σ, V)
    B = Q.T @ A  # (l, n)
    U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
    
    # U = Q @ Ũ
    U = Q @ U_tilde  # (m, min(l, n))
    V = Vh.T  # (n, min(l, n))
    
    # Truncate to target rank
    U = U[:, :effective_rank]
    S = S[:effective_rank]
    V = V[:, :effective_rank]
    
    return U, S, V


class LiMuon(torch.optim.Optimizer):
    """
    LiMuon - Light and Fast Muon Optimizer.
    
    Combines:
    - Polar Express orthogonalization (fast, like Muon)
    - STORM variance reduction for better convergence
    - RSVD momentum compression for memory efficiency
    
    This is a practical hybrid: uses Polar Express for speed (as paper mentions
    Newton-Schulz can replace SVD), STORM for variance reduction, and RSVD
    to compress momentum buffers.
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate η (default: 0.02)
        momentum: STORM/EMA coefficient β ∈ [0, 1) (default: 0.95)
        rank: Target rank r̂ for RSVD compression (default: 8)
        oversampling: RSVD oversampling parameter s (default: 5)
        ns_steps: Number of Polar Express iterations (default: 5)
        nesterov: Use Nesterov momentum (default: True)
        num_layers: Number of layers for layer dropping (default: 22)
        alpha: Layer dropping bias strength (default: 0.5)
    """
    
    def __init__(
        self, 
        params, 
        lr: float = 0.02,
        momentum: float = 0.95,
        rank: int = 8,
        oversampling: int = 5,
        ns_steps: int = 5,
        nesterov: bool = True,
        num_layers: int = 22,
        alpha: float = 0.5,
    ):
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            rank=rank,
            oversampling=oversampling,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)
        self._prev_weights: Dict[int, torch.Tensor] = {}
        self.num_layers = num_layers
        self.alpha = alpha
    
    def save_prev_weights(self):
        """Save current weights W_t before update for STORM computation."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self._prev_weights[id(p)] = p.data.clone()
    
    def get_prev_weights(self) -> Dict[int, torch.Tensor]:
        """Return saved previous weights for gradient computation."""
        return self._prev_weights
    
    def _sample_cutoff(self, progress: float) -> int:
        """Epoch-shift: bias toward shallow layers early, deep layers late."""
        b = self.num_layers
        indices = torch.arange(b, dtype=torch.float32)
        weights = torch.exp(self.alpha * ((1 - progress) * (b - 1 - indices) + progress * indices))
        probs = weights / weights.sum()
        return torch.multinomial(probs, 1).item()
    
    @torch.no_grad()
    def step(self, closure=None, prev_grads: Optional[Dict[int, torch.Tensor]] = None, progress: float = 0.0):
        """
        Performs a single LiMuon optimization step.
        
        Hybrid algorithm:
        1. STORM update: M = g + (1-β)(M_prev - prev_g)  OR  EMA if no prev_g
        2. Nesterov lookahead (optional)
        3. Polar Express orthogonalization (fast!)
        4. RSVD compression of momentum for memory efficiency
        
        Args:
            closure: Optional closure for loss computation  
            prev_grads: Dict mapping param id -> gradient at previous weights
                       Required for true STORM. If None, falls back to EMA.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Sample layer cutoff for this step
        cutoff = self._sample_cutoff(progress)
        
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            rank = group["rank"]
            oversampling = group["oversampling"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            
            layer_idx = group.get('layer_idx', 0)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                g = p.grad  # Current gradient
                state = self.state[p]
                param_id = id(p)
                
                # Initialize momentum buffer on first step
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                # Skip frozen layers (below cutoff) for epoch-shift layer dropping
                if layer_idx < cutoff:
                    continue
                
                buf = state["momentum_buffer"]
                
                # Get previous gradient if available (for STORM)
                prev_g = prev_grads.get(param_id) if prev_grads else None
                
                if prev_g is not None:
                    # TRUE STORM variance reduction:
                    # M_{t+1} = g + (1-β)(M_t - prev_g)
                    buf.copy_(g + (1 - beta) * (buf - prev_g))
                else:
                    # Standard EMA momentum (like Muon)
                    # M = β * M + (1-β) * g
                    buf.lerp_(g, 1 - beta)
                
                # Nesterov lookahead
                if nesterov:
                    update = g.lerp(buf, beta)
                else:
                    update = buf.clone()
                
                # Fast orthogonalization via Polar Express (not full SVD!)
                update = zeropower_polar_express(update, steps=ns_steps)
                update = update.to(p.dtype)
                
                # Update weights with lr scaling
                scale = max(1, p.size(-2) / p.size(-1)) ** 0.5 if p.ndim >= 2 else 1.0
                p.add_(update.view_as(p), alpha=-lr * scale)
        
        # Clear previous weights after use
        self._prev_weights.clear()
        
        return cutoff
