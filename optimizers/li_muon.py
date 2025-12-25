import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

# Import Polar Express from muon.py for orthogonalization
from .muon import zeropower_polar_express


def rsvd(
    A: torch.Tensor, 
    rank: int, 
    oversampling: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD (RSVD) for low-rank matrix approximation.
    
    Compresses matrix A ∈ R^(m×n) into low-rank components U, S, V
    such that A ≈ U @ diag(S) @ V.T
    
    Args:
        A: Input matrix of shape (m, n)
        rank: Target rank for approximation
        oversampling: Extra dimensions for better accuracy (default: 5)
    
    Returns:
        U: Left singular vectors (m, rank)
        S: Singular values (rank,)
        V: Right singular vectors (n, rank)
    """
    m, n = A.shape
    l = rank + oversampling
    
    # Clamp rank to matrix dimensions
    effective_rank = min(rank, min(m, n) - 1)
    if effective_rank < 1:
        effective_rank = 1
    l = effective_rank + oversampling
    
    # Generate random Gaussian matrix for sketching
    Omega = torch.randn(n, l, device=A.device, dtype=A.dtype)
    
    # Sketching: Y = A @ Omega
    Y = A @ Omega  # (m, l)
    
    # QR decomposition for orthogonal basis
    Q, _ = torch.linalg.qr(Y)  # Q is (m, l)
    
    # Project A onto Q: B = Q.T @ A
    B = Q.T @ A  # (l, n)
    
    # SVD of smaller matrix B
    U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
    
    # Recover U = Q @ U_tilde
    U = Q @ U_tilde  # (m, min(l, n))
    V = Vh.T  # (n, min(l, n))
    
    # Truncate to target rank
    U = U[:, :effective_rank]
    S = S[:effective_rank]
    V = V[:, :effective_rank]
    
    return U, S, V


class LiMuon(torch.optim.Optimizer):
    """
    LiMuon - Light and Fast Muon Optimizer with TRUE STORM Variance Reduction
    
    Combines:
    - RSVD momentum compression for memory efficiency
    - TRUE STORM variance reduction: M = ∇f(W_new) + (1-β)(M̂ - ∇f(W_old))
    - Epoch-shift layer dropping for speed (optional)
    
    STORM requires computing gradients at BOTH current and previous weights
    on the SAME batch of data. The training loop must:
    1. Compute grad at current weights
    2. Temporarily restore previous weights  
    3. Compute grad at previous weights (same batch)
    4. Call optimizer.step() with prev_grads
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.02, same as Muon)
        momentum: Momentum coefficient for STORM (default: 0.95, same as Muon)
        rank: Target rank for RSVD compression (default: 8)
        oversampling: RSVD oversampling parameter (default: 5)
        ns_steps: Number of Polar Express iterations (default: 5)
        nesterov: Use Nesterov momentum (default: True)
        num_layers: Number of transformer layers for drop scheduling (default: 22)
        alpha: Epoch-shift temperature (default: 0.5)
        enable_drop: Enable epoch-shift layer dropping (default: False)
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
        enable_drop: bool = False,
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
        self.num_layers = num_layers
        self.alpha = alpha
        self.enable_drop = enable_drop
        self._prev_weights: Dict[int, torch.Tensor] = {}  # Store previous weights by param id
    
    def _sample_cutoff(self, progress: float) -> int:
        """Epoch-shift sampling: bias toward shallow layers early, deep layers late."""
        if not self.enable_drop:
            return 0
        
        b = self.num_layers
        indices = torch.arange(b, dtype=torch.float32)
        weights = torch.exp(self.alpha * ((1 - progress) * (b - 1 - indices) + progress * indices))
        probs = weights / weights.sum()
        return torch.multinomial(probs, 1).item()
    
    def save_prev_weights(self):
        """Save current weights as previous weights for STORM computation."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self._prev_weights[id(p)] = p.data.clone()
    
    def get_prev_weights(self) -> Dict[int, torch.Tensor]:
        """Return saved previous weights."""
        return self._prev_weights
    
    @torch.no_grad()
    def step(self, closure=None, progress: float = 0.0, prev_grads: Optional[Dict[int, torch.Tensor]] = None):
        """
        Performs a single optimization step with TRUE STORM variance reduction.
        
        Args:
            closure: Optional closure for loss computation
            progress: Training progress [0, 1] for epoch-shift layer dropping
            prev_grads: Dict mapping param id -> gradient at previous weights
                       Required for true STORM. If None, falls back to simplified update.
        
        The TRUE STORM algorithm:
        1. M_new = ∇f(W_new; batch) + (1-β) * (M̂ - ∇f(W_old; batch))
        2. Orthogonalize M_new using Polar Express
        3. Update weights: W = W - lr * orthogonalized(M_new)
        4. Compress M_new to low-rank via RSVD
        
        Returns:
            cutoff: The sampled cutoff layer index (for dropping stats)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        cutoff = self._sample_cutoff(progress)
        
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            rank = group["rank"]
            oversampling = group["oversampling"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            layer_idx = group.get("layer_idx", 0)
            
            # Skip frozen layers (below cutoff)
            if self.enable_drop and layer_idx < cutoff:
                continue
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                g = p.grad  # ∇f(W_new; batch)
                state = self.state[p]
                param_id = id(p)
                
                # Get previous gradient if available (for true STORM)
                prev_g = prev_grads.get(param_id) if prev_grads else None
                
                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    state["use_lowrank"] = False
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                state["step"] += 1
                
                if not state["use_lowrank"]:
                    # First few steps: use full momentum buffer
                    buf = state["momentum_buffer"]
                    
                    if prev_g is not None:
                        # TRUE STORM: M = g + (1-β)(M̂ - prev_g)
                        storm_correction = buf - prev_g
                        buf.copy_(g + (1 - momentum) * storm_correction)
                    else:
                        # Fallback: standard momentum
                        buf.lerp_(g, 1 - momentum)
                    
                    # After accumulating some momentum, switch to low-rank
                    if state["step"] >= 3 and g.ndim == 2:
                        try:
                            U, S, V = rsvd(buf, rank, oversampling)
                            state["U"] = U
                            state["S"] = S
                            state["V"] = V
                            state["use_lowrank"] = True
                            del state["momentum_buffer"]
                        except Exception:
                            pass
                    
                    # Apply Nesterov momentum
                    if nesterov:
                        g = g.lerp_(buf, momentum)
                    else:
                        g = buf.clone()
                else:
                    # Low-rank path: reconstruct, STORM update, compress
                    U, S, V = state["U"], state["S"], state["V"]
                    
                    # Reconstruct momentum: M̂ = U @ diag(S) @ V.T
                    M_hat = U @ torch.diag(S) @ V.T
                    
                    if prev_g is not None:
                        # TRUE STORM: M = g + (1-β)(M̂ - prev_g)
                        storm_correction = M_hat - prev_g
                        M_new = g + (1 - momentum) * storm_correction
                    else:
                        # Fallback: simplified update
                        M_new = g + (1 - momentum) * M_hat
                    
                    # Compress back to low-rank
                    try:
                        U, S, V = rsvd(M_new, rank, oversampling)
                        state["U"] = U
                        state["S"] = S
                        state["V"] = V
                    except Exception:
                        pass
                    
                    # Apply Nesterov momentum
                    if nesterov:
                        g = g.lerp_(M_new, momentum)
                    else:
                        g = M_new
                
                # Orthogonalize using Polar Express (same as Muon)
                g = zeropower_polar_express(g, steps=ns_steps)
                g = g.to(p.dtype)
                
                # Update weights with learning rate scaling
                p.add_(g.view_as(p), alpha=-lr * max(1, p.size(-2) / p.size(-1))**0.5)
        
        # Clear previous weights after use
        self._prev_weights.clear()
        
        return cutoff
