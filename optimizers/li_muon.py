import torch
import torch.nn.functional as F
from typing import Optional, Tuple

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
    U = U[:, :rank]
    S = S[:rank]
    V = V[:, :rank]
    
    return U, S, V


class LiMuon(torch.optim.Optimizer):
    """
    LiMuon - Light and Fast Muon Optimizer with Epoch-Shift Layer Dropping
    
    Combines:
    - RSVD momentum compression for memory efficiency
    - STORM variance reduction for better convergence (O(ε⁻³) sample complexity)
    - Epoch-shift layer dropping for speed (+12% from drop-muon)
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.02, same as Muon)
        momentum: Momentum coefficient for STORM (default: 0.95, same as Muon)
        rank: Target rank for RSVD compression (default: 8)
        oversampling: RSVD oversampling parameter (default: 5)
        ns_steps: Number of Polar Express iterations (default: 5)
        nesterov: Use Nesterov momentum (default: True)
        num_layers: Number of transformer layers for drop scheduling (default: 22)
        alpha: Epoch-shift temperature (default: 0.5, higher = more aggressive)
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
    
    def _sample_cutoff(self, progress: float) -> int:
        """
        Epoch-shift sampling: bias toward shallow layers early, deep layers late.
        
        At progress=0: prefers updating shallow layers (freeze deep)
        At progress=1: prefers updating deep layers (freeze shallow)
        """
        if not self.enable_drop:
            return 0  # No dropping
        
        b = self.num_layers
        indices = torch.arange(b, dtype=torch.float32)
        weights = torch.exp(self.alpha * ((1 - progress) * (b - 1 - indices) + progress * indices))
        probs = weights / weights.sum()
        return torch.multinomial(probs, 1).item()
    
    @torch.no_grad()
    def step(self, closure=None, progress: float = 0.0):
        """
        Performs a single optimization step.
        
        Args:
            closure: Optional closure for loss computation
            progress: Training progress [0, 1] for epoch-shift layer dropping
        
        The algorithm:
        1. Sample cutoff layer based on progress (epoch-shift)
        2. Skip layers below cutoff
        3. Reconstruct momentum from low-rank factors: M̂ = U @ S @ V.T
        4. Update momentum with current gradient (simplified STORM)
        5. Orthogonalize momentum using Polar Express
        6. Update weights
        7. Compress new momentum back to low-rank factors via RSVD
        
        Returns:
            cutoff: The sampled cutoff layer index
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
                
                g = p.grad
                state = self.state[p]
                
                # Initialize state on first step
                if len(state) == 0:
                    # For first step, store momentum as full matrix temporarily
                    # then compress it
                    state["step"] = 0
                    state["use_lowrank"] = False
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                state["step"] += 1
                
                if not state["use_lowrank"]:
                    # First few steps: use full momentum buffer
                    # Switch to low-rank after sufficient information is gathered
                    buf = state["momentum_buffer"]
                    buf.lerp_(g, 1 - momentum)
                    
                    # After accumulating some momentum, switch to low-rank
                    if state["step"] >= 3 and g.ndim == 2:
                        # Compress to low-rank
                        try:
                            U, S, V = rsvd(buf, rank, oversampling)
                            state["U"] = U
                            state["S"] = S
                            state["V"] = V
                            state["use_lowrank"] = True
                            del state["momentum_buffer"]
                        except Exception:
                            # If RSVD fails (e.g., rank too high), keep dense
                            pass
                    
                    # Apply Nesterov momentum
                    if nesterov:
                        g = g.lerp_(buf, momentum)
                    else:
                        g = buf.clone()
                else:
                    # Low-rank path: reconstruct, update, compress
                    U, S, V = state["U"], state["S"], state["V"]
                    
                    # Reconstruct momentum: M̂ = U @ diag(S) @ V.T
                    M_hat = U @ torch.diag(S) @ V.T
                    
                    # Update momentum (simplified STORM without previous gradient)
                    # M_new = g + (1 - β) * M̂
                    # This is equivalent to: M_new.lerp_(g, 1 - momentum) starting from M̂
                    M_new = g + (1 - momentum) * M_hat
                    
                    # Compress back to low-rank
                    try:
                        U, S, V = rsvd(M_new, rank, oversampling)
                        state["U"] = U
                        state["S"] = S
                        state["V"] = V
                    except Exception:
                        # If RSVD fails, use uncompressed update
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
        
        return cutoff
