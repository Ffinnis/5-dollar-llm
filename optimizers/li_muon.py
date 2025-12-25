import torch
from typing import Optional, Dict, Tuple


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
    l = rank + oversampling
    
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
    LiMuon - Light and Fast Muon Optimizer (Algorithm 1 from paper).
    
    Combines:
    - SVD-based orthogonalization: O_t = U_t @ V_t^T
    - STORM variance reduction: M_{t+1} = ∇f(W_{t+1}) + (1-β)(M̂_t - ∇f(W_t))
    - RSVD momentum compression for memory efficiency (Option #2)
    
    Key difference from Muon:
    - Uses SVD for orthogonalization instead of Newton-Schulz/Polar Express
    - Uses STORM variance-reduced gradient estimator
    - Stores momentum in compressed low-rank form
    
    STORM requires gradients at BOTH current and previous weights on the SAME batch.
    The training loop must:
    1. Forward pass on W_{t+1}, compute grad ∇f(W_{t+1}; ξ)
    2. Forward pass on W_t (saved), compute grad ∇f(W_t; ξ) - same batch!
    3. Call optimizer.step() with prev_grads dict
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate η (default: 0.02)
        momentum: STORM coefficient β ∈ [0, 1) (default: 0.95)
        rank: Target rank r̂ for RSVD compression (default: 8)
        oversampling: RSVD oversampling parameter s (default: 5)
    """
    
    def __init__(
        self, 
        params, 
        lr: float = 0.02,
        momentum: float = 0.95,
        rank: int = 8,
        oversampling: int = 5,
    ):
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            rank=rank,
            oversampling=oversampling,
        )
        super().__init__(params, defaults)
        self._prev_weights: Dict[int, torch.Tensor] = {}
    
    def save_prev_weights(self):
        """Save current weights W_t before update for STORM computation."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self._prev_weights[id(p)] = p.data.clone()
    
    def get_prev_weights(self) -> Dict[int, torch.Tensor]:
        """Return saved previous weights for gradient computation."""
        return self._prev_weights
    
    @torch.no_grad()
    def step(self, closure=None, prev_grads: Optional[Dict[int, torch.Tensor]] = None):
        """
        Performs a single LiMuon optimization step.
        
        Algorithm 1 from paper (Option #2 - with RSVD compression):
        
        For each parameter:
        1. (U_t, Σ_t, V_t) = SVD(M_t)           # Full SVD for orthogonalization
        2. W_{t+1} = W_t - η * U_t @ V_t^T      # Update with orthogonalized momentum
        3. M_{t+1} = ∇f(W_{t+1}) + (1-β)(M̂_t - ∇f(W_t))  # STORM update
        4. (Ũ, S̃, Ṽ) = RSVD(M_{t+1})           # Compress for storage
        
        Note: Steps 1-2 use PREVIOUS M_t, steps 3-4 compute NEXT M_{t+1}
        Since gradients come from outside, we:
        - Use stored M_t to compute orthogonalized update
        - Update M_{t+1} using current gradient and STORM correction
        
        Args:
            closure: Optional closure for loss computation  
            prev_grads: Dict mapping param id -> gradient at previous weights ∇f(W_t; ξ)
                       Required for true STORM. If None, falls back to standard momentum.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]  # β in paper
            rank = group["rank"]
            oversampling = group["oversampling"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                g = p.grad  # ∇f(W_{t+1}; ξ_{t+1}) - gradient at current weights
                state = self.state[p]
                param_id = id(p)
                
                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    # M_0 = ∇f(W_0; ξ_0)
                    state["M"] = g.clone()
                
                state["step"] += 1
                M_t = state["M"]  # Current momentum estimate
                
                # --- Step 4-5: Orthogonalize M_t and update weights ---
                if g.ndim == 2 and min(g.shape) > 1:
                    # SVD for orthogonalization: O_t = U_t @ V_t^T
                    try:
                        U, S, Vh = torch.linalg.svd(M_t.float(), full_matrices=False)
                        # Orthogonalized direction: O_t = U @ V^T
                        O_t = (U @ Vh).to(p.dtype)
                    except Exception:
                        # Fallback if SVD fails
                        O_t = M_t / (M_t.norm() + 1e-8)
                else:
                    # For 1D tensors, just normalize
                    O_t = M_t / (M_t.norm() + 1e-8)
                
                # W_{t+1} = W_t - η_t * O_t (with lr scaling like Muon)
                scale = max(1, p.size(-2) / p.size(-1)) ** 0.5 if p.ndim >= 2 else 1.0
                p.add_(O_t.view_as(p), alpha=-lr * scale)
                
                # --- Steps 7/9: STORM update for M_{t+1} ---
                # M_{t+1} = ∇f(W_{t+1}; ξ) + (1-β)(M̂_t - ∇f(W_t; ξ))
                prev_g = prev_grads.get(param_id) if prev_grads else None
                
                if prev_g is not None:
                    # True STORM variance reduction
                    # M_{t+1} = g + (1-β)(M_t - prev_g)
                    M_new = g + (1 - beta) * (M_t - prev_g)
                else:
                    # Fallback: standard EMA momentum (like regular Muon)
                    # M_{t+1} = β * M_t + (1-β) * g
                    M_new = beta * M_t + (1 - beta) * g
                
                # --- Step 8: RSVD compression for memory efficiency ---
                if g.ndim == 2 and min(g.shape) > rank:
                    try:
                        U, S, V = rsvd(M_new, rank, oversampling)
                        # Store compressed: M̂ = U @ diag(S) @ V^T
                        # Reconstruct immediately for next iteration
                        state["M"] = U @ torch.diag(S) @ V.T
                    except Exception:
                        state["M"] = M_new
                else:
                    # Small tensors: store full momentum
                    state["M"] = M_new
        
        # Clear previous weights after use
        self._prev_weights.clear()
        
        return loss
