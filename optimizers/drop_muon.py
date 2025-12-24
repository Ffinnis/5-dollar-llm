"""
Drop-Muon: Muon with Randomized Progressive Training (RPT)

Instead of updating all layers at every step, Drop-Muon samples a subset 
of layers to update while freezing the rest. This reduces computational 
cost and wall-clock training time.

Reference: Drop-Muon paper - Randomized Progressive Training for Muon
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Any
from .muon import zeropower_polar_express


class DropMuon(torch.optim.Optimizer):
    """
    Drop-Muon - Muon optimizer with Randomized Progressive Training.
    
    At each step, samples a cutoff layer index. Only layers >= cutoff are updated,
    while layers < cutoff are frozen (parameters and momentum buffers preserved).
    
    Args:
        params: Iterable of parameters or param groups with 'layer_idx' key
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        drop_strategy: 'uniform', 'epoch_shift', or 'none' (default: 'uniform')
        num_layers: Total number of layers in the model (default: 22)
        alpha: Sharpness parameter for epoch-shift distribution (default: 0.5)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        drop_strategy: str = 'uniform',
        num_layers: int = 22,
        alpha: float = 0.5,
    ):
        if drop_strategy not in ('uniform', 'epoch_shift', 'none'):
            raise ValueError(f"drop_strategy must be 'uniform', 'epoch_shift', or 'none', got {drop_strategy}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        
        self.drop_strategy = drop_strategy
        self.num_layers = num_layers
        self.alpha = alpha
        
        # Cache for epoch-shift distribution weights
        self._epoch_shift_cache: Dict[float, torch.Tensor] = {}
        
    def _compute_epoch_shift_weights(self, progress: float) -> torch.Tensor:
        """
        Compute epoch-shift distribution weights.
        
        Early in training (progress ~= 0): bias toward shallow layers (low indices)
        Late in training (progress ~= 1): bias toward deep layers (high indices)
        
        w_i = exp(alpha * [(1 - progress) * (num_layers - 1 - i) + progress * i])
        """
        # Round progress to 2 decimal places for caching
        cache_key = round(progress, 2)
        
        if cache_key in self._epoch_shift_cache:
            return self._epoch_shift_cache[cache_key]
        
        b = self.num_layers
        indices = torch.arange(b, dtype=torch.float32)
        
        # Weight calculation: bias toward shallow early, deep later
        weights = torch.exp(
            self.alpha * ((1 - progress) * (b - 1 - indices) + progress * indices)
        )
        
        # Normalize to probabilities
        probs = weights / weights.sum()
        
        # Cache for reuse
        self._epoch_shift_cache[cache_key] = probs
        
        # Keep cache size bounded
        if len(self._epoch_shift_cache) > 100:
            # Remove oldest entries
            keys = list(self._epoch_shift_cache.keys())[:-50]
            for k in keys:
                del self._epoch_shift_cache[k]
        
        return probs
    
    def sample_active_layers(self, progress: float = 0.0) -> int:
        """
        Sample the cutoff layer index.
        
        Args:
            progress: Training progress in [0, 1] (epoch / max_epochs)
        
        Returns:
            start_idx: Layers with index >= start_idx are active (updated).
                      Layers with index < start_idx are frozen.
        """
        if self.drop_strategy == 'none':
            return 0  # All layers active
        
        if self.drop_strategy == 'uniform':
            # Each layer has equal probability of being the cutoff
            start_idx = torch.randint(0, self.num_layers, (1,)).item()
            return start_idx
        
        elif self.drop_strategy == 'epoch_shift':
            # Sample from epoch-shift distribution
            probs = self._compute_epoch_shift_weights(progress)
            start_idx = torch.multinomial(probs, 1).item()
            return start_idx
        
        return 0
    
    @torch.no_grad()
    def step(self, start_idx: Optional[int] = None, progress: float = 0.0):
        """
        Perform optimization step with optional layer dropping.
        
        Args:
            start_idx: If None, sample a new cutoff using the configured strategy.
                      If provided, use this as the cutoff index.
            progress: Training progress in [0, 1] for epoch-shift distribution.
        
        Returns:
            start_idx: The cutoff index that was used.
        """
        if start_idx is None:
            start_idx = self.sample_active_layers(progress)
        
        for group in self.param_groups:
            layer_idx = group.get('layer_idx', 0)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer if needed
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                # Check if this layer is active
                if layer_idx < start_idx:
                    # FROZEN: Do not update parameters or momentum
                    # Gradient is already computed, we just skip the update
                    continue
                
                # ACTIVE: Update momentum and parameters
                buf = state['momentum_buffer']
                
                # Update momentum buffer: M = (1 - beta) * M + beta * grad
                buf.lerp_(g, 1 - group['momentum'])
                
                # Apply Nesterov momentum if enabled
                if group['nesterov']:
                    g = g.lerp_(buf, group['momentum'])
                else:
                    g = buf
                
                # Apply Newton-Schulz orthogonalization (Polar Express)
                g = zeropower_polar_express(g, steps=group['ns_steps'])
                g = g.to(p.dtype)
                
                # Update parameters with scaled learning rate
                # Scale by sqrt(max(1, rows/cols)) as in original Muon
                scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(g.view_as(p), alpha=-group['lr'] * scale)
        
        return start_idx
    
    def get_drop_stats(self) -> Dict[str, Any]:
        """Return statistics about the optimizer configuration."""
        return {
            'drop_strategy': self.drop_strategy,
            'num_layers': self.num_layers,
            'alpha': self.alpha if self.drop_strategy == 'epoch_shift' else None,
        }
