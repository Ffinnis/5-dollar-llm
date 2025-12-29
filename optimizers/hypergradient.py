"""
Hypergradient Descent for Adaptive Learning Rate Tuning

This module implements hypergradient descent as described in:
"Online Learning Rate Adaptation with Hypergradient Descent" (Baydin et al., 2017)

The key insight is that we can compute the gradient of the loss with respect to
the learning rate and use it to adapt LR during training:

    η_{t+1} = η_t - β · ∂L/∂η
    
where ∂L/∂η ≈ -⟨g_t, Δθ_{t-1}⟩ (dot product of current gradient and previous update)
"""

import torch
from typing import List, Optional, Dict, Any
from collections import deque


class HypergradientWrapper:
    """
    Wraps optimizer(s) to add hypergradient-based learning rate adaptation.
    
    This wrapper computes the hypergradient (gradient of loss w.r.t. learning rate)
    and uses it to automatically tune the learning rate during training.
    
    Args:
        optimizers: List of optimizers to wrap (typically [Muon, AdamW])
        hyper_lr: Learning rate for updating the learning rate (meta-LR)
        lr_min: Minimum allowed learning rate
        lr_max: Maximum allowed learning rate
        warmup_steps: Number of steps before enabling hypergradient updates
        loss_window: Size of loss history window for stability checks
        adapt_muon_only: If True, only adapt LR for Muon optimizer (index 0)
    """
    
    def __init__(
        self,
        optimizers: List[torch.optim.Optimizer],
        hyper_lr: float = 1e-7,
        lr_min: float = 1e-5,
        lr_max: float = 0.1,
        warmup_steps: int = 50,
        loss_window: int = 10,
        adapt_muon_only: bool = True,
    ):
        self.optimizers = optimizers
        self.hyper_lr = hyper_lr
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.loss_window = loss_window
        self.adapt_muon_only = adapt_muon_only
        
        # State tracking
        self.step_count = 0
        self.prev_updates: Dict[int, torch.Tensor] = {}  # param id -> previous update
        self.loss_history = deque(maxlen=loss_window)
        self.lr_history: List[float] = []  # Track LR changes for logging
        
        # Store initial LRs
        self.initial_lrs = []
        for opt in optimizers:
            self.initial_lrs.append([pg['lr'] for pg in opt.param_groups])
    
    def _compute_hypergradient(self, optimizer: torch.optim.Optimizer) -> float:
        """
        Compute hypergradient for an optimizer.
        
        The hypergradient is: ∂L/∂η ≈ -⟨g_t, Δθ_{t-1}⟩
        
        Returns:
            Scalar hypergradient value
        """
        hypergradient = 0.0
        count = 0
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                param_id = id(p)
                
                if param_id in self.prev_updates:
                    # Compute dot product: -⟨g_t, Δθ_{t-1}⟩
                    # Note: negative because we want ∂L/∂η
                    prev_update = self.prev_updates[param_id]
                    
                    # Flatten both tensors for dot product
                    g_flat = p.grad.view(-1).float()
                    delta_flat = prev_update.view(-1).float()
                    
                    # Accumulate dot product
                    dot = torch.dot(g_flat, delta_flat).item()
                    hypergradient -= dot  # Negative because Δθ = -lr * g
                    count += 1
        
        # Normalize by number of parameters
        if count > 0:
            hypergradient /= count
            
        return hypergradient
    
    def _store_updates(self, optimizer: torch.optim.Optimizer, lr: float):
        """Store current parameter updates for next step's hypergradient calculation."""
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                param_id = id(p)
                # The update is approximately: Δθ = -lr * g (simplified)
                # For more accuracy with momentum, we'd need optimizer state
                # But this approximation works well in practice
                self.prev_updates[param_id] = (p.grad.clone() * lr).detach()
    
    def _check_loss_stability(self, current_loss: Optional[float]) -> float:
        """
        Check loss trajectory and return a stability multiplier.
        
        Returns:
            Multiplier in [0, 1] - lower if loss is unstable
        """
        if current_loss is None or len(self.loss_history) < 2:
            return 1.0
            
        # Check if loss increased significantly
        avg_recent = sum(list(self.loss_history)[-3:]) / min(3, len(self.loss_history))
        
        if current_loss > avg_recent * 1.5:
            # Loss spiked - reduce LR adaptation
            return 0.1
        elif current_loss > avg_recent * 1.1:
            # Mild increase - be more conservative
            return 0.5
            
        return 1.0
    
    def step(self, current_loss: Optional[float] = None):
        """
        Perform optimizer step with hypergradient LR adaptation.
        
        Args:
            current_loss: Current training loss (optional, for stability checks)
        """
        self.step_count += 1
        
        # Update loss history
        if current_loss is not None:
            self.loss_history.append(current_loss)
        
        # Determine which optimizers to adapt
        optimizers_to_adapt = [self.optimizers[0]] if self.adapt_muon_only else self.optimizers
        
        for opt_idx, optimizer in enumerate(self.optimizers):
            # Store updates before optimizer step (need gradients)
            current_lr = optimizer.param_groups[0]['lr']
            self._store_updates(optimizer, current_lr)
            
            # Perform the actual optimizer step
            optimizer.step()
            
            # Skip LR adaptation for warmup period or non-adapted optimizers
            if self.step_count <= self.warmup_steps:
                continue
            if optimizer not in optimizers_to_adapt:
                continue
                
            # Compute hypergradient
            hg = self._compute_hypergradient(optimizer)
            
            # Apply stability check
            stability = self._check_loss_stability(current_loss)
            
            # Update learning rate
            for group in optimizer.param_groups:
                old_lr = group['lr']
                
                # Hypergradient descent step with stability scaling
                new_lr = old_lr - self.hyper_lr * hg * stability
                
                # Clip to bounds
                new_lr = max(self.lr_min, min(self.lr_max, new_lr))
                
                group['lr'] = new_lr
            
            # Track LR for the first optimizer (Muon)
            if opt_idx == 0:
                self.lr_history.append(optimizer.param_groups[0]['lr'])
    
    def zero_grad(self):
        """Zero gradients for all wrapped optimizers."""
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    
    def get_current_lr(self, optimizer_idx: int = 0) -> float:
        """Get current learning rate for specified optimizer."""
        return self.optimizers[optimizer_idx].param_groups[0]['lr']
    
    def get_lr_history(self) -> List[float]:
        """Get history of learning rate values."""
        return self.lr_history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current stats for logging."""
        return {
            'step': self.step_count,
            'muon_lr': self.get_current_lr(0),
            'adamw_lr': self.get_current_lr(1) if len(self.optimizers) > 1 else None,
            'lr_history_len': len(self.lr_history),
            'warmup_active': self.step_count <= self.warmup_steps,
        }


def create_hypergradient_optimizers(
    model: torch.nn.Module,
    config,
    enable_hypergradient: bool = True,
):
    """
    Factory function to create optimizers with optional hypergradient wrapper.
    
    Args:
        model: The model to optimize
        config: BlueberryConfig with optimizer settings
        enable_hypergradient: Whether to wrap with HypergradientWrapper
        
    Returns:
        Either HypergradientWrapper or list of optimizers
    """
    from optimizers.muon import Muon
    
    # Separate parameters for Muon vs AdamW
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=config.muon_momentum)
    adamw_optimizer = torch.optim.AdamW(
        adamw_params,
        lr=config.adamw_lr,
        weight_decay=config.weight_decay,
        fused=torch.cuda.is_available()
    )
    
    optimizers = [muon_optimizer, adamw_optimizer]
    
    if enable_hypergradient and getattr(config, 'use_hypergradient', False):
        print("  🎯 Hypergradient LR adaptation enabled")
        wrapper = HypergradientWrapper(
            optimizers=optimizers,
            hyper_lr=getattr(config, 'hyper_lr', 1e-7),
            lr_min=getattr(config, 'lr_min', 1e-5),
            lr_max=getattr(config, 'lr_max', 0.1),
            warmup_steps=getattr(config, 'hg_warmup_steps', 50),
            loss_window=getattr(config, 'loss_window', 10),
            adapt_muon_only=True,  # Only adapt Muon by default
        )
        return wrapper
    
    return optimizers

