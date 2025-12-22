"""
MuonSVD: Muon optimizer with exact SVD orthogonalization

This is a simplified version that replaces Newton-Schulz5 with exact SVD,
WITHOUT low-rank projection. This tests if SVD itself is beneficial.
"""

import torch


@torch.compile
def orth_via_svd(G: torch.Tensor) -> torch.Tensor:
    """
    Exact orthogonalization via SVD.
    Replaces Newton-Schulz5 with exact computation.
    """
    # Handle both orientations like Newton-Schulz does
    if G.size(-2) > G.size(-1):
        # Tall matrix - work with transpose
        U, S, Vt = torch.linalg.svd(G.mT, full_matrices=False)
        return (U @ Vt).mT
    else:
        U, S, Vt = torch.linalg.svd(G, full_matrices=False)
        return U @ Vt


class MuonSVD(torch.optim.Optimizer):
    """
    Muon with exact SVD orthogonalization instead of Newton-Schulz5.
    
    This is identical to Muon except for the orthogonalization method.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                
                # Use exact SVD instead of Newton-Schulz5
                g = orth_via_svd(g)
                
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
