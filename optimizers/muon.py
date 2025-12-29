import torch
import torch.nn.functional as F

# coeffs for polar express 
coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]


@torch.compile()
def zeropower_polar_express(G: torch.Tensor, steps: int = 5):
    """Polar express orthonormalization (Newton-Schulz replacement)"""
    assert G.ndim >= 2
    assert steps <= len(coeffs_list)

    X = G.bfloat16()

    transpose_needed = G.size(-2) > G.size(-1)
    if transpose_needed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    coeffs = coeffs_list[:steps]
    for a, b, c in coeffs:
        A = X @ X.mT
        A2 = A @ A
        B = b * A + c * A2
        X = a * X + B @ X

    if transpose_needed:
        X = X.mT

    return X


class Muon(torch.optim.Optimizer):
    """
    DION2 - Sparse Muon with α-fraction row selection and error feedback.
    
    Algorithm (alpha < 1.0):
        1. M ← M + G                              (accumulate gradient into momentum)
        2. K ← Select_α(M)                        (select α-fraction of rows by ℓ₁ norm)
        3. O ← NewtonSchulz(M[K, :])              (orthonormalize only the submatrix)
        4. M[K, :] ← μ · M[K, :]                  (decay ONLY selected rows - error feedback)
        5. W[K, :] ← W[K, :] - η√(fan-out/fan-in) · O  (sparse parameter update)
    
    With alpha=1.0, falls back to original Muon with EMA momentum.
    """
    
    def __init__(self, params, lr=0.02, momentum=0.95, alpha=1.0, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            alpha = group["alpha"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]

                # DION2 sparse mode (alpha < 1.0)
                if p.ndim == 2 and alpha < 1.0:
                    # Step 1: Accumulate gradient (no decay yet)
                    buf.add_(g)
                    
                    num_rows = p.size(0)
                    k = max(1, int(alpha * num_rows))

                    # Step 2: Select α-fraction of rows by ℓ₁ norm
                    row_norms = buf.abs().sum(dim=1)
                    _, K = torch.topk(row_norms, k, largest=True, sorted=False)

                    # Step 3: Orthonormalize only selected rows
                    O = zeropower_polar_express(buf[K, :], steps=ns_steps)
                    O = O.to(p.dtype)

                    # Step 4: Decay ONLY selected rows (error feedback)
                    buf[K, :].mul_(mu)

                    # Step 5: Sparse parameter update
                    scale = max(1, p.size(0) / p.size(1)) ** 0.5
                    p[K, :].add_(O, alpha=-lr * scale)
                else:
                    # Original Muon behavior (alpha=1.0 or non-2D)
                    buf.lerp_(g, 1 - mu)
                    g = g.lerp_(buf, mu) if nesterov else buf
                    O = zeropower_polar_express(g, steps=ns_steps)
                    O = O.to(p.dtype)
                    scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
                    p.add_(O.view_as(p), alpha=-lr * scale)
