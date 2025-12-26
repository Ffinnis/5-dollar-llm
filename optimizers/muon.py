import torch
import torch.nn.functional as F

# coeffs for polar express 
# not pre_computed, same as modded-nanoGPT 
coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile()
def zeropower_polar_express(G:torch.Tensor, steps: int = 5,):
    """Polar express as replacement for Newton-Schulz iteration"""
    assert G.ndim >= 2
    assert steps <= len(coeffs_list)

    X = G.bfloat16()
    # X = G.half()

    transpose_needed = G.size(-2) > G.size(-1) # transposing if tall matrix
    if transpose_needed: 
        X = X.mT 
    
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7) # safety factor
    
    coeffs = coeffs_list[:steps]
    for a , b, c in coeffs:
        A = X @ X.mT 
        A2 = A @ A 
        B = b * A + c * A2
        X = a * X + B @ X  # Right-multiplication for left polar factor
    
    if transpose_needed: 
        X = X.mT 
    
    return X # orthogonalized 




class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Polar Express / Newton Schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
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
                g = zeropower_polar_express(g, steps=group["ns_steps"]) # steps are 5 for both ns and pe
                g = g.to(p.dtype)
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


class MuonAll(torch.optim.Optimizer):
    """
    MuonAll - Modified Muon that handles ALL parameters.
    
    For params with use_muon=True: applies Muon (momentum + Newton-Schulz)
    For params with use_muon=False: applies Adam-style update (momentum + RMSprop scaling)
    
    This removes the need for a separate AdamW optimizer.
    """
    def __init__(self, params, lr=0.02, lr_1d=0.006, momentum=0.95, beta2=0.999, eps=1e-8, 
                 weight_decay=0.0, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, lr_1d=lr_1d, momentum=momentum, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, nesterov=nesterov, ns_steps=ns_steps, use_muon=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            use_muon = group.get('use_muon', True)
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    if not use_muon:
                        state["v"] = torch.zeros_like(g)  # RMSprop variance
                        state["step"] = 0

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                
                # Apply Newton-Schulz only for muon params (2D matrices, not embedding/norm)
                if use_muon and p.ndim >= 2:
                    # Nesterov momentum
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_polar_express(g, steps=group["ns_steps"])
                    g = g.to(p.dtype)
                    p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
                else:
                    # Adam-style update for non-muon params
                    state["step"] += 1
                    v = state["v"]
                    
                    # Update second moment (RMSprop)
                    v.mul_(group["beta2"]).addcmul_(g, g, value=1 - group["beta2"])
                    
                    # Bias correction
                    bias_correction1 = 1 - group["momentum"] ** state["step"]
                    bias_correction2 = 1 - group["beta2"] ** state["step"]
                    
                    # Corrected moments
                    m_hat = buf / bias_correction1
                    v_hat = v / bias_correction2
                    
                    # Weight decay (decoupled, like AdamW)
                    if group["weight_decay"] > 0:
                        p.add_(p, alpha=-group["lr_1d"] * group["weight_decay"])
                    
                    # Adam update
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-group["lr_1d"])
