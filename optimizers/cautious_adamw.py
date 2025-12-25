import torch


class CautiousAdamW(torch.optim.AdamW):
    """AdamW with optional Cautious Weight Decay (CWD).

    CWD applies decoupled weight decay only when the (raw) first moment and
    parameter have aligned sign: I(m_t ⊙ x_t >= 0).
    """

    _ALLOWED_WD_MODES = {"none", "decoupled", "cautious"}

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        *,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=False,
        fused=None,
        weight_decay_mode: str = "decoupled",
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )

        for group in self.param_groups:
            mode = group.get("weight_decay_mode", weight_decay_mode)
            if mode not in self._ALLOWED_WD_MODES:
                raise ValueError(
                    f"weight_decay_mode must be one of {sorted(self._ALLOWED_WD_MODES)}, got {mode!r}"
                )
            group["weight_decay_mode"] = mode

            if mode == "cautious":
                group["cautious_weight_decay"] = group.get("cautious_weight_decay", group["weight_decay"])
                group["weight_decay"] = 0.0
            elif mode == "none":
                group["weight_decay"] = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("weight_decay_mode") != "cautious":
                continue

            wd = group.get("cautious_weight_decay", 0.0)
            if wd == 0.0:
                continue

            lr = group["lr"]
            beta1, _ = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("CautiousAdamW does not support sparse gradients")

                grad = p.grad
                exp_avg = self.state[p].get("exp_avg")
                if exp_avg is None:
                    m_t = grad.mul(1.0 - beta1)
                else:
                    m_t = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)

                mask = (m_t * p) >= 0
                p.addcmul_(p, mask, value=-lr * wd)

        super().step(None)
        return loss
