import unittest

import torch

from optimizers.cautious_adamw import CautiousAdamW


class TestCautiousAdamWWeightDecay(unittest.TestCase):
    def test_cautious_weight_decay_masks_entries(self):
        p = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float32))
        grad = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
        p.grad = grad.clone()

        opt = CautiousAdamW(
            [p],
            lr=0.1,
            betas=(0.0, 0.0),
            eps=1e-8,
            weight_decay=1.0,
            weight_decay_mode="cautious",
        )

        p0 = p.detach().clone()
        m_t = grad.clone()
        mask = torch.signbit(m_t) == torch.signbit(p0)
        mask |= (m_t == 0) | (p0 == 0)
        expected = p0.clone()
        expected.addcmul_(p0, mask.to(dtype=p.dtype), value=-0.1 * 1.0)
        expected.add_(m_t / (m_t.abs() + 1e-8), alpha=-0.1)

        opt.step()

        torch.testing.assert_close(p.detach(), expected, rtol=0, atol=1e-6)

    def test_cautious_weight_decay_uses_first_moment(self):
        lr = 0.1
        wd = 1.0
        betas = (0.9, 0.0)
        eps = 1e6

        p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        p_ref = torch.nn.Parameter(p.detach().clone())

        opt = CautiousAdamW(
            [p],
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=wd,
            weight_decay_mode="cautious",
        )
        ref = torch.optim.AdamW([p_ref], lr=lr, betas=betas, eps=eps, weight_decay=0.0)

        # Step 1: positive grad establishes positive exp_avg.
        grad1 = torch.tensor([1.0], dtype=torch.float32)
        p.grad = grad1.clone()
        p_ref.grad = grad1.clone()

        with torch.no_grad():
            p0 = p_ref.detach().clone()
            m1 = grad1.mul(1.0 - betas[0])
            mask1 = torch.signbit(m1) == torch.signbit(p0)
            mask1 |= (m1 == 0) | (p0 == 0)
            p_ref.addcmul_(p0, mask1.to(dtype=p_ref.dtype), value=-lr * wd)
        ref.step()
        opt.step()

        # Step 2: negative grad, but momentum keeps m_t positive, so CWD should still apply.
        grad2 = torch.tensor([-0.1], dtype=torch.float32)
        p.grad = grad2.clone()
        p_ref.grad = grad2.clone()

        with torch.no_grad():
            exp_avg_prev = ref.state[p_ref]["exp_avg"]
            p0 = p_ref.detach().clone()
            m2 = exp_avg_prev.mul(betas[0]).add(grad2, alpha=1.0 - betas[0])
            mask2 = torch.signbit(m2) == torch.signbit(p0)
            mask2 |= (m2 == 0) | (p0 == 0)
            p_ref.addcmul_(p0, mask2.to(dtype=p_ref.dtype), value=-lr * wd)
        ref.step()
        opt.step()

        torch.testing.assert_close(p.detach(), p_ref.detach(), rtol=0, atol=1e-6)

    def test_cautious_weight_decay_scales_with_current_lr(self):
        p = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32))
        p.grad = torch.zeros_like(p)

        opt = CautiousAdamW(
            [p],
            lr=0.1,
            weight_decay=1.0,
            weight_decay_mode="cautious",
        )
        opt.step()
        torch.testing.assert_close(p.detach(), torch.full_like(p, 0.9), rtol=0, atol=1e-6)

        p2 = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32))
        p2.grad = torch.zeros_like(p2)
        opt2 = CautiousAdamW(
            [p2],
            lr=0.2,
            weight_decay=1.0,
            weight_decay_mode="cautious",
        )
        opt2.step()
        torch.testing.assert_close(p2.detach(), torch.full_like(p2, 0.8), rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

