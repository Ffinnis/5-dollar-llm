import unittest

import torch

from optimizers.muon import Muon, zeropower_polar_express


class TestMuonWeightDecay(unittest.TestCase):
    def test_cautious_weight_decay_masks_entries(self):
        p = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float32))
        grad = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
        p.grad = grad.clone()

        opt = Muon(
            [p],
            lr=0.1,
            momentum=0.0,
            nesterov=False,
            ns_steps=0,
            weight_decay=1.0,
            weight_decay_mode="cautious",
        )

        p0 = p.detach().clone()
        u = zeropower_polar_express(grad, steps=0).to(p.dtype)

        mask = (u * p0) >= 0
        expected = p0.clone()
        expected.addcmul_(p0, mask, value=-0.1 * 1.0)
        expected.add_(u, alpha=-0.1)

        opt.step()

        torch.testing.assert_close(p.detach(), expected, rtol=0, atol=1e-6)

    def test_decoupled_weight_decay_affects_all_entries(self):
        p = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float32))
        grad = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
        p.grad = grad.clone()

        opt = Muon(
            [p],
            lr=0.1,
            momentum=0.0,
            nesterov=False,
            ns_steps=0,
            weight_decay=1.0,
            weight_decay_mode="decoupled",
        )

        p0 = p.detach().clone()
        u = zeropower_polar_express(grad, steps=0).to(p.dtype)

        expected = p0.clone()
        expected.add_(p0, alpha=-0.1 * 1.0)
        expected.add_(u, alpha=-0.1)

        opt.step()

        torch.testing.assert_close(p.detach(), expected, rtol=0, atol=1e-6)

    def test_weight_decay_scales_with_current_lr(self):
        p = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32))
        p.grad = torch.zeros_like(p)

        opt = Muon(
            [p],
            lr=0.1,
            momentum=0.0,
            nesterov=False,
            ns_steps=0,
            weight_decay=1.0,
            weight_decay_mode="decoupled",
        )

        opt.step()
        torch.testing.assert_close(p.detach(), torch.full_like(p, 0.9), rtol=0, atol=1e-6)

        p2 = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32))
        p2.grad = torch.zeros_like(p2)
        opt2 = Muon(
            [p2],
            lr=0.2,
            momentum=0.0,
            nesterov=False,
            ns_steps=0,
            weight_decay=1.0,
            weight_decay_mode="decoupled",
        )

        opt2.step()
        torch.testing.assert_close(p2.detach(), torch.full_like(p2, 0.8), rtol=0, atol=1e-6)

    def test_weight_decay_not_scaled_by_aspect_ratio(self):
        p = torch.nn.Parameter(torch.ones((4, 2), dtype=torch.float32))
        p.grad = torch.zeros_like(p)

        opt = Muon(
            [p],
            lr=0.1,
            momentum=0.0,
            nesterov=False,
            ns_steps=0,
            weight_decay=1.0,
            weight_decay_mode="decoupled",
        )

        opt.step()
        torch.testing.assert_close(p.detach(), torch.full_like(p, 0.9), rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
