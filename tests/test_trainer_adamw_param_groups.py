import unittest

import torch

from configs.llm_config import BlueberryConfig
from models.llm import MinimalLLM
from training.trainer import setup_muon_optimizer


class TestTrainerAdamWParamGroups(unittest.TestCase):
    def test_adamw_no_decay_for_embeddings_and_norms_under_cautious(self):
        cfg = BlueberryConfig()
        cfg.adamw_weight_decay_mode = "cautious"
        cfg.weight_decay = 0.2

        model = MinimalLLM(cfg)
        muon_opt, adamw_opt = setup_muon_optimizer(model, cfg)

        # Blueberry-Nano routes almost all 2D matrices to Muon, leaving embeddings/norms for AdamW.
        # Ensure we don't apply (cautious) weight decay to embeddings/norms.
        self.assertIsInstance(adamw_opt, torch.optim.AdamW)

        self.assertEqual(len(adamw_opt.param_groups), 1)
        self.assertEqual(adamw_opt.param_groups[0]["weight_decay"], 0.0)

        tok = model.token_embedding.weight
        muon_params = {id(p) for g in muon_opt.param_groups for p in g["params"]}
        adamw_params = {id(p) for g in adamw_opt.param_groups for p in g["params"]}

        self.assertIn(id(tok), adamw_params)
        self.assertNotIn(id(tok), muon_params)


if __name__ == "__main__":
    unittest.main()

