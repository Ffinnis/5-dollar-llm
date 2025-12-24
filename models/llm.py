import torch
import torch.nn as nn
import math
from typing import Optional, Dict, List
from configs.llm_config import BlueberryConfig
from models.layers import TransformerBlock


class MinimalLLM(nn.Module):
    """Minimal dense LLM"""

    def __init__(self, config: BlueberryConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.max_seq_len,
                    config.dropout,
                    n_kv_heads=config.n_kv_heads,
                )
                for i in range(config.n_layers)
            ]
        )

        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        return logits

    def get_layer_param_mapping(self) -> Dict[int, List[nn.Parameter]]:
        """
        Returns mapping from layer index to list of 2D parameters (for Muon/Drop-Muon).
        
        Layer 0 = first transformer block, etc.
        Only includes 2D weight matrices (not embeddings, norms, or biases).
        These are the parameters that benefit from Newton-Schulz orthogonalization.
        
        Returns:
            Dict mapping layer_idx -> list of 2D parameters
        """
        layer_params: Dict[int, List[nn.Parameter]] = {}
        
        for layer_idx, block in enumerate(self.transformer_blocks):
            params: List[nn.Parameter] = []
            for name, param in block.named_parameters():
                # Only include 2D weight matrices that require grad
                if param.ndim == 2 and param.requires_grad:
                    params.append(param)
            layer_params[layer_idx] = params
        
        return layer_params

