import torch
import torch.nn as nn
import math
from typing import Optional
from configs.llm_config import BlueberryConfig
from models.layers import TransformerBlock


class MinimalLLM(nn.Module):
    """Minimal dense LLM with optional μP (Maximal Update Parameterization)"""

    def __init__(self, config: BlueberryConfig):
        super().__init__()
        self.config = config
        
        # μP width multiplier: m_d = d_model / base_width
        self.mup_width_mult = config.d_model / config.mup_base_width if config.use_mup else 1.0

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
                    use_mup_attention=config.use_mup,
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

        # Initialize weights (μP-aware if enabled)
        if config.use_mup:
            self.apply(self._init_weights_mup)
        else:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Standard initialization (SP)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_weights_mup(self, module):
        """μP-aware initialization with width-scaled variance for hidden layers"""
        base_std = 0.02
        if isinstance(module, nn.Linear):
            # Hidden and output weights: variance scales as 1/m_d
            scaled_std = base_std / math.sqrt(self.mup_width_mult)
            torch.nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embeddings: use base std (no scaling for input layer in μP)
            torch.nn.init.normal_(module.weight, mean=0.0, std=base_std)

    def forward(self, x):
        # Token embeddings (with input multiplier)
        embed_mult = self.config.mup_input_mult if self.config.use_mup else 1.0
        x = self.token_embedding(x) * (math.sqrt(self.config.d_model) * embed_mult)
        x = self.position_dropout(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        
        # μP output scaling: scale logits by output_mult / m_d
        if self.config.use_mup:
            logits = logits * (self.config.mup_output_mult / self.mup_width_mult)

        return logits

