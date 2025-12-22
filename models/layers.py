import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .components import SwiGLUFeedForward


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(
            dim=dim, max_seq_len=max_seq_len, base=10000
        )

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)


class LightningIndexer(nn.Module):
    def __init__(self, d_model: int, n_index_heads: int, index_dim: int):
        super().__init__()
        self.n_index_heads = n_index_heads
        self.index_dim = index_dim

        # Query-side projections
        self.q_proj = nn.Linear(d_model, n_index_heads * index_dim, bias=False)
        self.w_proj = nn.Linear(d_model, n_index_heads, bias=False)

        # Key-side projection
        self.k_proj = nn.Linear(d_model, index_dim, bias=False)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor):
        B, T_q, _ = x_q.size()
        _, T_k, _ = x_k.size()

        # q_I: [B, T_q, H_I, D_I]
        q_I = self.q_proj(x_q).view(B, T_q, self.n_index_heads, self.index_dim)
        
        # w_I: [B, T_q, H_I]
        w_I = self.w_proj(x_q)

        # k_I: [B, T_k, D_I]
        k_I = self.k_proj(x_k).view(B, T_k, 1, self.index_dim)

        # Compute term: ReLU( q_t,j . k_s )
        # [B, H_I, T_q, D_I]
        q_poly = q_I.permute(0, 2, 1, 3) 
        # [B, 1, D_I, T_k]
        k_poly = k_I.view(B, T_k, self.index_dim).permute(0, 2, 1).unsqueeze(1) 
        
        # [B, H_I, T_q, T_k]
        dot = torch.matmul(q_poly, k_poly)
        activated = F.relu(dot)

        # w_I: [B, H_I, T_q, 1]
        w_poly = w_I.permute(0, 2, 1).unsqueeze(-1)
        
        # Weighted sum across heads -> [B, T_q, T_k]
        index_scores = (activated * w_poly).sum(dim=1)
        return index_scores


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
        # DSA Config
        use_dsa: bool = False,
        dsa_n_index_heads: int = 4,
        dsa_index_dim: int = 64,
        dsa_top_k: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        
        # DSA
        self.use_dsa = use_dsa
        self.dsa_top_k = dsa_top_k
        if use_dsa:
            self.indexer = LightningIndexer(d_model, dsa_n_index_heads, dsa_index_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # Calculate queries, keys, and values
        Q = self.q_proj(x).reshape(
            batch_size, seq_len, self.n_heads, self.d_k
        )  # [B, T, H, D]
        K = self.k_proj(x).reshape(
            batch_size, seq_len, self.n_kv_heads, self.d_k
        )  # [B, T, KV_H, D]
        V = self.v_proj(x).reshape(
            batch_size, seq_len, self.n_kv_heads, self.d_k
        )  # [B, T, KV_H, D]

        # Apply RoPE
        Q = self.rotary(self.q_norm(Q))
        K = self.rotary(self.k_norm(K))

        # Repeat K/V for GQA if needed
        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        attn_mask = None
        if self.use_dsa:
            # Create mask based on Top-K
            # 1. Causal Masking
            # Calculate index scores: [B, T, T]
            index_scores = self.indexer(x, x)
            
            causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
            scores_for_topk = index_scores.masked_fill(~causal_mask, float('-inf'))
            
            # 2. Top-K Selection
            k_val = min(self.dsa_top_k, seq_len)
            top_scores, _ = torch.topk(scores_for_topk, k_val, dim=-1)
            
            # Threshold [B, T, 1]
            threshold = top_scores[:, :, -1].unsqueeze(-1)
            
            # Hard Mask [B, T, T] (True means keep)
            hard_mask = (scores_for_topk >= threshold) & causal_mask
            hard_mask_float = hard_mask.float()
            
            # 3. Straight-Through Estimator (STE)
            # Use safe scores for sigmoid to avoid NaN from -inf
            safe_scores = index_scores.clone()
            safe_scores = safe_scores.masked_fill(~causal_mask, -1e4) 
            
            # Scale to keep sigmoid gradients in useful range
            soft_mask = torch.sigmoid(safe_scores * 0.1) 
            ste_mask = (hard_mask_float - soft_mask).detach() + soft_mask
            
            # 4. Attention Bias
            # (1.0 - ste_mask) * large_neg
            large_neg = -1e9
            attn_mask = (1.0 - ste_mask) * large_neg
            
            attn_mask = attn_mask.unsqueeze(1) # [B, 1, T, T]
            
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True if attn_mask is None else False 
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        # attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        return self.w_o(attn_output)


class TransformerBlock(nn.Module):
    """Standard transformer block with dense feed-forward"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
        # DSA Config
        use_dsa: bool = False,
        dsa_n_index_heads: int = 4,
        dsa_index_dim: int = 64,
        dsa_top_k: int = 32,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model, n_heads, max_seq_len, dropout, n_kv_heads,
            use_dsa=use_dsa,
            dsa_n_index_heads=dsa_n_index_heads,
            dsa_index_dim=dsa_index_dim,
            dsa_top_k=dsa_top_k
        )
        self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x
