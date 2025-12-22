
class LightningIndexer(nn.Module):
    def __init__(self, d_model: int, n_index_heads: int, index_dim: int):
        super().__init__()
        self.n_index_heads = n_index_heads
        self.index_dim = index_dim

        # Query-side projections: produces q_I (heads, dim) and w_I (heads, 1) per token
        self.q_proj = nn.Linear(d_model, n_index_heads * index_dim, bias=False)
        self.w_proj = nn.Linear(d_model, n_index_heads, bias=False)

        # Key-side projection: produces k_I (dim) per token
        # Note: In the paper "k_I_s is derived from the preceding token h_s".
        # It seems k_I is shared across index heads or we need k indices?
        # Eq (1): sum_j ( w_t,j * ReLU( q_t,j * k_s ) )
        # Here k_s has no index j, so k_s is common. But q_t,j is R^d_I.
        # So k_s must be R^d_I.
        self.k_proj = nn.Linear(d_model, index_dim, bias=False)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor):
        """
        x_q: [batch, query_len, d_model]
        x_k: [batch, key_len, d_model]
        """
        B, T_q, _ = x_q.size()
        _, T_k, _ = x_k.size()

        # Project
        # q_I: [B, T_q, H_I, D_I]
        q_I = self.q_proj(x_q).view(B, T_q, self.n_index_heads, self.index_dim)
        
        # w_I: [B, T_q, H_I]
        w_I = self.w_proj(x_q)

        # k_I: [B, T_k, D_I] -> reshape to [B, T_k, 1, D_I] for broadcasting
        k_I = self.k_proj(x_k).view(B, T_k, 1, self.index_dim)

        # Compute term: ReLU( q_t,j . k_s )
        # We need dot product between q_I and k_I for each head.
        # q_I: [B, T_q, H_I, D_I]
        # k_I: [B, T_k,  1,  D_I]
        # We want [B, T_q, T_k, H_I] output before sum.
        # Let's permute k to use matmul?
        # Alternatively:
        # q * k^T
        # q_I: [B, T_q, H_I, D_I] -> [B, H_I, T_q, D_I]
        # k_I: [B, T_k, D_I] -> [B, 1, D_I, T_k] (transpose last two)
        
        q_poly = q_I.permute(0, 2, 1, 3) # [B, H_I, T_q, D_I]
        k_poly = k_I.view(B, T_k, self.index_dim).permute(0, 2, 1).unsqueeze(1) # [B, 1, D_I, T_k]
        
        # [B, H_I, T_q, T_k]
        dot = torch.matmul(q_poly, k_poly)
        
        # Apply ReLU
        activated = F.relu(dot)

        # Weighting
        # w_I: [B, T_q, H_I] -> [B, H_I, T_q, 1]
        w_poly = w_I.permute(0, 2, 1).unsqueeze(-1)
        
        # Weighted sum across heads
        # sum_j ( w_j * activated_j )
        # [B, H_I, T_q, T_k] * [B, H_I, T_q, 1] -> [B, H_I, T_q, T_k]
        weighted = activated * w_poly
        
        # Sum across heads -> [B, T_q, T_k]
        index_scores = weighted.sum(dim=1)
        
        return index_scores
