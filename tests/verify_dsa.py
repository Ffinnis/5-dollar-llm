
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.layers import LightningIndexer, MultiHeadAttention

def test_lightning_indexer():
    print("Testing LightningIndexer...")
    d_model = 128
    n_heads = 4
    idx_dim = 16
    B, T = 2, 10
    
    indexer = LightningIndexer(d_model, n_heads, idx_dim)
    x = torch.randn(B, T, d_model)
    
    scores = indexer(x, x)
    # Output should be [B, T, T]
    assert scores.shape == (B, T, T), f"Expected shape {(B, T, T)}, got {scores.shape}"
    print("LightningIndexer shape check passed.")

def test_dsa_attention():
    print("Testing DeepSeekSparseAttention (integrated)...")
    d_model = 128
    n_heads = 4
    max_seq_len = 20
    
    # Initialize with DSA
    attn = MultiHeadAttention(
        d_model=d_model, 
        n_heads=n_heads, 
        max_seq_len=max_seq_len,
        use_dsa=True,
        dsa_n_index_heads=2,
        dsa_index_dim=16,
        dsa_top_k=5
    )
    
    B, T = 2, 15
    x = torch.randn(B, T, d_model)
    
    # Forward pass
    out = attn(x)
    
    assert out.shape == (B, T, d_model), f"Expected {(B, T, d_model)}, got {out.shape}"
    print("DeepSeekSparseAttention forward pass passed.")
    
    # Check if masking actually happened? 
    # Hard to check black-box without hooks. 
    # But if it ran, dimensions matched.

if __name__ == "__main__":
    try:
        test_lightning_indexer()
        test_dsa_attention()
        print("\nAll DSA tests passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
