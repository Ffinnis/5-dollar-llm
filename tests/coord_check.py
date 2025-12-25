"""
Coordinate Check for μP (Maximal Update Parameterization)
Verifies that μP is implemented correctly by checking activation stability across widths.

Usage:
    python tests/coord_check.py --widths 64,128,256,512 --steps 10

Success: Activation norms should be stable (roughly constant) across widths.
Failure: Norms explode or vanish with increasing width.
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from configs.llm_config import BlueberryConfig
from models.llm import MinimalLLM


def get_activation_norms(model, x, y):
    """Forward pass and collect activation statistics"""
    norms = {}
    
    # Register hooks to capture activations
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().float().abs().mean().item()
        return hook
    
    hooks = []
    
    # Hook into key layers
    hooks.append(model.token_embedding.register_forward_hook(make_hook('embedding')))
    
    for i, block in enumerate(model.transformer_blocks):
        hooks.append(block.attention.register_forward_hook(make_hook(f'attn_{i}')))
        hooks.append(block.feed_forward.register_forward_hook(make_hook(f'ffn_{i}')))
    
    hooks.append(model.norm.register_forward_hook(make_hook('final_norm')))
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
        norms['logits'] = logits.float().abs().mean().item()
    
    # Copy activations to norms
    norms.update(activations)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return norms


def run_coord_check(widths, steps, base_width, n_layers=4, batch_size=4, seq_len=128, device='cpu'):
    """
    Run coordinate check across different widths.
    
    Args:
        widths: List of d_model values to test
        steps: Number of training steps per width
        base_width: The μP base width
        n_layers: Number of layers (use small for speed)
        batch_size: Batch size for test
        seq_len: Sequence length for test
        device: 'cuda' or 'cpu'
    """
    print(f"\n{'='*60}")
    print(f"μP Coordinate Check")
    print(f"{'='*60}")
    print(f"Widths: {widths}")
    print(f"Base width: {base_width}")
    print(f"Steps: {steps}")
    print(f"Layers: {n_layers}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    results = {width: {'initial': {}, 'final': {}} for width in widths}
    
    for width in widths:
        print(f"\n--- Width: {width} ---")
        
        # Create config with μP enabled
        config = BlueberryConfig(
            d_model=width,
            n_heads=max(2, width // 64),  # At least 2 heads
            n_kv_heads=max(1, width // 128),
            d_ff=width * 4,
            n_layers=n_layers,
            vocab_size=1024,  # Small vocab for testing
            max_seq_len=seq_len,
            use_mup=True,
            mup_base_width=base_width,
            dropout=0.0,
        )
        
        # Create model
        torch.manual_seed(42)
        model = MinimalLLM(config).to(device)
        model.train()
        
        # Simple optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy data
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        # Initial norms
        norms_init = get_activation_norms(model, x, y)
        results[width]['initial'] = norms_init
        print(f"  Initial logits norm: {norms_init['logits']:.6f}")
        
        # Training loop
        for step in range(steps):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                y.view(-1)
            )
            loss.backward()
            optimizer.step()
        
        # Final norms
        norms_final = get_activation_norms(model, x, y)
        results[width]['final'] = norms_final
        print(f"  Final logits norm: {norms_final['logits']:.6f}")
        print(f"  Loss: {loss.item():.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Activation Norms After Training")
    print(f"{'='*60}")
    print(f"{'Width':<10} {'Logits':<12} {'Embedding':<12} {'Attn_0':<12} {'FFN_0':<12}")
    print("-" * 60)
    
    for width in widths:
        final = results[width]['final']
        print(f"{width:<10} {final.get('logits', 0):<12.6f} {final.get('embedding', 0):<12.6f} "
              f"{final.get('attn_0', 0):<12.6f} {final.get('ffn_0', 0):<12.6f}")
    
    # Check if norms are stable
    logit_norms = [results[w]['final']['logits'] for w in widths]
    ratio = max(logit_norms) / (min(logit_norms) + 1e-10)
    
    print(f"\n{'='*60}")
    if ratio < 5.0:
        print("✅ PASS: Logit norms are stable across widths (ratio < 5x)")
        print(f"   Max/Min ratio: {ratio:.2f}")
    else:
        print("❌ FAIL: Logit norms vary significantly across widths")
        print(f"   Max/Min ratio: {ratio:.2f}")
        print("   This indicates μP may not be correctly implemented.")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="μP Coordinate Check")
    parser.add_argument("--widths", type=str, default="64,128,256,512",
                        help="Comma-separated list of d_model widths to test")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of training steps per width")
    parser.add_argument("--base_width", type=int, default=64,
                        help="μP base width")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of layers (small for speed)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    args = parser.parse_args()
    
    widths = [int(w.strip()) for w in args.widths.split(",")]
    
    run_coord_check(
        widths=widths,
        steps=args.steps,
        base_width=args.base_width,
        n_layers=args.n_layers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
