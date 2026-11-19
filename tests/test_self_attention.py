import numpy as np
from ml_lib.llm import SelfAttention

np.random.seed(42)

# 1. Create a fake embedded sequence: 4 tokens, each with embedding dimension 8
seq_len = 4
embed_dim = 8
head_dim = 8

x = np.random.randn(seq_len, embed_dim)

# 2. Initialize Causal Self-Attention and run forward pass
attn_layer = SelfAttention(embed_dim=embed_dim, head_dim=head_dim)
out = attn_layer.forward(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")

# 3. Inspect the causal attention weights (notice upper triangle is all 0.0!)
print("\nCausal Attention Weights (rows=tokens, cols=attended positions):")
print(np.round(attn_layer.cache["attn_weights"], 3))

# 4. Verify backward pass updates weights and returns valid input gradient
fake_d_out = np.ones_like(out)
d_x = attn_layer.backward(fake_d_out, learning_rate=0.01)
print(f"\nGradient d_x shape: {d_x.shape}")
print(f"Mean absolute gradient: {np.mean(np.abs(d_x)):.6f}")
