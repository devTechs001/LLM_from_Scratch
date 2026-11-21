import numpy as np
import pytest
from ml_lib.llm import SelfAttention


def test_self_attention_causal_mask():
    """Verify that causal attention mask strictly prevents future token leakage."""
    np.random.seed(42)
    seq_len = 4
    embed_dim = 8
    head_dim = 8

    x = np.random.randn(seq_len, embed_dim)
    attn_layer = SelfAttention(embed_dim=embed_dim, head_dim=head_dim)
    out = attn_layer.forward(x)

    attn_weights = attn_layer.cache["attn_weights"]

    # In causal attention, weights above main diagonal must be exactly 0.0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert attn_weights[i, j] == 0.0, f"Future token leakage at [{i}, {j}]: {attn_weights[i, j]}"

    # Each row must sum to 1.0
    assert np.allclose(np.sum(attn_weights, axis=-1), 1.0)


def test_self_attention_backward_pass():
    """Verify backpropagation through single-head causal self-attention."""
    np.random.seed(42)
    seq_len = 3
    embed_dim = 6
    head_dim = 6

    x = np.random.randn(seq_len, embed_dim)
    attn = SelfAttention(embed_dim=embed_dim, head_dim=head_dim)
    out = attn.forward(x)

    d_out = np.random.randn(*out.shape)
    w_q_before = attn.W_q.copy()
    dx = attn.backward(d_out, learning_rate=0.01)

    assert dx.shape == x.shape
    assert not np.array_equal(attn.W_q, w_q_before)
    assert not np.isnan(dx).any()


if __name__ == "__main__":
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
