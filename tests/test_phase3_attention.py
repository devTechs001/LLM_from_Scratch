import numpy as np
import pytest
import time
from ml_lib.llm import (
    MultiHeadAttention,
    precompute_freqs_cis,
    apply_rotary_emb,
)
from ml_lib.losses import compute_numerical_gradient, relative_error


def test_multi_head_attention_shapes_2d_and_3d():
    """Verify MultiHeadAttention processes 2D and 3D batched inputs correctly."""
    np.random.seed(42)
    B, T, D, H = 2, 5, 8, 2
    mha = MultiHeadAttention(embed_dim=D, num_heads=H)

    # 3D Batch
    x_3d = np.random.randn(B, T, D)
    out_3d = mha.forward(x_3d)
    assert out_3d.shape == (B, T, D)
    dx_3d = mha.backward(np.ones_like(out_3d))
    assert dx_3d.shape == (B, T, D)

    # 2D Single Sequence
    x_2d = np.random.randn(T, D)
    out_2d = mha.forward(x_2d)
    assert out_2d.shape == (T, D)
    dx_2d = mha.backward(np.ones_like(out_2d))
    assert dx_2d.shape == (T, D)


def test_multi_head_attention_gradient_check():
    """Verify MultiHeadAttention analytical input gradient dX against finite differences."""
    np.random.seed(42)
    B, T, D, H = 2, 3, 6, 2
    mha = MultiHeadAttention(embed_dim=D, num_heads=H)
    x = np.random.randn(B, T, D)
    d_out = np.random.randn(B, T, D)

    out = mha.forward(x)
    dx_analytic = mha.backward(d_out)

    def f_mha(x_val):
        out_val = mha.forward(x_val)
        return np.sum(out_val * d_out)

    dx_numeric = compute_numerical_gradient(f_mha, x.copy())
    err = relative_error(dx_analytic, dx_numeric)
    assert err < 1e-6, f"MHA input gradient mismatch! Relative error: {err}"


def test_causal_mask_future_invariance():
    """
    Mathematical causal guarantee:
    Changing future tokens (at position t >= k) must NOT affect predictions
    at past positions (t < k).
    """
    np.random.seed(42)
    T, D, H = 6, 8, 2
    mha = MultiHeadAttention(embed_dim=D, num_heads=H)

    seq_a = np.random.randn(T, D)
    # seq_b has identical tokens at indices 0, 1, 2, but completely different tokens at 3, 4, 5
    seq_b = seq_a.copy()
    seq_b[3:] = np.random.randn(3, D)

    out_a = mha.forward(seq_a)
    out_b = mha.forward(seq_b)

    # Outputs for tokens 0, 1, 2 must be EXACTLY identical between seq_a and seq_b
    assert np.allclose(out_a[:3], out_b[:3], atol=1e-12), "Future tokens leaked into past positions!"
    # Outputs for token 3+ must differ
    assert not np.allclose(out_a[3:], out_b[3:])


def test_rope_relative_position_property():
    """
    Verify Rotary Position Embedding (RoPE) relative position invariance:
    <R_m q, R_n k> = g(q, k, m - n).
    The inner product depends solely on relative distance, not absolute positions.
    """
    np.random.seed(42)
    head_dim = 8
    cos, sin = precompute_freqs_cis(head_dim, max_seq_len=64)

    q = np.random.randn(1, head_dim)
    k = np.random.randn(1, head_dim)

    # Pair 1: position m=4, n=1 (relative distance = 3)
    q_4 = apply_rotary_emb(q, cos[4:5], sin[4:5])
    k_1 = apply_rotary_emb(k, cos[1:2], sin[1:2])
    dot_1 = float(np.sum(q_4 * k_1))

    # Pair 2: position m=15, n=12 (relative distance = 3)
    q_15 = apply_rotary_emb(q, cos[15:16], sin[15:16])
    k_12 = apply_rotary_emb(k, cos[12:13], sin[12:13])
    dot_2 = float(np.sum(q_15 * k_12))

    assert np.isclose(dot_1, dot_2, atol=1e-10), f"RoPE relative position invariance violated! {dot_1} vs {dot_2}"


def test_mha_with_rope():
    """Verify MultiHeadAttention runs smoothly with RoPE enabled."""
    np.random.seed(42)
    mha_rope = MultiHeadAttention(embed_dim=8, num_heads=2, use_rope=True, max_seq_len=32)
    x = np.random.randn(2, 5, 8)
    out = mha_rope.forward(x)
    assert out.shape == (2, 5, 8)
    assert not np.isnan(out).any()


def test_kv_cache_exact_equivalence():
    """
    Verify that KV-cached autoregressive generation generates EXACTLY the same
    activations as full-sequence non-cached evaluation down to floating-point precision.
    """
    np.random.seed(42)
    mha = MultiHeadAttention(embed_dim=8, num_heads=2)

    prompt = np.random.randn(1, 4, 8)
    new_token = np.random.randn(1, 1, 8)
    full_sequence = np.concatenate([prompt, new_token], axis=1)  # Length 5

    # 1. Full non-cached evaluation
    full_out = mha.forward(full_sequence, use_cache=False)
    expected_new_tok = full_out[:, -1:, :]

    # 2. KV-cached step-by-step evaluation
    mha.reset_cache()
    _ = mha.forward(prompt, use_cache=True)
    cached_new_tok = mha.forward(new_token, use_cache=True)

    # Must be mathematically identical
    diff = np.max(np.abs(expected_new_tok - cached_new_tok))
    assert diff < 1e-12, f"KV cache produced mismatched output! Max diff: {diff}"


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3: SEQUENCE MODELING & MULTI-HEAD ATTENTION (VERIFICATION)")
    print("=" * 60)

    # 1. Multi-Head Attention Dimensions
    print("\n[Step 1] Multi-Head Attention Forward & Backward Pass:")
    mha = MultiHeadAttention(embed_dim=16, num_heads=4)
    x = np.random.randn(2, 5, 16)
    out = mha.forward(x)
    dx = mha.backward(np.ones_like(out))
    print(f"Batch Input:   {x.shape} (batch=2, seq_len=5, embed_dim=16)")
    print(f"Heads:         {mha.num_heads} heads x {mha.head_dim} head_dim")
    print(f"Output Shape:  {out.shape}")
    print(f"dX Grad Shape: {dx.shape}")

    # 2. RoPE Invariance Demo
    print("\n[Step 2] Rotary Position Embeddings (RoPE) Invariance Proof:")
    cos, sin = precompute_freqs_cis(8, max_seq_len=32)
    q_sample = np.random.randn(1, 8)
    k_sample = np.random.randn(1, 8)
    d1 = float(np.sum(apply_rotary_emb(q_sample, cos[3:4], sin[3:4]) * apply_rotary_emb(k_sample, cos[1:2], sin[1:2])))
    d2 = float(np.sum(apply_rotary_emb(q_sample, cos[8:9], sin[8:9]) * apply_rotary_emb(k_sample, cos[6:7], sin[6:7])))
    print(f"Inner product (pos 3 & 1, diff=2): {d1:.8f}")
    print(f"Inner product (pos 8 & 6, diff=2): {d2:.8f}")
    print(f"Absolute Difference:               {abs(d1 - d2):.2e} (Passed: {abs(d1 - d2) < 1e-10})")

    # 3. KV Cache Equivalence
    print("\n[Step 3] Key-Value (KV) Cache Mathematical Identity:")
    mha.reset_cache()
    p = np.random.randn(1, 4, 16)
    tok = np.random.randn(1, 1, 16)
    expected = mha.forward(np.concatenate([p, tok], axis=1), use_cache=False)[:, -1:, :]
    mha.reset_cache()
    _ = mha.forward(p, use_cache=True)
    actual = mha.forward(tok, use_cache=True)
    print(f"Max difference between cached & non-cached step: {np.max(np.abs(expected - actual)):.2e}")

    print("\n[Step 4] Running Pytest Suite:")
    import sys
    res = pytest.main(["-v", __file__])
    sys.exit(res)
