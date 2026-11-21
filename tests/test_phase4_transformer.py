import numpy as np
import pytest
from ml_lib.llm import (
    CharTokenizer,
    TransformerBlock,
    TransformerLM,
)
from ml_lib.losses import softmax, cross_entropy_loss, cross_entropy_softmax_gradient


def test_transformer_block_forward_backward_shapes():
    """Verify TransformerBlock processes 3D and 2D tensors and returns matching gradient shapes."""
    np.random.seed(42)
    B, T, D, H = 2, 4, 16, 2
    block = TransformerBlock(embed_dim=D, num_heads=H, norm_type="rmsnorm")

    # 3D
    x_3d = np.random.randn(B, T, D)
    out_3d = block.forward(x_3d)
    assert out_3d.shape == (B, T, D)
    dx_3d = block.backward(np.ones_like(out_3d), learning_rate=0.01)
    assert dx_3d.shape == (B, T, D)

    # 2D
    x_2d = np.random.randn(T, D)
    out_2d = block.forward(x_2d)
    assert out_2d.shape == (T, D)
    dx_2d = block.backward(np.ones_like(out_2d), learning_rate=0.01)
    assert dx_2d.shape == (T, D)


def test_residual_identity_gradient_highway():
    """
    Mathematical verification of the residual skip connection:
    y = x + F(x) => dy/dx = I + dF/dx
    Even if sublayer weights are scaled down, gradients flow cleanly without vanishing.
    """
    np.random.seed(42)
    D = 8
    block = TransformerBlock(embed_dim=D, num_heads=2)
    # Scale sublayer weights down to near zero
    block.attn.W_o *= 1e-4
    block.ffn_w2.weights *= 1e-4

    x = np.random.randn(1, 3, D)
    _ = block.forward(x)
    incoming_grad = np.random.randn(1, 3, D)
    dx = block.backward(incoming_grad)

    # With tiny sublayer weights, dx should be dominated by the identity skip: dx ~ incoming_grad
    assert np.allclose(dx, incoming_grad, rtol=0.05)


def test_transformer_lm_multi_layer_rope():
    """Verify multi-layer TransformerLM with RoPE and RMSNorm runs forward and backward."""
    np.random.seed(42)
    vocab_size = 25
    embed_dim = 16
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
        use_rope=True,
        norm_type="rmsnorm",
    )

    tokens = np.array([3, 7, 12, 18, 22])
    logits = model.forward(tokens)
    assert logits.shape == (5, vocab_size)

    d_logits = np.random.randn(*logits.shape)
    model.backward(d_logits, learning_rate=0.01)


def test_transformer_lm_kv_cache_generation():
    """Verify that multi-layer TransformerLM generates text with KV cache matching non-cached output."""
    np.random.seed(42)
    vocab_size = 20
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
        use_rope=True,
    )

    start_tokens = np.array([2, 5, 8])
    cached_gen = model.generate(start_tokens, max_new_tokens=6, temperature=0.0, use_cache=True)
    non_cached_gen = model.generate(start_tokens, max_new_tokens=6, temperature=0.0, use_cache=False)

    assert np.array_equal(cached_gen, non_cached_gen), "Cached generation diverged from non-cached generation!"


def test_transformer_pretraining_convergence():
    """Verify multi-layer TransformerLM pre-trains and achieves low cross-entropy loss."""
    np.random.seed(42)
    text = "attention is all you need for deep learning transformers from scratch"
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    X_train = data[:-1]
    y_train = data[1:]

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        num_layers=2,
        num_heads=2,
        max_seq_len=len(text),
        use_rope=True,
        norm_type="rmsnorm",
    )

    lr = 0.05
    initial_logits = model.forward(X_train)
    initial_loss = cross_entropy_loss(y_train, softmax(initial_logits))

    for epoch in range(150):
        logits = model.forward(X_train)
        probs = softmax(logits)
        d_logits = cross_entropy_softmax_gradient(y_train, probs)
        model.backward(d_logits, learning_rate=lr)

    final_logits = model.forward(X_train)
    final_loss = cross_entropy_loss(y_train, softmax(final_logits))

    assert final_loss < initial_loss * 0.25, f"Training failed to reduce loss significantly: {final_loss} vs {initial_loss}"


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4: TRANSFORMER ARCHITECTURE & PRE-TRAINING (VERIFICATION)")
    print("=" * 60)

    # 1. Training Corpus & Tokenization
    corpus = "attention is all you need for deep learning transformers from scratch"
    tokenizer = CharTokenizer(corpus)
    data = tokenizer.encode(corpus)
    X = data[:-1]
    y = data[1:]

    print(f"\n[Step 1] Pre-training Corpus:")
    print(f"Corpus:     '{corpus}'")
    print(f"Vocab Size: {tokenizer.vocab_size} unique characters")
    print(f"Sequence:   {len(data)} tokens")

    # 2. Multi-layer TransformerLM with Pre-LN & RoPE
    print("\n[Step 2] Initializing 2-Layer TransformerLM (Pre-RMSNorm + RoPE):")
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        num_layers=2,
        num_heads=2,
        max_seq_len=len(corpus),
        use_rope=True,
        norm_type="rmsnorm"
    )

    # 3. Pre-training Loop
    print("\n[Step 3] Pre-training Convergence:")
    lr = 0.05
    for epoch in range(1, 201):
        logits = model.forward(X)
        probs = softmax(logits)
        loss = cross_entropy_loss(y, probs)
        
        d_logits = cross_entropy_softmax_gradient(y, probs)
        model.backward(d_logits, learning_rate=lr)
        
        if epoch in [1, 50, 100, 150, 200]:
            print(f" Epoch {epoch:3d} | Cross-Entropy Loss: {loss:.4f}")

    # 4. KV-Cached Autoregressive Text Generation
    print("\n[Step 4] Fast KV-Cached Text Generation:")
    prompt = "attention"
    prompt_ids = tokenizer.encode(prompt)
    output_ids = model.generate(prompt_ids, max_new_tokens=45, temperature=0.0, use_cache=True)
    generated = tokenizer.decode(output_ids)
    print(f"Prompt:    '{prompt}'")
    print(f"Generated: '{generated}'")

    print("\n[Step 5] Running Pytest Suite:")
    import sys
    res = pytest.main(["-v", __file__])
    sys.exit(res)
