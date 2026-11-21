import numpy as np
import pytest
from ml_lib.llm import CharTokenizer, Embedding


def test_char_tokenizer_encode_decode():
    """Verify tokenizer maps characters to tokens and reconstructs original text."""
    corpus = "hello world! building an llm from scratch."
    tokenizer = CharTokenizer(corpus)
    assert tokenizer.vocab_size == 21

    prompt = "hello llm"
    ids = tokenizer.encode(prompt)
    assert len(ids) == len(prompt)
    decoded = tokenizer.decode(ids)
    assert decoded == prompt


def test_embedding_forward_and_backward():
    """Verify Embedding table lookup and gradient accumulation for duplicate tokens."""
    np.random.seed(42)
    vocab_size, embed_dim = 10, 4
    emb = Embedding(vocab_size, embed_dim)

    # Token 2 appears twice: [2, 5, 2]
    indices = np.array([2, 5, 2])
    out = emb.forward(indices)
    assert out.shape == (3, 4)
    assert np.array_equal(out[0], out[2])  # Same vector for same token ID

    d_out = np.ones((3, 4))
    initial_w2 = emb.weights[2].copy()
    initial_w5 = emb.weights[5].copy()

    lr = 0.1
    emb.backward(d_out, learning_rate=lr)

    # Gradient for token 2 should be accumulated twice (2 * lr * 1.0 = 0.2)
    assert np.allclose(emb.weights[2], initial_w2 - 0.2)
    # Gradient for token 5 should be accumulated once (1 * lr * 1.0 = 0.1)
    assert np.allclose(emb.weights[5], initial_w5 - 0.1)


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Sample training corpus
    corpus = "hello world! building an llm from scratch."
    tokenizer = CharTokenizer(corpus)
    print(f"Vocab Size: {tokenizer.vocab_size} unique characters")

    # 2. Encode a prompt into token IDs
    prompt = "hello llm"
    token_ids = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Encoded IDs: {token_ids}")
    print(f"Decoded back: '{tokenizer.decode(token_ids)}'")

    # 3. Create Token & Positional Embedding tables
    embed_dim = 8
    max_seq_len = 32

    token_embed_layer = Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=embed_dim)
    pos_embed_layer = Embedding(num_embeddings=max_seq_len, embedding_dim=embed_dim)

    # 4. Forward pass: combine Token Embeddings + Positional Embeddings
    seq_len = len(token_ids)
    positions = np.arange(seq_len)

    tok_emb = token_embed_layer.forward(token_ids)
    pos_emb = pos_embed_layer.forward(positions)
    x = tok_emb + pos_emb

    print(f"\nToken Embedding shape:      {tok_emb.shape}")
    print(f"Positional Embedding shape: {pos_emb.shape}")
    print(f"Combined Input Matrix shape:{x.shape} (seq_len={seq_len}, embed_dim={embed_dim})")
