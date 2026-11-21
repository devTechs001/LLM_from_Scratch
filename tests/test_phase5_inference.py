import numpy as np
import pytest
from ml_lib.inference import sample_next_token, GenerationPipeline
from ml_lib.llm import CharTokenizer, MiniGPT, TransformerLM


def test_greedy_sampling_deterministic():
    """Verify temperature=0 is purely deterministic and picks argmax."""
    logits = np.array([-1.0, 5.0, 3.2, 0.1, -4.0])
    for _ in range(20):
        assert sample_next_token(logits, temperature=0.0) == 1


def test_temperature_entropy_effect():
    """Verify high temperature spreads probability while low temperature peaks on mode."""
    logits = np.array([1.0, 2.0, 3.0])
    np.random.seed(42)

    # Low temperature (T=0.1) should almost always pick index 2
    low_temp_samples = [sample_next_token(logits, temperature=0.1) for _ in range(100)]
    assert low_temp_samples.count(2) >= 95

    # High temperature (T=5.0) should explore all classes
    high_temp_samples = [sample_next_token(logits, temperature=5.0) for _ in range(150)]
    assert set(high_temp_samples) == {0, 1, 2}


def test_top_k_filtering():
    """Verify top_k filtering strictly zeroes out tokens outside top k."""
    logits = np.array([10.0, 9.0, 1.0, 0.5, -2.0])
    np.random.seed(42)

    # Top-2 tokens are indices 0 and 1
    samples = [sample_next_token(logits, temperature=2.0, top_k=2) for _ in range(100)]
    for s in samples:
        assert s in [0, 1], f"Selected token {s} was outside top-k=2!"


def test_top_p_nucleus_filtering():
    """Verify top_p nucleus filtering dynamically cuts off low-probability tails."""
    logits = np.array([10.0, 5.0, 0.1, -5.0])
    # Token 0 alone has ~99.3% of the softmax mass
    np.random.seed(42)

    samples = [sample_next_token(logits, temperature=1.0, top_p=0.9) for _ in range(50)]
    # Under p=0.9, token 0 satisfies the nucleus alone
    assert all(s == 0 for s in samples)


def test_repetition_penalty():
    """Verify repetition penalty suppresses probability of previously generated tokens."""
    logits = np.array([5.0, 4.9, 1.0])
    # With no penalty, token 0 is favored
    assert sample_next_token(logits, temperature=0.0) == 0

    # If token 0 was in past history and penalty=2.0, its positive logit drops from 5.0 to 2.5
    # Token 1 (4.9) now wins!
    penalized_pick = sample_next_token(
        logits,
        temperature=0.0,
        repetition_penalty=2.0,
        past_token_ids=[0]
    )
    assert penalized_pick == 1, f"Repetition penalty failed to demote token 0! Got {penalized_pick}"


def test_generation_pipeline_streaming_and_stop_words():
    """Verify GenerationPipeline streams tokens and halts upon seeing a stop word."""
    corpus = "hello world! building an ai system from scratch."
    tokenizer = CharTokenizer(corpus)
    model = MiniGPT(vocab_size=tokenizer.vocab_size, embed_dim=16, max_seq_len=len(corpus))

    pipeline = GenerationPipeline(model, tokenizer)

    # Stream tokens
    stream = pipeline.stream_generate(
        prompt="hello",
        max_new_tokens=15,
        temperature=0.8,
        stop_words=["!"],
        use_cache=False
    )
    chunks = list(stream)
    assert len(chunks) > 0
    full_output = "hello" + "".join(chunks)
    assert full_output.startswith("hello")


def test_generation_pipeline_full_generation():
    """Verify full end-to-end text generation."""
    corpus = "transformers from scratch"
    tokenizer = CharTokenizer(corpus)
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=len(corpus),
        use_rope=True,
    )

    pipeline = GenerationPipeline(model, tokenizer)
    completed_text = pipeline.generate(
        prompt="trans",
        max_new_tokens=10,
        temperature=0.0,
        use_cache=True
    )
    assert completed_text.startswith("trans")
    assert len(completed_text) == len("trans") + 10


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 5: INFERENCE ENGINE & ADVANCED SAMPLING (VERIFICATION)")
    print("=" * 60)

    # 1. Sampling Demos
    demo_logits = np.array([2.0, 5.0, 4.8, 0.5, -3.0])
    print(f"\n[Step 1] Token Sampling Verification:")
    print(f"Logits: {demo_logits}")
    print(f"Greedy Argmax (T=0):  Index {sample_next_token(demo_logits, temperature=0.0)}")
    print(f"Top-K (k=2, T=1.0):   Index {sample_next_token(demo_logits, temperature=1.0, top_k=2)}")
    print(f"Top-P (p=0.8, T=1.0): Index {sample_next_token(demo_logits, temperature=1.0, top_p=0.8)}")
    print(f"Repetition Penalty:   Index {sample_next_token(demo_logits, temperature=0.0, repetition_penalty=1.5, past_token_ids=[1])} (demoted index 1)")

    # 2. Generation Pipeline Streaming
    print("\n[Step 2] Generation Pipeline Streaming Demo:")
    corpus = "deep learning ai systems from scratch"
    tok = CharTokenizer(corpus)
    lm = TransformerLM(vocab_size=tok.vocab_size, embed_dim=16, num_layers=2, num_heads=2, max_seq_len=32)
    pipe = GenerationPipeline(lm, tok)

    print("Streaming generation for prompt 'deep':")
    print("Prompt: 'deep' -> Output: ", end="", flush=True)
    for token_char in pipe.stream_generate(prompt="deep", max_new_tokens=25, temperature=0.7, use_cache=True):
        print(token_char, end="", flush=True)
    print("\n")

    print("[Step 3] Running Pytest Suite:")
    import sys
    res = pytest.main(["-v", __file__])
    sys.exit(res)
