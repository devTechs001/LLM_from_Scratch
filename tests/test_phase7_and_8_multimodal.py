import numpy as np
import pytest
from ml_lib.data import (
    TensorDataset,
    DataLoader,
    AudioSpectrogramTokenizer,
    ContinuousFeatureEncoder,
    random_horizontal_flip,
    random_erasing,
)
from ml_lib.multimodal import VisionLanguageModel
from ml_lib.llm import CharTokenizer
from ml_lib.losses import softmax, cross_entropy_loss, cross_entropy_softmax_gradient


def test_dataloader_shuffling_and_collation():
    """Verify DataLoader correctly batches, shuffles, and handles remainder batches."""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    dataset = TensorDataset(X, y)
    assert len(dataset) == 50

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    assert len(loader) == 4  # 16 + 16 + 16 + 2 = 50

    total_samples = 0
    for batch_x, batch_y in loader:
        total_samples += len(batch_x)
        assert len(batch_x) == len(batch_y)
    assert total_samples == 50


def test_audio_spectrogram_tokenizer():
    """Verify 1D continuous audio waveform decomposes into spectral frame tokens."""
    np.random.seed(42)
    B, L = 2, 64  # 64 time samples
    frame_size = 16
    hop_size = 8
    embed_dim = 24

    audio_tok = AudioSpectrogramTokenizer(frame_size=frame_size, hop_size=hop_size, embed_dim=embed_dim)
    waveforms = np.random.randn(B, L)
    tokens = audio_tok.forward(waveforms)

    expected_frames = (L - frame_size) // hop_size + 1  # (64 - 16) // 8 + 1 = 7 frames
    assert tokens.shape == (B, expected_frames, embed_dim)


def test_continuous_feature_encoder():
    """Verify tabular feature z-score normalization and token projection."""
    np.random.seed(42)
    features = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    enc = ContinuousFeatureEncoder(in_features=2, embed_dim=16)

    norm_x = enc.fit_normalize(features)
    assert np.allclose(np.mean(norm_x, axis=0), 0.0, atol=1e-6)
    assert np.allclose(np.std(norm_x, axis=0), 1.0, atol=1e-3)

    tokens = enc.forward(features)
    assert tokens.shape == (3, 1, 16)


def test_image_augmentations():
    """Verify pure NumPy image augmentations (flip and erasing)."""
    np.random.seed(42)
    imgs = np.ones((4, 1, 8, 8))
    imgs[:, :, :, :4] = 0.5  # Left half is 0.5, right half is 1.0

    # Flip with p=1.0: left half should become 1.0, right half 0.5
    flipped = random_horizontal_flip(imgs, p=1.0)
    assert np.allclose(flipped[:, :, :, :4], 1.0)
    assert np.allclose(flipped[:, :, :, 4:], 0.5)

    # Random erasing
    erased = random_erasing(imgs, p=1.0, erase_size=3)
    assert np.sum(erased == 0.0) > 0


def test_vision_language_model_forward_backward():
    """Verify VisionLanguageModel processes multimodal prefix and backpropagates to pixels."""
    np.random.seed(42)
    B, C, H, W = 2, 1, 8, 8
    vocab_size = 20
    text_len = 3

    vlm = VisionLanguageModel(
        vocab_size=vocab_size,
        image_size=8,
        patch_size=4,
        in_channels=1,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        use_mlp_projector=True
    )

    images = np.random.randn(B, C, H, W)
    text_ids = np.random.randint(0, vocab_size, size=(B, text_len))

    # Forward: N_patches = (8//4)*(8//4) = 4, Total seq = 4 + 3 = 7
    logits = vlm.forward(images, text_ids)
    assert logits.shape == (B, 7, vocab_size)

    # Backward pass returns exact pixel gradient shape
    d_logits = np.random.randn(*logits.shape)
    d_pixels = vlm.backward(d_logits, learning_rate=0.01)
    assert d_pixels.shape == images.shape
    assert not np.isnan(d_pixels).any()


def test_vision_language_model_generation():
    """Verify VisionLanguageModel generates text continuation conditioned on visual prefix."""
    np.random.seed(42)
    vlm = VisionLanguageModel(
        vocab_size=15,
        image_size=8,
        patch_size=4,
        in_channels=1,
        embed_dim=16,
        num_layers=1,
        num_heads=2
    )

    image = np.random.randn(1, 8, 8)
    prompt_ids = [1, 4]
    generated = vlm.generate(image, prompt_ids, max_new_tokens=5, temperature=0.0)

    assert len(generated) == len(prompt_ids) + 5
    assert generated[:2] == prompt_ids


def test_vlm_multimodal_training_convergence():
    """Verify VisionLanguageModel trains on multimodal (image, text) pairs."""
    np.random.seed(42)
    # Class 0 Image (horizontal line) -> Caption text 'horiz'
    # Class 1 Image (vertical line) -> Caption text 'verti'
    corpus = "horiz verti"
    tok = CharTokenizer(corpus)

    img_h = np.zeros((1, 1, 8, 8)); img_h[:, :, 0::2, :] = 1.0
    txt_h = tok.encode("horiz")

    vlm = VisionLanguageModel(
        vocab_size=tok.vocab_size,
        image_size=8,
        patch_size=4,
        in_channels=1,
        embed_dim=16,
        num_layers=1,
        num_heads=2
    )

    # Shifted next-token prediction targets
    # Sequence length is N_patches (4) + T (5) = 9
    # We predict the next text token at positions 4, 5, 6, 7
    initial_logits = vlm.forward(img_h, np.array([txt_h]))
    initial_loss = float(cross_entropy_loss(txt_h[1:], softmax(initial_logits[0, 4:8])))

    lr = 0.05
    for _ in range(40):
        logits = vlm.forward(img_h, np.array([txt_h]))
        probs = softmax(logits[0])
        # Target gradient on text positions 4..7
        d_logits = np.zeros_like(logits)
        d_logits[0, 4:8] = cross_entropy_softmax_gradient(txt_h[1:], probs[4:8])
        vlm.backward(d_logits, learning_rate=lr)

    final_logits = vlm.forward(img_h, np.array([txt_h]))
    final_loss = float(cross_entropy_loss(txt_h[1:], softmax(final_logits[0, 4:8])))
    assert final_loss < initial_loss * 0.3, f"Multimodal training failed to reduce loss! {final_loss} vs {initial_loss}"


if __name__ == "__main__":
    print("=" * 60)
    print("PHASES 7 & 8: MULTIMODAL DATASETS & VISION-LANGUAGE MODEL (VLM)")
    print("=" * 60)

    # 1. Audio Tokenizer Demo
    print("\n[Step 1] Audio Spectrogram Tokenizer Demo:")
    audio_tok = AudioSpectrogramTokenizer(frame_size=16, hop_size=8, embed_dim=32)
    fake_audio = np.random.randn(2, 64)
    a_tokens = audio_tok.forward(fake_audio)
    print(f"Waveform Input: {fake_audio.shape} (2 audio clips, 64 time samples)")
    print(f"Audio Tokens:   {a_tokens.shape} (2 clips, 7 spectral frames, 32 embed_dim)")

    # 2. VLM Prefix Injection Demo
    print("\n[Step 2] Vision-Language Model (LLaVA / VLM) Forward Demo:")
    vlm_demo = VisionLanguageModel(
        vocab_size=25,
        image_size=8,
        patch_size=4,
        in_channels=1,
        embed_dim=16,
        num_layers=2,
        num_heads=2
    )
    fake_img = np.random.randn(2, 1, 8, 8)
    prompt_tokens = np.array([[1, 2, 3], [4, 5, 6]])
    vlm_logits = vlm_demo.forward(fake_img, prompt_tokens)
    print(f"Image Input:      {fake_img.shape} (2 images, 8x8 pixels)")
    print(f"Text Prompt IDs:  {prompt_tokens.shape} (2 prompts, 3 token IDs)")
    print(f"Combined Output:  {vlm_logits.shape} (2 samples, 4 visual + 3 text tokens, 25 vocab logits)")

    # 3. VLM Image-Conditioned Generation Demo
    print("\n[Step 3] VLM Image-Conditioned Autoregressive Text Generation:")
    prompt = [2, 7]
    gen_tokens = vlm_demo.generate(fake_img[0], prompt_ids=prompt, max_new_tokens=8, temperature=0.7)
    print(f"Prompt IDs:    {prompt}")
    print(f"Generated IDs: {gen_tokens}")

    print("\n[Step 4] Running Pytest Suite:")
    import sys
    res = pytest.main(["-v", __file__])
    sys.exit(res)
