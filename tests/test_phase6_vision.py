import numpy as np
import pytest
from ml_lib.vision import (
    PatchEmbedding,
    VisionTransformer,
    l2_normalize,
    ContrastiveLoss,
)
from ml_lib.losses import softmax, cross_entropy_loss, cross_entropy_softmax_gradient


def test_patch_embedding_shapes_and_inversion():
    """Verify PatchEmbedding transforms (B, C, H, W) -> (B, N, D) and inverts cleanly in backward."""
    B, C, H, W, P, D = 2, 3, 16, 16, 4, 32
    patch_embed = PatchEmbedding(image_size=H, patch_size=P, in_channels=C, embed_dim=D)

    images = np.random.randn(B, C, H, W)
    patches = patch_embed.forward(images)

    expected_num_patches = (16 // 4) * (16 // 4)  # 16 patches
    assert patches.shape == (B, expected_num_patches, D)

    # Backward pass returns exact pixel gradient shape
    d_out = np.random.randn(*patches.shape)
    d_images = patch_embed.backward(d_out, learning_rate=0.01)
    assert d_images.shape == images.shape


def test_vision_transformer_forward_backward():
    """Verify VisionTransformer full pipeline forward and backward through [CLS] token and blocks."""
    np.random.seed(42)
    B, C, H, W = 4, 1, 8, 8
    num_classes = 3
    vit = VisionTransformer(
        image_size=H,
        patch_size=4,
        in_channels=C,
        num_classes=num_classes,
        embed_dim=16,
        depth=1,
        num_heads=2,
    )

    images = np.random.randn(B, C, H, W)
    logits = vit.forward(images)
    assert logits.shape == (B, num_classes)

    d_logits = np.random.randn(B, num_classes)
    d_pixels = vit.backward(d_logits, learning_rate=0.01)
    assert d_pixels.shape == images.shape
    assert not np.isnan(d_pixels).any()


def test_vision_transformer_image_classification_training():
    """Verify VisionTransformer trains from scratch and separates synthetic image patterns."""
    np.random.seed(42)
    N = 25
    # Class 0: Horizontal stripes
    img_h = np.zeros((N, 1, 8, 8))
    img_h[:, :, 0::2, :] = 1.0
    y_h = np.zeros(N, dtype=int)

    # Class 1: Vertical stripes
    img_v = np.zeros((N, 1, 8, 8))
    img_v[:, :, :, 0::2] = 1.0
    y_v = np.ones(N, dtype=int)

    X = np.concatenate([img_h, img_v], axis=0) + np.random.randn(2 * N, 1, 8, 8) * 0.05
    y = np.concatenate([y_h, y_v])

    vit = VisionTransformer(
        image_size=8,
        patch_size=4,
        in_channels=1,
        num_classes=2,
        embed_dim=16,
        depth=1,
        num_heads=2,
    )

    lr = 0.05
    for epoch in range(60):
        logits = vit.forward(X)
        probs = softmax(logits)
        d_logits = cross_entropy_softmax_gradient(y, probs)
        vit.backward(d_logits, learning_rate=lr)

    final_probs = softmax(vit.forward(X))
    preds = np.argmax(final_probs, axis=1)
    acc = np.mean(preds == y) * 100
    assert acc >= 90.0, f"ViT failed to learn image patterns! Accuracy: {acc:.1f}%"


def test_l2_normalize_unit_hypersphere():
    """Verify l2_normalize projects vectors to unit Euclidean length."""
    v = np.array([[3.0, 4.0], [1.0, 1.0]])
    normed = l2_normalize(v)
    lengths = np.linalg.norm(normed, axis=-1)
    assert np.allclose(lengths, 1.0, atol=1e-6)


def test_contrastive_loss_clip():
    """Verify symmetric InfoNCE loss rewards aligned representations and penalizes orthogonal ones."""
    loss_fn = ContrastiveLoss(temperature=0.1)

    # 1. Perfectly aligned representations (I == T)
    I_perfect = np.eye(4)
    T_perfect = np.eye(4)
    loss_low, _ = loss_fn.forward(I_perfect, T_perfect)

    # 2. Random misaligned representations
    np.random.seed(42)
    I_rand = l2_normalize(np.random.randn(4, 8))
    T_rand = l2_normalize(np.random.randn(4, 8))
    loss_high, _ = loss_fn.forward(I_rand, T_rand)

    assert loss_low < loss_high, "Aligned representations should have lower contrastive loss!"


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 6: VISION TRANSFORMERS & MULTIMODAL SYSTEMS (VERIFICATION)")
    print("=" * 60)

    # 1. Patch Embedding Demo
    print("\n[Step 1] Image Patching Demo:")
    pe = PatchEmbedding(image_size=16, patch_size=4, in_channels=3, embed_dim=32)
    dummy_imgs = np.random.randn(2, 3, 16, 16)
    patch_tokens = pe.forward(dummy_imgs)
    print(f"Input Images: {dummy_imgs.shape} (batch=2, channels=3, 16x16 pixels)")
    print(f"Patch Tokens: {patch_tokens.shape} (batch=2, 16 patches, 32 embed_dim)")

    # 2. Vision Transformer Training Demo
    print("\n[Step 2] Training Vision Transformer on Visual Patterns:")
    N = 20
    h_imgs = np.zeros((N, 1, 8, 8)); h_imgs[:, :, 0::2, :] = 1.0
    v_imgs = np.zeros((N, 1, 8, 8)); v_imgs[:, :, :, 0::2] = 1.0
    X_train = np.concatenate([h_imgs, v_imgs], axis=0)
    y_train = np.concatenate([np.zeros(N, dtype=int), np.ones(N, dtype=int)])

    vit_demo = VisionTransformer(image_size=8, patch_size=4, in_channels=1, num_classes=2, embed_dim=16)
    for ep in range(1, 41):
        logits = vit_demo.forward(X_train)
        probs = softmax(logits)
        loss = cross_entropy_loss(y_train, probs)
        d_logits = cross_entropy_softmax_gradient(y_train, probs)
        vit_demo.backward(d_logits, learning_rate=0.05)
        if ep in [1, 20, 40]:
            print(f" Epoch {ep:2d} | Cross-Entropy Loss: {loss:.4f}")

    final_preds = np.argmax(softmax(vit_demo.forward(X_train)), axis=1)
    print(f"Training Accuracy: {np.mean(final_preds == y_train) * 100:.1f}%")

    # 3. Multimodal CLIP Contrastive Demo
    print("\n[Step 3] Multimodal CLIP Alignment Demo:")
    clip_loss = ContrastiveLoss(temperature=0.07)
    img_embs = l2_normalize(np.random.randn(4, 16))
    txt_embs = l2_normalize(img_embs + np.random.randn(4, 16) * 0.1) # correlated text
    c_loss, sim_matrix = clip_loss.forward(img_embs, txt_embs)
    print(f"Contrastive Loss: {c_loss:.4f}")
    print(f"Diagonal Sim (Positive pairs): {np.round(np.diag(sim_matrix * 0.07), 3)}")

    print("\n[Step 4] Running Pytest Suite:")
    import sys
    res = pytest.main(["-v", __file__])
    sys.exit(res)
