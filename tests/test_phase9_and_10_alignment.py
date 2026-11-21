import numpy as np
import pytest
from ml_lib.nn import Linear
from ml_lib.peft import LoRALinear, apply_lora_to_linear, sft_masked_cross_entropy
from ml_lib.alignment import (
    RewardModel,
    pairwise_reward_loss,
    compute_sequence_log_probs,
    dpo_loss_and_gradients,
)


def test_lora_initialization_zero_delta():
    """Verify that at initialization LoRA B=0 produces exact equivalence with base layer."""
    np.random.seed(42)
    base = Linear(in_features=8, out_features=4)
    lora = LoRALinear(base, rank=2, alpha=4.0)

    x = np.random.randn(3, 8)
    base_out = base.forward(x)
    lora_out = lora.forward(x)

    assert np.allclose(base_out, lora_out, atol=1e-7)


def test_lora_forward_backward_shapes():
    """Verify LoRA forward and backward passes compute gradients for low-rank matrices."""
    np.random.seed(42)
    base = Linear(in_features=12, out_features=6)
    lora = LoRALinear(base, rank=3, alpha=6.0, freeze_base=True)

    x = np.random.randn(4, 5, 12)  # 3D tensor input (B, T, D)
    out = lora.forward(x)
    assert out.shape == (4, 5, 6)

    d_out = np.random.randn(4, 5, 6)
    d_x = lora.backward(d_out, learning_rate=0.01)

    assert d_x.shape == x.shape
    assert lora.grad_lora_A.shape == (3, 12)
    assert lora.grad_lora_B.shape == (6, 3)
    assert not np.isnan(d_x).any()


def test_lora_weight_merging_and_unmerging():
    """Verify merging LoRA weights into base Linear gives exact same output."""
    np.random.seed(42)
    base = Linear(in_features=10, out_features=5)
    lora = LoRALinear(base, rank=2, alpha=4.0)

    # Put non-zero values into B
    lora.lora_B = np.random.randn(5, 2) * 0.1

    x = np.random.randn(4, 10)
    unmerged_out = lora.forward(x)

    # Merge weights
    lora.merge()
    merged_out = lora.forward(x)
    assert np.allclose(unmerged_out, merged_out, atol=1e-7)

    # Unmerge weights
    lora.unmerge()
    reverted_out = lora.forward(x)
    assert np.allclose(unmerged_out, reverted_out, atol=1e-7)


def test_sft_masked_cross_entropy():
    """Verify SFT loss only penalizes target completion tokens, ignoring prompts."""
    np.random.seed(42)
    B, T, V = 2, 6, 10
    logits = np.random.randn(B, T, V)
    target_ids = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    # Sample 0 has prompt length 3 (indices 0, 1, 2 are prompt; target completion starts at 3)
    # Sample 1 has prompt length 4 (indices 0, 1, 2, 3 are prompt; target completion starts at 4)
    prompt_lengths = np.array([3, 4])

    loss, grad_logits = sft_masked_cross_entropy(logits, target_ids, prompt_lengths)
    assert loss > 0.0

    # Gradient at prompt prediction positions must be strictly 0
    # For sample 0, positions 0, 1 should be zero
    assert np.allclose(grad_logits[0, :2], 0.0)
    assert not np.allclose(grad_logits[0, 2:5], 0.0)

    # For sample 1, positions 0, 1, 2 should be zero
    assert np.allclose(grad_logits[1, :3], 0.0)
    assert not np.allclose(grad_logits[1, 3:5], 0.0)


def test_reward_model_pairwise_loss():
    """Verify Bradley-Terry reward model ranking loss and gradients."""
    np.random.seed(42)
    rm = RewardModel(in_features=16)

    h_chosen = np.random.randn(3, 4, 16)
    h_rejected = np.random.randn(3, 4, 16)

    r_w = rm.forward(h_chosen)
    r_l = rm.forward(h_rejected)

    assert r_w.shape == (3, 1)
    assert r_l.shape == (3, 1)

    loss, grad_w, grad_l = pairwise_reward_loss(r_w, r_l)
    assert loss > 0.0
    assert grad_w.shape == (3, 1)
    assert grad_l.shape == (3, 1)

    # Gradients must be opposite in sign
    assert np.allclose(grad_w, -grad_l)

    # Backprop to hidden states
    d_h_chosen = rm.backward(grad_w, learning_rate=0.01)
    assert d_h_chosen.shape == h_chosen.shape


def test_dpo_loss_and_gradients():
    """Verify Direct Preference Optimization (DPO) objective & analytic policy gradients."""
    np.random.seed(42)
    B, T, V = 2, 4, 8

    pi_chosen = np.random.randn(B, T, V)
    pi_rejected = np.random.randn(B, T, V)
    ref_chosen = np.random.randn(B, T, V)
    ref_rejected = np.random.randn(B, T, V)

    chosen_ids = np.array([[1, 2, 3, 4], [0, 2, 4, 6]])
    rejected_ids = np.array([[1, 3, 2, 5], [0, 1, 3, 5]])

    # Case 1: Neutral policy
    loss_neutral, grad_w, grad_l = dpo_loss_and_gradients(
        pi_chosen, pi_rejected, ref_chosen, ref_rejected,
        chosen_ids, rejected_ids, beta=0.1
    )
    assert loss_neutral > 0.0
    assert grad_w.shape == pi_chosen.shape
    assert grad_l.shape == pi_rejected.shape

    # For chosen targets, gradient pushes probability mass UP (so dL/dz < 0)
    for b in range(B):
        for t in range(T):
            target = chosen_ids[b, t]
            assert grad_w[b, t, target] < 0.0

    # For rejected targets, gradient pushes probability mass DOWN (so dL/dz > 0)
    for b in range(B):
        for t in range(T):
            target = rejected_ids[b, t]
            assert grad_l[b, t, target] > 0.0


def test_lora_finetuning_convergence():
    """Verify LoRA training converges and learns new weights while freezing base weights."""
    np.random.seed(42)
    base = Linear(in_features=8, out_features=4)
    base_w_original = base.weights.copy()

    lora = LoRALinear(base, rank=2, alpha=4.0, freeze_base=True)

    # Synthetic regression task
    X = np.random.randn(10, 8)
    Y_target = np.random.randn(10, 4)

    initial_loss = np.mean((lora.forward(X) - Y_target) ** 2)

    lr = 0.05
    for _ in range(60):
        pred = lora.forward(X)
        d_out = 2.0 * (pred - Y_target) / len(X)
        lora.backward(d_out, learning_rate=lr)

    final_loss = np.mean((lora.forward(X) - Y_target) ** 2)
    assert final_loss < initial_loss * 0.4
    # Ensure base weights were never modified
    assert np.allclose(base.weights, base_w_original)


if __name__ == "__main__":
    print("=" * 60)
    print("PHASES 9 & 10: PEFT / LoRA & ALIGNMENT (DPO & REWARD MODEL)")
    print("=" * 60)

    # 1. LoRA Demo
    print("\n[Step 1] LoRA Parameter-Efficient Adaptation Demo:")
    base_linear = Linear(64, 32)
    lora_linear = LoRALinear(base_linear, rank=4, alpha=8.0)
    base_params = base_linear.weights.size + (base_linear.bias.size if base_linear.bias is not None else 0)
    lora_params = lora_linear.lora_A.size + lora_linear.lora_B.size
    print(f"Base Parameters: {base_params}")
    print(f"LoRA Trainable Parameters (r=4): {lora_params} ({lora_params / base_params * 100:.1f}%)")

    # 2. DPO Demo
    print("\n[Step 2] Direct Preference Optimization (DPO) Demo:")
    B, T, V = 1, 3, 10
    pi_w = np.random.randn(B, T, V)
    pi_l = np.random.randn(B, T, V)
    ref_w = np.zeros_like(pi_w)
    ref_l = np.zeros_like(pi_l)
    w_ids = np.array([[2, 4, 6]])
    l_ids = np.array([[1, 3, 5]])
    dpo_loss, g_w, g_l = dpo_loss_and_gradients(pi_w, pi_l, ref_w, ref_l, w_ids, l_ids, beta=0.2)
    print(f"DPO Loss: {dpo_loss:.4f}")
    print(f"Chosen Logits Gradient Range:   [{g_w.min():.4f}, {g_w.max():.4f}]")
    print(f"Rejected Logits Gradient Range: [{g_l.min():.4f}, {g_l.max():.4f}]")

    print("\n[Step 3] Running Pytest Suite:")
    import sys
    res = pytest.main(["-v", __file__])
    sys.exit(res)
