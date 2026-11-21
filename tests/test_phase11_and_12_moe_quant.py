import numpy as np
import pytest
from ml_lib.nn import Linear
from ml_lib.quantization import (
    quantize_symmetric_int8,
    dequantize_symmetric_int8,
    quantize_asymmetric_int8,
    dequantize_asymmetric_int8,
    quantize_symmetric_int4,
    dequantize_symmetric_int4,
    QuantizedLinear,
)
from ml_lib.moe import ExpertMLP, SparseMoE


def test_symmetric_int8_quantization():
    """Verify symmetric INT8 quantization bounds and low dequantization error."""
    np.random.seed(42)
    w = np.random.randn(64, 32).astype(np.float32)

    w_q, scale = quantize_symmetric_int8(w)
    assert w_q.dtype == np.int8
    assert np.all(w_q >= -127) and np.all(w_q <= 127)

    w_recon = dequantize_symmetric_int8(w_q, scale)
    mae = np.mean(np.abs(w - w_recon))
    # INT8 error is within max/127
    assert mae < 0.05


def test_asymmetric_int8_quantization():
    """Verify asymmetric INT8 quantization captures biased activations."""
    np.random.seed(42)
    # Biased positive range
    act = (np.random.rand(10, 20) * 10.0 + 5.0).astype(np.float32)

    data_q, scale, min_val = quantize_asymmetric_int8(act)
    assert data_q.dtype == np.uint8
    assert np.all(data_q >= 0) and np.all(data_q <= 255)

    data_recon = dequantize_asymmetric_int8(data_q, scale, min_val)
    mae = np.mean(np.abs(act - data_recon))
    assert mae < 0.1


def test_symmetric_int4_quantization():
    """Verify 4-bit quantization bounds and 8x compression ratio."""
    np.random.seed(42)
    w = np.random.randn(32, 16).astype(np.float32)

    w_q, scale = quantize_symmetric_int4(w)
    assert np.all(w_q >= -7) and np.all(w_q <= 7)

    w_recon = dequantize_symmetric_int4(w_q, scale)
    mae = np.mean(np.abs(w - w_recon))
    assert mae < 0.3


def test_quantized_linear_forward_accuracy():
    """Verify QuantizedLinear produces output close to full FP32 Linear."""
    np.random.seed(42)
    base = Linear(in_features=32, out_features=16)
    q_linear_8 = QuantizedLinear(base, bits=8)
    q_linear_4 = QuantizedLinear(base, bits=4)

    assert q_linear_8.compression_ratio == 4.0
    assert q_linear_4.compression_ratio == 8.0

    x = np.random.randn(4, 32)
    y_fp32 = base.forward(x)
    y_q8 = q_linear_8.forward(x)
    y_q4 = q_linear_4.forward(x)

    mae_8 = np.mean(np.abs(y_fp32 - y_q8))
    mae_4 = np.mean(np.abs(y_fp32 - y_q4))

    assert mae_8 < 0.1
    assert mae_4 < 0.5


def test_sparse_moe_forward_backward_shapes():
    """Verify SparseMoE routes 3D sequences and backpropagates correctly."""
    np.random.seed(42)
    B, T, D = 2, 4, 16
    d_ff = 32
    num_experts = 4
    top_k = 2

    moe = SparseMoE(d_model=D, d_ff=d_ff, num_experts=num_experts, top_k=top_k)
    x = np.random.randn(B, T, D)

    out = moe.forward(x)
    assert out.shape == (B, T, D)
    assert moe.aux_loss >= 0.0

    d_out = np.random.randn(B, T, D)
    d_x = moe.backward(d_out, learning_rate=0.01)

    assert d_x.shape == x.shape
    assert not np.isnan(d_x).any()


def test_moe_auxiliary_load_balancing_loss():
    """Verify auxiliary load-balancing loss increases when gating is severely skewed."""
    np.random.seed(42)
    moe = SparseMoE(d_model=8, d_ff=16, num_experts=4, top_k=1, aux_loss_coeff=1.0)

    # Uniform routing probabilities across 4 experts
    uniform_probs = np.ones((100, 4)) * 0.25
    uniform_topk = np.tile(np.array([[0], [1], [2], [3]]), (25, 1))
    loss_balanced = moe.compute_auxiliary_loss(uniform_probs, uniform_topk)

    # Completely skewed routing to expert 0
    skewed_probs = np.zeros((100, 4))
    skewed_probs[:, 0] = 1.0
    skewed_topk = np.zeros((100, 1), dtype=int)
    loss_skewed = moe.compute_auxiliary_loss(skewed_probs, skewed_topk)

    assert loss_skewed > loss_balanced


def test_sparse_moe_training_convergence():
    """Verify SparseMoE layer learns non-linear mapping and reduces loss."""
    np.random.seed(42)
    D = 8
    moe = SparseMoE(d_model=D, d_ff=16, num_experts=4, top_k=2)

    X = np.random.randn(12, D)
    Y_target = np.sin(X)  # Non-linear function

    initial_loss = np.mean((moe.forward(X) - Y_target) ** 2)

    lr = 0.02
    for _ in range(50):
        pred = moe.forward(X)
        d_out = 2.0 * (pred - Y_target) / len(X)
        moe.backward(d_out, learning_rate=lr)

    final_loss = np.mean((moe.forward(X) - Y_target) ** 2)
    assert final_loss < initial_loss * 0.5


if __name__ == "__main__":
    print("=" * 60)
    print("PHASES 11 & 12: QUANTIZATION (INT8/INT4) & SPARSE MIXTURE-OF-EXPERTS (MoE)")
    print("=" * 60)

    # 1. Quantization Demo
    print("\n[Step 1] Quantization & Model Compression Demo:")
    layer = Linear(256, 128)
    q8 = QuantizedLinear(layer, bits=8)
    q4 = QuantizedLinear(layer, bits=4)
    print(f"FP32 Weights Size:  {layer.weights.nbytes} bytes")
    print(f"INT8 Weights Size:  {q8.w_q.nbytes} bytes (Compression: {q8.compression_ratio:.1f}x)")
    print(f"INT4 Quantized:     {q4.w_q.nbytes} bytes (Compression: {q4.compression_ratio:.1f}x)")

    # 2. Sparse MoE Demo
    print("\n[Step 2] Sparse Mixture of Experts (MoE) Demo:")
    moe_layer = SparseMoE(d_model=32, d_ff=64, num_experts=4, top_k=2)
    tokens = np.random.randn(2, 6, 32)
    moe_out = moe_layer.forward(tokens)
    print(f"Input Shape:     {tokens.shape} (2 sequences, 6 tokens, 32 d_model)")
    print(f"Active Experts:  top_k=2 out of 4 total experts")
    print(f"Output Shape:    {moe_out.shape}")
    print(f"Aux Load Loss:   {moe_layer.aux_loss:.6f}")

    print("\n[Step 3] Running Pytest Suite:")
    import sys
    res = pytest.main(["-v", __file__])
    sys.exit(res)
