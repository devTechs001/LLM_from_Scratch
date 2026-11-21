import numpy as np
import pytest
from ml_lib.nn import (
    Linear,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
    LayerNorm,
    RMSNorm,
    Dropout,
    Sequential,
    xavier_normal,
    xavier_uniform,
    kaiming_normal,
    kaiming_uniform,
)
from ml_lib.losses import (
    compute_numerical_gradient,
    relative_error,
    mse_loss,
    mse_gradient,
    softmax,
    cross_entropy_loss,
    cross_entropy_softmax_gradient,
)


def test_linear_layer_2d_and_3d():
    """Verify Linear layer supports both 2D (batch, dim) and 3D (batch, seq, dim) tensors."""
    layer = Linear(in_features=6, out_features=4)

    # 2D test
    x_2d = np.random.randn(8, 6)
    out_2d = layer.forward(x_2d)
    assert out_2d.shape == (8, 4)
    dx_2d = layer.backward(np.ones_like(out_2d))
    assert dx_2d.shape == (8, 6)
    assert layer.d_weights.shape == (6, 4)
    assert layer.d_bias.shape == (1, 4)

    # 3D test (transformer sequence format)
    x_3d = np.random.randn(2, 5, 6)
    out_3d = layer.forward(x_3d)
    assert out_3d.shape == (2, 5, 4)
    dx_3d = layer.backward(np.ones_like(out_3d))
    assert dx_3d.shape == (2, 5, 6)


def test_linear_layer_gradient_check():
    """Verify Linear layer analytical gradients against finite differences."""
    np.random.seed(42)
    B, Din, Dout = 3, 4, 2
    layer = Linear(in_features=Din, out_features=Dout)
    X = np.random.randn(B, Din)
    d_out = np.random.randn(B, Dout)

    # Forward & backward pass
    layer.forward(X)
    dx_analytic = layer.backward(d_out)
    dw_analytic = layer.d_weights.copy()
    db_analytic = layer.d_bias.copy()

    # Numerical gradient check for X
    def f_X(x_val):
        out = np.dot(x_val, layer.weights) + layer.bias
        return np.sum(out * d_out)

    dx_numeric = compute_numerical_gradient(f_X, X.copy())
    err_x = relative_error(dx_analytic, dx_numeric)
    assert err_x < 1e-6, f"Linear dX mismatch: {err_x}"

    # Numerical gradient check for W
    def f_W(w_val):
        out = np.dot(X, w_val) + layer.bias
        return np.sum(out * d_out)

    dw_numeric = compute_numerical_gradient(f_W, layer.weights.copy())
    err_w = relative_error(dw_analytic, dw_numeric)
    assert err_w < 1e-6, f"Linear dW mismatch: {err_w}"


def test_activations_finite_difference_gradients():
    """Verify analytical gradients for GELU, SiLU, Tanh, LeakyReLU against finite differences."""
    np.random.seed(42)
    # Exclude exact 0.0 to avoid the non-differentiable subgradient kink of LeakyReLU/ReLU
    x = np.array([-2.5, -1.5, -0.7, -0.2, 0.3, 0.9, 1.8, 2.7]).reshape(2, 4)
    d_out = np.random.randn(2, 4)

    # 1. GELU
    gelu = GELU()
    gelu.forward(x)
    dx_ana = gelu.backward(d_out)
    dx_num = compute_numerical_gradient(lambda inp: np.sum(gelu.forward(inp) * d_out), x.copy())
    assert relative_error(dx_ana, dx_num) < 1e-6, "GELU gradient check failed!"

    # 2. SiLU (Swish)
    silu = SiLU()
    silu.forward(x)
    dx_ana = silu.backward(d_out)
    dx_num = compute_numerical_gradient(lambda inp: np.sum(silu.forward(inp) * d_out), x.copy())
    assert relative_error(dx_ana, dx_num) < 1e-6, "SiLU gradient check failed!"

    # 3. Tanh
    tanh = Tanh()
    tanh.forward(x)
    dx_ana = tanh.backward(d_out)
    dx_num = compute_numerical_gradient(lambda inp: np.sum(tanh.forward(inp) * d_out), x.copy())
    assert relative_error(dx_ana, dx_num) < 1e-6, "Tanh gradient check failed!"

    # 4. LeakyReLU
    lrelu = LeakyReLU(alpha=0.1)
    lrelu.forward(x)
    dx_ana = lrelu.backward(d_out)
    dx_num = compute_numerical_gradient(lambda inp: np.sum(lrelu.forward(inp) * d_out), x.copy())
    assert relative_error(dx_ana, dx_num) < 1e-6, "LeakyReLU gradient check failed!"


def test_layernorm_gradient_check():
    """Verify LayerNorm forward and exact backpropagation against numerical gradients."""
    np.random.seed(42)
    B, D = 4, 6
    ln = LayerNorm(normalized_shape=D)
    x = np.random.randn(B, D)
    d_out = np.random.randn(B, D)

    # Forward & backward
    out = ln.forward(x)
    assert np.allclose(np.mean(out, axis=-1), 0.0, atol=1e-5)
    assert np.allclose(np.var(out, axis=-1), 1.0, atol=1e-3)

    dx_ana = ln.backward(d_out)
    dgamma_ana = ln.d_gamma.copy()
    dbeta_ana = ln.d_beta.copy()

    # Numerical check for x
    def f_ln(x_val):
        mean = np.mean(x_val, axis=-1, keepdims=True)
        var = np.var(x_val, axis=-1, keepdims=True)
        x_hat = (x_val - mean) / np.sqrt(var + ln.eps)
        y = x_hat * ln.gamma + ln.beta
        return np.sum(y * d_out)

    dx_num = compute_numerical_gradient(f_ln, x.copy())
    assert relative_error(dx_ana, dx_num) < 1e-6, "LayerNorm dx mismatch!"


def test_rmsnorm_gradient_check():
    """Verify RMSNorm forward and backward against numerical gradients."""
    np.random.seed(42)
    B, D = 4, 6
    rms = RMSNorm(dim=D)
    x = np.random.randn(B, D)
    d_out = np.random.randn(B, D)

    out = rms.forward(x)
    dx_ana = rms.backward(d_out)

    def f_rms(x_val):
        val_rms = np.sqrt(np.mean(x_val ** 2, axis=-1, keepdims=True) + rms.eps)
        y = (x_val / val_rms) * rms.gamma
        return np.sum(y * d_out)

    dx_num = compute_numerical_gradient(f_rms, x.copy())
    assert relative_error(dx_ana, dx_num) < 1e-6, "RMSNorm dx mismatch!"


def test_dropout_train_vs_eval():
    """Verify Dropout applies inverted scaling during training and identity during evaluation."""
    np.random.seed(42)
    drop = Dropout(p=0.4)
    x = np.ones((100, 50))

    # In train mode: fraction of zeros should be approximately p
    drop.train()
    out_train = drop.forward(x)
    zero_ratio = np.mean(out_train == 0.0)
    assert np.isclose(zero_ratio, 0.4, atol=0.05)
    # Expected value must be preserved: E[out] == E[in] = 1.0
    assert np.isclose(np.mean(out_train), 1.0, atol=0.08)

    # In eval mode: identity
    drop.eval()
    out_eval = drop.forward(x)
    assert np.array_equal(out_eval, x)


def test_weight_initializations():
    """Verify Xavier and Kaiming weight initialization variance properties."""
    np.random.seed(42)
    fan_in, fan_out = 1000, 500

    # He / Kaiming normal: Var = 2 / fan_in
    w_he = kaiming_normal((fan_in, fan_out))
    expected_he_var = 2.0 / fan_in
    actual_he_var = np.var(w_he)
    assert np.isclose(actual_he_var, expected_he_var, rtol=0.1)

    # Xavier normal: Var = 2 / (fan_in + fan_out)
    w_xavier = xavier_normal((fan_in, fan_out))
    expected_xav_var = 2.0 / (fan_in + fan_out)
    actual_xav_var = np.var(w_xavier)
    assert np.isclose(actual_xav_var, expected_xav_var, rtol=0.1)


def test_deep_mlp_with_gelu_and_layernorm():
    """Train a deep modern MLP (Linear -> LayerNorm -> GELU -> Linear) on non-linear data."""
    np.random.seed(42)
    X = np.random.randn(60, 4)
    # Target is non-linear combination
    y = ((X[:, 0] * X[:, 1] + np.sin(X[:, 2])) > 0).astype(int)

    model = Sequential([
        Linear(4, 16),
        LayerNorm(16),
        GELU(),
        Linear(16, 2)
    ])

    lr = 0.1
    for epoch in range(200):
        logits = model.forward(X)
        probs = softmax(logits)
        d_logits = cross_entropy_softmax_gradient(y, probs)
        model.backward(d_logits, lr)

    final_probs = softmax(model.forward(X))
    acc = np.mean(np.argmax(final_probs, axis=1) == y)
    assert acc >= 0.90, f"Deep MLP failed to train! Final accuracy: {acc:.2f}"

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2: DEEP LEARNING PRIMITIVES & NON-LINEARITY (VERIFICATION)")
    print("=" * 60)

    # 1. 2D & 3D Linear Layer Demo
    print("\n[Step 1] Linear Layer Multidimensional Tensor Verification:")
    linear = Linear(in_features=8, out_features=4)
    x_seq = np.random.randn(2, 6, 8)
    out_seq = linear.forward(x_seq)
    dx_seq = linear.backward(np.ones_like(out_seq))
    print(f"Input Shape:  {x_seq.shape} (batch=2, seq_len=6, embed_dim=8)")
    print(f"Output Shape: {out_seq.shape} (batch=2, seq_len=6, out_dim=4)")
    print(f"Grad dX Shape: {dx_seq.shape}")

    # 2. Modern Activations (GELU & SiLU)
    print("\n[Step 2] Modern LLM Activations (GELU vs SiLU vs ReLU):")
    sample_inputs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"Inputs: {sample_inputs}")
    print(f"GELU:   {np.round(GELU().forward(sample_inputs), 4)}")
    print(f"SiLU:   {np.round(SiLU().forward(sample_inputs), 4)}")
    print(f"ReLU:   {np.round(ReLU().forward(sample_inputs), 4)}")

    # 3. LayerNorm vs RMSNorm
    print("\n[Step 3] Normalization: LayerNorm vs RMSNorm:")
    ln = LayerNorm(normalized_shape=4)
    rms = RMSNorm(dim=4)
    sample_vec = np.array([[10.0, 20.0, 30.0, 40.0]])
    ln_out = ln.forward(sample_vec)
    rms_out = rms.forward(sample_vec)
    print(f"Raw Input:        {sample_vec}")
    print(f"LayerNorm Output: {np.round(ln_out, 4)} (mean={np.mean(ln_out):.2f}, var={np.var(ln_out):.2f})")
    print(f"RMSNorm Output:   {np.round(rms_out, 4)}")

    print("\n[Step 4] Running Pytest Suite:")
    import sys
    res = pytest.main(["-v", __file__])
    sys.exit(res)
