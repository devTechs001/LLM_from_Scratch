import numpy as np
import pytest
from ml_lib.losses import (
    mse_loss,
    mse_gradient,
    huber_loss,
    huber_gradient,
    sigmoid,
    binary_cross_entropy,
    bce_gradient,
    log_sum_exp,
    softmax,
    cross_entropy_loss,
    cross_entropy_softmax_gradient,
    compute_numerical_gradient,
    relative_error,
)
from ml_lib.optimizers import (
    GradientDescent,
    SGDMomentum,
    RMSprop,
    Adam,
    AdamW,
    clip_grad_norm,
    clip_grad_value,
)


# =====================================================================
# PYTEST UNIT TESTS: Phase 1 Foundations & Mathematical Verification
# =====================================================================

def test_mse_loss_and_gradient():
    """Verify MSE loss computation and analytic derivative."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])
    # Residuals: [0.5, 0.5, -1.0], squared: [0.25, 0.25, 1.0], mean = 1.5 / 3 = 0.5
    loss = mse_loss(y_true, y_pred)
    assert np.isclose(loss, 0.5, atol=1e-7)

    # Analytic gradient: (2/n) * (y_pred - y_true) = (2/3) * [0.5, 0.5, -1.0]
    grad = mse_gradient(y_true, y_pred)
    expected_grad = (2.0 / 3.0) * np.array([0.5, 0.5, -1.0])
    assert np.allclose(grad, expected_grad, atol=1e-7)


def test_numerical_gradient_verification_mse():
    """
    Finite differences gradient check:
    Verify that analytic gradient matches numerical gradient to machine precision.
    """
    np.random.seed(42)
    y_true = np.random.randn(10)
    y_pred = np.random.randn(10)

    grad_analytic = mse_gradient(y_true, y_pred)
    grad_numeric = compute_numerical_gradient(lambda p: mse_loss(y_true, p), y_pred.copy())

    err = relative_error(grad_analytic, grad_numeric)
    assert err < 1e-6, f"MSE analytic gradient mismatch! Relative error: {err}"


def test_huber_loss_and_gradient():
    """Verify Huber loss (Smooth L1) transitions smoothly from quadratic to linear."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.5, 3.0])  # 0.5 <= delta=1.0 (quadratic), 3.0 > delta=1.0 (linear)
    delta = 1.0

    # For 0.5: 0.5 * 0.5^2 = 0.125
    # For 3.0: 1.0 * (3.0 - 0.5 * 1.0) = 2.5
    # Mean: (0.125 + 2.5) / 2 = 1.3125
    loss = huber_loss(y_true, y_pred, delta=delta)
    assert np.isclose(loss, 1.3125, atol=1e-7)

    # Gradient check
    grad_analytic = huber_gradient(y_true, y_pred, delta=delta)
    grad_numeric = compute_numerical_gradient(lambda p: huber_loss(y_true, p, delta=delta), y_pred.copy())
    err = relative_error(grad_analytic, grad_numeric)
    assert err < 1e-6, f"Huber gradient mismatch! Relative error: {err}"


def test_sigmoid_numerical_stability():
    """Verify Sigmoid produces exact [0, 1] bounds and handles extreme values without overflow."""
    extreme_inputs = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
    sig = sigmoid(extreme_inputs)
    assert not np.any(np.isnan(sig)), "Sigmoid produced NaNs on extreme inputs!"
    assert not np.any(np.isinf(sig)), "Sigmoid produced Infs on extreme inputs!"
    assert np.isclose(sig[0], 0.0, atol=1e-7)
    assert np.isclose(sig[2], 0.5, atol=1e-7)
    assert np.isclose(sig[4], 1.0, atol=1e-7)


def test_binary_cross_entropy_stability():
    """Verify BCE handles edge probabilities (0.0, 1.0) without NaN/Inf via clipping."""
    y_true = np.array([1.0, 0.0])
    y_pred_extreme = np.array([0.0, 1.0])  # Completely wrong, would trigger -log(0) without clip
    loss = binary_cross_entropy(y_true, y_pred_extreme)
    assert not np.isnan(loss)
    assert not np.isinf(loss)
    assert loss > 30.0  # High loss, but finite and stable


def test_log_sum_exp_stability():
    """Verify Log-Sum-Exp prevents overflow for large logits."""
    large_logits = np.array([[1000.0, 1001.0, 1002.0]])
    lse = log_sum_exp(large_logits)
    # math: 1002 + log(exp(-2) + exp(-1) + 1) ~ 1002 + 0.4076
    assert not np.isnan(lse).any()
    assert not np.isinf(lse).any()
    expected = 1002.0 + np.log(np.exp(-2.0) + np.exp(-1.0) + 1.0)
    assert np.isclose(lse[0, 0], expected, atol=1e-5)


def test_softmax_and_cross_entropy_gradient():
    """
    Verify softmax numerical stability and combined cross-entropy gradient:
    d L / d z = p - y
    """
    np.random.seed(42)
    logits = np.random.randn(4, 5) * 2.0
    probs = softmax(logits)

    # Probabilities must be positive and sum to 1 per row
    assert np.all(probs >= 0.0)
    assert np.allclose(np.sum(probs, axis=-1), 1.0, atol=1e-6)

    # Test combined gradient against finite differences
    y_true = np.array([0, 2, 4, 1])
    analytic_grad = cross_entropy_softmax_gradient(y_true, probs)

    def loss_given_logits(z):
        p = softmax(z)
        return cross_entropy_loss(y_true, p)

    numeric_grad = compute_numerical_gradient(loss_given_logits, logits.copy())
    err = relative_error(analytic_grad, numeric_grad)
    assert err < 1e-6, f"Softmax + Cross-Entropy gradient mismatch! Rel error: {err}"


def test_gradient_descent_convergence():
    """Verify vanilla Gradient Descent minimizes quadratic loss."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    opt = GradientDescent(learning_rate=0.1)

    loss_start = mse_loss(y_true, y_pred)
    for _ in range(80):
        grad = mse_gradient(y_true, y_pred)
        y_pred = opt.step(y_pred, grad)

    loss_end = mse_loss(y_true, y_pred)
    assert loss_end < loss_start * 1e-3, "Gradient descent failed to converge!"
    assert np.allclose(y_pred, y_true, atol=0.2)


def test_sgd_momentum_convergence():
    """Verify SGD with Momentum converges and dampens oscillations."""
    y_true = np.array([5.0, -5.0])
    y_pred = np.array([0.0, 0.0])
    opt = SGDMomentum(learning_rate=0.05, momentum=0.8)

    for _ in range(80):
        grad = mse_gradient(y_true, y_pred)
        y_pred = opt.step(y_pred, grad, param_id="y_pred")

    assert np.allclose(y_pred, y_true, atol=0.2)


def test_adam_and_adamw_convergence():
    """Verify Adam and AdamW converge on quadratic problem with adaptive step sizing."""
    y_true = np.array([12.0, -8.0, 4.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    opt = Adam(learning_rate=0.5)

    for _ in range(100):
        grad = mse_gradient(y_true, y_pred)
        y_pred = opt.step(y_pred, grad, param_id="test_param")

    assert np.allclose(y_pred, y_true, atol=0.2)

    # Verify AdamW weight decay penalizes parameter magnitude
    param_w = np.array([10.0, 10.0])
    zero_grad = np.array([0.0, 0.0])
    adamw = AdamW(learning_rate=0.1, weight_decay=0.1)
    param_decayed = adamw.step(param_w, zero_grad, param_id="w")
    # With zero gradient, param_decayed should be smaller than param_w due to weight decay
    assert np.all(param_decayed < param_w)


def test_gradient_clipping():
    """Verify norm clipping prevents exploding gradient values."""
    exploding_grad = np.array([100.0, 200.0, 300.0])
    max_norm = 5.0

    orig_norm = clip_grad_norm(exploding_grad, max_norm=max_norm)
    clipped_norm = np.linalg.norm(exploding_grad)
    assert np.isclose(orig_norm, np.sqrt(100**2 + 200**2 + 300**2))
    assert np.isclose(clipped_norm, max_norm, atol=1e-5)


# =====================================================================
# STANDALONE DEMO EXECUTION (Backwards-compatible Phase 1 check)
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1: FOUNDATIONS OF ML & OPTIMIZATION (VERIFICATION)")
    print("=" * 60)

    # 1. Classic target & initial prediction demonstration
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([2.0, 2.0, 2.0])

    optimizer = GradientDescent(learning_rate=0.1)

    print("\n[Step 1] Initial State:")
    print(f"Target:      {y_true}")
    print(f"Prediction:  {y_pred}")
    print(f"Initial MSE: {mse_loss(y_true, y_pred):.4f}")

    # Run 10 optimization steps directly on y_pred
    for step in range(1, 11):
        grad = mse_gradient(y_true, y_pred)
        y_pred = optimizer.step(y_pred, grad)
        if step in [1, 5, 10]:
            print(f" Step {step:2d} -> Loss: {mse_loss(y_true, y_pred):.4f} | Pred: {np.round(y_pred, 2)}")

    print(f"\nFinal MSE Loss after 10 steps: {round(mse_loss(y_true, y_pred))}")
    print(f"Final predictions: {np.round(y_pred, 2)}")

    # 2. Mathematical Finite-Differences Verification Check
    print("\n[Step 2] Finite Difference Analytic Gradient Verification:")
    sample_true = np.array([3.0, 5.0, 7.0])
    sample_pred = np.array([1.0, 2.0, 4.0])
    g_analytic = mse_gradient(sample_true, sample_pred)
    g_numeric = compute_numerical_gradient(lambda p: mse_loss(sample_true, p), sample_pred.copy())
    rel_err = relative_error(g_analytic, g_numeric)
    print(f"Analytic Gradient: {np.round(g_analytic, 6)}")
    print(f"Numeric Gradient:  {np.round(g_numeric, 6)}")
    print(f"Relative Error:    {rel_err:.2e} (Passed: {rel_err < 1e-6})")

    # 3. AdamW Optimizer Verification
    print("\n[Step 3] Advanced Optimizers: AdamW Convergence Check:")
    target_weights = np.array([5.0, -3.0])
    weights = np.array([0.0, 0.0])
    adamw = AdamW(learning_rate=0.2, weight_decay=0.01)
    for _ in range(50):
        g = mse_gradient(target_weights, weights)
        weights = adamw.step(weights, g, param_id="w")
    print(f"Target Weights:   {target_weights}")
    print(f"Learned Weights:  {np.round(weights, 2)}")

    print("\n[Step 4] Running Pytest Suite:")
    import sys
    res = pytest.main(["-v", __file__])
    sys.exit(res)
