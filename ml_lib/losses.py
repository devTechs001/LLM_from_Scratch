import numpy as np
from typing import Callable, Union

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Mean Squared Error (L2 Loss):
    MSE = (1 / n) * sum((y_true - y_pred) ** 2)
    """
    return float(np.mean((y_true - y_pred) ** 2))

def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes the analytic derivative of MSE with respect to predictions y_pred:
    d(MSE) / d(y_pred) = (2 / n) * (y_pred - y_true)
    """
    n = y_true.size
    return (2.0 / n) * (y_pred - y_true)

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """
    Computes Huber Loss (Smooth L1 Loss).
    Combines quadratic MSE for small residuals with linear MAE for large residuals,
    providing robustness against outliers.
    For |r| <= delta: 0.5 * r^2
    For |r| > delta:  delta * (|r| - 0.5 * delta)
    """
    residual = np.abs(y_true - y_pred)
    quadratic = np.minimum(residual, delta)
    linear = residual - quadratic
    loss = 0.5 * (quadratic ** 2) + delta * linear
    return float(np.mean(loss))

def huber_gradient(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    Computes the derivative of Huber Loss with respect to predictions y_pred.
    For |r| <= delta: -(y_true - y_pred) / n = (y_pred - y_true) / n
    For |r| > delta:  -delta * sign(y_true - y_pred) / n = delta * sign(y_pred - y_true) / n
    """
    n = y_true.size
    diff = y_pred - y_true
    grad = np.where(np.abs(diff) <= delta, diff, delta * np.sign(diff))
    return grad / n

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Logistic sigmoid function: sigma(z) = 1 / (1 + exp(-z)).
    Clips z to [-250, 250] to prevent floating-point underflow/overflow.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Binary Cross-Entropy (Log Loss) for binary classification:
    BCE = - (1 / n) * sum(y * log(p) + (1 - y) * log(1 - p))
    Clips predictions to [1e-15, 1 - 1e-15] to prevent log(0) NaN.
    """
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))

def bce_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of BCE with respect to probability predictions y_pred:
    d(BCE)/d(p) = ((p - y) / (p * (1 - p))) / n
    """
    n = y_true.size
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    grad = (y_pred - y_true) / (y_pred * (1.0 - y_pred))
    return grad / n

def log_sum_exp(x: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    """
    Numerically stable computation of log(sum(exp(x))):
    LSE(x) = c + log(sum(exp(x - c))), where c = max(x).
    """
    c = np.max(x, axis=axis, keepdims=True)
    return c + np.log(np.sum(np.exp(x - c), axis=axis, keepdims=keepdims))

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Converts raw logits into a probability distribution summing to 1 along the last axis.
    Subtracts max along axis=-1 for numerical stability (prevents overflow in exp).
    Supports 1D, 2D, and multi-dimensional tensors.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)

def cross_entropy_loss(y_true_indices: np.ndarray, probs: np.ndarray) -> float:
    """
    Computes Categorical Cross-Entropy given integer class labels and probabilities.
    Loss = - (1 / n) * sum(log(p_{i, y_i}))
    """
    n_samples = len(y_true_indices)
    clipped_probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
    # Pick the predicted probability corresponding to each sample's true class
    correct_class_probs = clipped_probs[np.arange(n_samples), y_true_indices]
    return float(-np.mean(np.log(correct_class_probs)))

def cross_entropy_softmax_gradient(y_true_indices: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Computes combined analytic gradient of Cross-Entropy + Softmax with respect to logits.
    d(Loss) / d(logits_k) = (probs_k - 1(y == k)) / n_samples
    """
    n_samples = len(y_true_indices)
    grad = probs.copy()
    grad[np.arange(n_samples), y_true_indices] -= 1.0
    return grad / n_samples

def compute_numerical_gradient(
    loss_fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Computes the numerical gradient using two-sided central difference approximation:
    d L / d x_i ~ (L(x + eps * e_i) - L(x - eps * e_i)) / (2 * eps)
    Essential for gradient verification in testing against analytical gradients.
    """
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]

        x[idx] = orig_val + eps
        loss_pos = loss_fn(x)

        x[idx] = orig_val - eps
        loss_neg = loss_fn(x)

        x[idx] = orig_val
        grad[idx] = (loss_pos - loss_neg) / (2.0 * eps)
        it.iternext()
    return grad

def relative_error(grad_analytic: np.ndarray, grad_numeric: np.ndarray, eps: float = 1e-8) -> float:
    """
    Computes relative Frobenius difference between analytical and numerical gradients:
    rel_error = ||g_a - g_n|| / max(||g_a|| + ||g_n||, eps)
    Values < 1e-5 demonstrate mathematical correctness of analytical backprop.
    """
    diff = np.linalg.norm(grad_analytic - grad_numeric)
    scale = max(float(np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric)), eps)
    return float(diff / scale)
