import numpy as np

def mse_loss(y_true: np.ndarry, y_pred: np.ndarray) -> float:
    """Computes Mean Squared Error"""
    return float(np.mean((y_true - y_pred) ** 2))

def mse_gradient(y_true: np.ndarray, y_pred : np.ndarray) -> np.ndarray:
    """Computes the derivatives of MSE with respect to preditions"""
    n = len(y_true)
    return (2.0 / n) * (y_pred - y_true)

def sigmoid(z: np.ndarray) -> ndarray:
    """Squashes real numbers into probabilities between 0 and 1"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Binary Cross-Entropy (Log Loss) for binary classification."""
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))

def softmax(logits: np.ndarray) -> np.ndarray:
    """Converts raw logits into a probability distribution summing to 1 per row."""
    # Subtract max per row for numerical stability (prevents exp overflow)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(y_true_indices: np.ndarray, probs: np.ndarray) -> float:
    """Computes Categorical Cross-Entropy given integer class labels and probabilities."""
    n_samples = len(y_true_indices)
    clipped_probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
    # Pick the predicted probability corresponding to each sample's true class
    correct_class_probs = clipped_probs[np.arange(n_samples), y_true_indices]
    return float(-np.mean(np.log(correct_class_probs)))

def cross_entropy_softmax_gradient(y_true_indices: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Computes combined gradient of Cross-Entropy + Softmax with respect to logits."""
    n_samples = len(y_true_indices)
    grad = probs.copy()
    grad[np.arange(n_samples), y_true_indices] -= 1.0
    return grad / n_samples
