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