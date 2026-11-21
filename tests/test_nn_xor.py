import numpy as np
import pytest
from ml_lib.nn import Linear, ReLU, Sigmoid, Sequential
from ml_lib.losses import mse_loss, mse_gradient


def test_xor_non_linear_convergence():
    """Verify 2-layer MLP solves non-linear XOR with high accuracy."""
    np.random.seed(42)

    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    model = Sequential([
        Linear(in_features=2, out_features=8),
        ReLU(),
        Linear(in_features=8, out_features=1),
        Sigmoid()
    ])

    learning_rate = 0.5
    for epoch in range(2000):
        y_pred = model.forward(X)
        d_loss = mse_gradient(y, y_pred)
        model.backward(d_loss, learning_rate)

    final_preds = model.forward(X)
    rounded_preds = np.round(final_preds)
    assert np.array_equal(rounded_preds, y), "MLP failed to classify XOR!"
    final_loss = mse_loss(y, final_preds)
    assert final_loss < 0.01, f"Final MSE loss {final_loss} too high for XOR convergence!"


def test_linear_layer_alone_fails_xor():
    """
    Mathematical proof test:
    A single linear layer (perceptron without non-linear hidden layer)
    cannot linearly separate the XOR problem (Minsky & Papert, 1969).
    Final accuracy cannot exceed 75% on XOR.
    """
    np.random.seed(42)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    linear_only = Sequential([
        Linear(in_features=2, out_features=1),
        Sigmoid()
    ])

    for _ in range(1000):
        preds = linear_only.forward(X)
        d_loss = mse_gradient(y, preds)
        linear_only.backward(d_loss, 0.5)

    preds = linear_only.forward(X)
    accuracy = np.mean(np.round(preds) == y)
    assert accuracy <= 0.75, "Single linear layer should NOT be able to solve XOR!"


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Classic non-linear XOR dataset
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    # 2. Build a 2-layer Neural Network: 2 -> 8 (ReLU) -> 1 (Sigmoid)
    model = Sequential([
        Linear(in_features=2, out_features=8),
        ReLU(),
        Linear(in_features=8, out_features=1),
        Sigmoid()
    ])

    # 3. Training loop
    learning_rate = 0.5
    epochs = 2000

    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss = mse_loss(y, y_pred)
        
        d_loss = mse_gradient(y, y_pred)
        model.backward(d_loss, learning_rate)
        
        if epoch % 400 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

    # 4. Final Predictions
    final_preds = model.forward(X)
    print("\nFinal XOR Predictions:")
    for inp, target, pred in zip(X, y, final_preds):
        print(f"Input: {inp} | Target: {int(target.item())} | Predicted: {pred.item():.4f} ({round(pred.item())})")
