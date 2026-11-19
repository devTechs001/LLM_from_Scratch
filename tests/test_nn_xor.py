import numpy as np
from ml_lib.nn import Linear, ReLU, Sigmoid, Sequential
from ml_lib.losses import mse_loss, mse_gradient

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
    # Forward pass
    y_pred = model.forward(X)
    loss = mse_loss(y, y_pred)
    
    # Backward pass
    d_loss = mse_gradient(y, y_pred)
    model.backward(d_loss, learning_rate)
    
    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

# 4. Final Predictions
final_preds = model.forward(X)
print("\nFinal XOR Predictions:")
for inp, target, pred in zip(X, y, final_preds):
    print(f"Input: {inp} | Target: {int(target.item())} | Predicted: {pred.item():.4f} ({round(pred.item())})")
