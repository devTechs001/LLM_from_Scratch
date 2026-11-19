import numpy as np
from ml_lib.linear_model import LinearRegression

# 1. Generate synthetic data: y = 3*x1 + 5*x2 + 2
np.random.seed(42)
X = np.random.rand(100, 2)
true_weights = np.array([3.0, 5.0])
true_bias = 2.0
y = np.dot(X, true_weights) + true_bias

# 2. Initialize and train our custom model (ensure epochs=1000)
model = LinearRegression(learning_rate=0.1, epochs=1000)
model.fit(X, y)

# 3. Inspect learned parameters and test predictions
print(f"Total epochs run: {len(model.loss_history)}")
print(f"Initial Loss: {model.loss_history[0]:.4f}")
print(f"Final Loss:   {model.loss_history[-1]:.6f}")
print(f"Learned Weights: {np.round(model.weights, 3)} (Expected: [3. 5.])")
print(f"Learned Bias:    {model.bias:.3f} (Expected: 2.000)")

# Test on a brand new data point: 3*(1) + 5*(2) + 2 = 15
X_new = np.array([[1.0, 2.0]])
print(f"Prediction: {model.predict(X_new)[0]:.3f} (Expected: 15.000)")
