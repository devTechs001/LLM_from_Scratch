import numpy as np
from ml_lib.nn import Linear, ReLU, Sequential
from ml_lib.losses import softmax, cross_entropy_loss, cross_entropy_softmax_gradient

np.random.seed(42)

# 1. Generate 3 distinct 2D clusters (Classes 0, 1, 2)
X0 = np.random.randn(40, 2) * 0.5 + np.array([0.0, 2.0])
X1 = np.random.randn(40, 2) * 0.5 + np.array([-2.0, -1.0])
X2 = np.random.randn(40, 2) * 0.5 + np.array([2.0, -1.0])

X = np.vstack([X0, X1, X2])
y = np.concatenate([
    np.zeros(40, dtype=int),
    np.ones(40, dtype=int),
    np.full(40, 2, dtype=int)
])

# 2. Build network outputting raw logits for 3 classes: 2 -> 16 -> 3
model = Sequential([
    Linear(in_features=2, out_features=16),
    ReLU(),
    Linear(in_features=16, out_features=3)
])

# 3. Training loop
learning_rate = 0.2
epochs = 500

for epoch in range(epochs):
    logits = model.forward(X)
    probs = softmax(logits)
    loss = cross_entropy_loss(y, probs)
    
    d_logits = cross_entropy_softmax_gradient(y, probs)
    model.backward(d_logits, learning_rate)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Cross-Entropy Loss: {loss:.4f}")

# 4. Evaluate accuracy
final_probs = softmax(model.forward(X))
predicted_classes = np.argmax(final_probs, axis=1)
accuracy = np.mean(predicted_classes == y) * 100
print(f"\nFinal Multi-Class Accuracy: {accuracy:.2f}%")

# Test on 3 new points near each cluster center
test_points = np.array([[0.0, 2.0], [-2.0, -1.0], [2.0, -1.0]])
test_probs = softmax(model.forward(test_points))
print("Predicted classes for, [-2,-1], [2,-1]:", np.argmax(test_probs, axis=1))
