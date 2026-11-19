import numpy as np
from ml_lib.linear_model import LogisticRegression

# 1. Generate linearly separable binary classification data
np.random.seed(42)
X_class0 = np.random.randn(50, 2) + np.array([-2.0, -2.0])
X_class1 = np.random.randn(50, 2) + np.array([2.0, 2.0])

X = np.vstack([X_class0, X_class1])
y = np.concatenate([np.zeros(50), np.ones(50)])

# 2. Train Logistic Regression
model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X, y)

# 3. Evaluate accuracy
predictions = model.predict(X)
accuracy = np.mean(predictions == y) * 100

initial_loss = next(iter(model.loss_history))
final_loss = model.loss_history[-1]

print(f"Initial BCE Loss: {initial_loss:.4f}")
print(f"Final BCE Loss:   {final_loss:.4f}")
print(f"Training Accuracy: {accuracy:.2f}%")

# Test on two clear examples
test_points = np.array([[-3.0, -3.0], [3.0, 3.0]])
print("Probabilities:", np.round(model.predict_proba(test_points), 4))
print("Predicted classes:", model.predict(test_points))
