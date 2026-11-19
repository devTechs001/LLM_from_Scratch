import numpy as np
from ml_lib.losses import mse_loss, mse_gradient, sigmoid, binary_cross_entropy
from ml_lib.optimizers import GradientDescent

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.epochs = epochs
        self.optimizer = GradientDescent(learning_rate=learning_rate)
        self.weights = None
        self.bias = None
        self.loss_history = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Computes linear predictions: y_pred = X @ weights + bias."""
        return np.dot(X, self.weights) + self.bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the model using Gradient Descent."""
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            # Forward pass
            y_pred = self.predict(X)
            
            # Record loss
            loss = mse_loss(y, y_pred)
            self.loss_history.append(loss)

            # Backward pass (must be indented inside the for loop!)
            d_ypred = mse_gradient(y, y_pred)
            dw = np.dot(X.T, d_ypred)
            db = np.sum(d_ypred)

            # Update weights and bias (must be indented inside the for loop!)
            self.weights = self.optimizer.step(self.weights, dw)
            self.bias = self.optimizer.step(self.bias, db)

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000):
        self.epochs = epochs
        self.optimizer = GradientDescent(learning_rate=learning_rate)
        self.weights = None
        self.bias = None
        self.loss_history = []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability estimates between 0 and 1."""
        linear_output = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary class predictions (0 or 1)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the binary classifier using Gradient Descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            # Forward pass
            y_prob = self.predict_proba(X)
            loss = binary_cross_entropy(y, y_prob)
            self.loss_history.append(loss)

            # Backward pass
            dz = (1.0 / n_samples) * (y_prob - y)
            dw = np.dot(X.T, dz)
            db = np.sum(dz)

            # Update parameters
            self.weights = self.optimizer.step(self.weights, dw)
            self.bias = self.optimizer.step(self.bias, db)