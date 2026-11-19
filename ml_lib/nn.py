import numpy as np

class Linear:
    """Fully connected layer: output = X @ weights + bias"""
    def __init__(self, in_features: int, out_features: int):
        # He initialization helps prevent vanishing/exploding gradients with ReLU
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros((1, out_features))
        
        self.inputs = None
        self.d_weights = None
        self.d_bias = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.inputs = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        # Gradients with respect to parameters
        self.d_weights = np.dot(self.inputs.T, d_output)
        self.d_bias = np.sum(d_output, axis=0, keepdims=True)
        
        # Gradient with respect to inputs (passed backward to previous layer)
        d_input = np.dot(d_output, self.weights.T)
        
        # Update parameters immediately
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias
        return d_input


class ReLU:
    """Rectified Linear Unit activation: max(0, x)"""
    def __init__(self):
        self.inputs = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.inputs = X
        return np.maximum(0, X)

    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        # Gradient flows only where input was > 0
        return d_output * (self.inputs > 0)


class Sigmoid:
    """Sigmoid activation for binary output probabilities"""
    def __init__(self):
        self.output = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.output = 1.0 / (1.0 + np.exp(-np.clip(X, -250, 250)))
        return self.output

    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        return d_output * (self.output * (1.0 - self.output))


class Sequential:
    """Chains multiple layers together sequentially"""
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, d_loss: np.ndarray, learning_rate: float) -> None:
        grad = d_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
