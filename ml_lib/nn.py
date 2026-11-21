import numpy as np
from typing import List, Tuple, Optional, Any

# =====================================================================
# Weight Initialization Schemes
# =====================================================================

def xavier_uniform(shape: Tuple[int, int]) -> np.ndarray:
    """Glorot/Xavier uniform initialization for Tanh/Sigmoid activations."""
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

def xavier_normal(shape: Tuple[int, int]) -> np.ndarray:
    """Glorot/Xavier normal initialization for Tanh/Sigmoid activations."""
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * std

def kaiming_uniform(shape: Tuple[int, int]) -> np.ndarray:
    """He/Kaiming uniform initialization for ReLU/LeakyReLU activations."""
    fan_in = shape[0]
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)

def kaiming_normal(shape: Tuple[int, int]) -> np.ndarray:
    """He/Kaiming normal initialization for ReLU activations."""
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


# =====================================================================
# Base Module
# =====================================================================

class Module:
    """Base class for all neural network layers and containers."""
    def __init__(self):
        self.training = True

    def train(self) -> None:
        """Sets module and submodules to training mode."""
        self.training = True

    def eval(self) -> None:
        """Sets module and submodules to evaluation/inference mode."""
        self.training = False

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError


# =====================================================================
# Linear (Dense / Fully-Connected) Layer
# =====================================================================

class Linear(Module):
    """
    Fully connected layer: output = X @ weights + bias
    Supports 2D (batch_size, in_features) and 3D (batch_size, seq_len, in_features) tensors.
    """
    def __init__(self, in_features: int, out_features: int, init: str = "kaiming"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if init == "kaiming":
            self.weights = kaiming_normal((in_features, out_features))
        elif init == "xavier":
            self.weights = xavier_normal((in_features, out_features))
        else:
            scale = np.sqrt(2.0 / in_features)
            self.weights = np.random.randn(in_features, out_features) * scale

        self.bias = np.zeros((1, out_features))
        
        self.inputs = None
        self.d_weights = None
        self.d_bias = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.inputs = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, d_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        in_shape = self.inputs.shape
        # Reshape to 2D for matrix multiplication if input has >= 3 dimensions
        X_flat = self.inputs.reshape(-1, self.in_features)
        d_out_flat = d_output.reshape(-1, self.out_features)

        # Gradients with respect to parameters
        self.d_weights = np.dot(X_flat.T, d_out_flat)
        self.d_bias = np.sum(d_out_flat, axis=0, keepdims=True)
        
        # Gradient with respect to inputs
        d_input_flat = np.dot(d_out_flat, self.weights.T)
        d_input = d_input_flat.reshape(in_shape)
        
        # Update parameters immediately if learning_rate is supplied
        if learning_rate is not None:
            self.weights -= learning_rate * self.d_weights
            self.bias -= learning_rate * self.d_bias

        return d_input


# =====================================================================
# Activation Functions
# =====================================================================

class ReLU(Module):
    """Rectified Linear Unit activation: max(0, x)"""
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.inputs = X
        return np.maximum(0, X)

    def backward(self, d_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        return d_output * (self.inputs > 0)


class LeakyReLU(Module):
    """Leaky ReLU activation: max(alpha * x, x) to prevent dying neurons."""
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.inputs = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.inputs = X
        return np.where(X > 0, X, self.alpha * X)

    def backward(self, d_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        dx = np.where(self.inputs > 0, 1.0, self.alpha)
        return d_output * dx


class Sigmoid(Module):
    """Sigmoid activation for binary probabilities: sigma(x) = 1 / (1 + exp(-x))."""
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.output = 1.0 / (1.0 + np.exp(-np.clip(X, -250, 250)))
        return self.output

    def backward(self, d_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        return d_output * (self.output * (1.0 - self.output))


class Tanh(Module):
    """Hyperbolic Tangent activation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))."""
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.output = np.tanh(X)
        return self.output

    def backward(self, d_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        return d_output * (1.0 - (self.output ** 2))


class GELU(Module):
    """
    Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016):
    The standard activation used in GPT-2, GPT-3, BERT, and modern Transformers.
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    def __init__(self):
        super().__init__()
        self.inputs = None
        self._sqrt_2_pi = np.sqrt(2.0 / np.pi)
        self._c = 0.044715

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.inputs = X
        u = self._sqrt_2_pi * (X + self._c * (X ** 3))
        return 0.5 * X * (1.0 + np.tanh(u))

    def backward(self, d_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        X = self.inputs
        u = self._sqrt_2_pi * (X + self._c * (X ** 3))
        tanh_u = np.tanh(u)
        du_dx = self._sqrt_2_pi * (1.0 + 3.0 * self._c * (X ** 2))
        d_gelu = 0.5 * (1.0 + tanh_u) + 0.5 * X * (1.0 - (tanh_u ** 2)) * du_dx
        return d_output * d_gelu


class SiLU(Module):
    """
    Sigmoid Linear Unit / Swish: silu(x) = x * sigmoid(x).
    Core activation of modern LLMs (LLaMA SwiGLU FFN).
    """
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.sig = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.inputs = X
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(X, -250, 250)))
        return X * self.sig

    def backward(self, d_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        # Derivative: sigma(x) * (1 + x * (1 - sigma(x)))
        d_silu = self.sig * (1.0 + self.inputs * (1.0 - self.sig))
        return d_output * d_silu


# =====================================================================
# Normalization & Regularization Layers
# =====================================================================

class LayerNorm(Module):
    """
    Layer Normalization over the last dimension (Ba, Kiros & Hinton, 2016):
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones((1, normalized_shape))
        self.beta = np.zeros((1, normalized_shape))
        self.d_gamma = None
        self.d_beta = None
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (X - mean) * inv_std
        out = x_hat * self.gamma + self.beta
        self.cache = (X, x_hat, inv_std)
        return out

    def backward(self, d_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        X, x_hat, inv_std = self.cache
        D = self.normalized_shape

        d_out_flat = d_output.reshape(-1, D)
        x_hat_flat = x_hat.reshape(-1, D)

        self.d_gamma = np.sum(d_out_flat * x_hat_flat, axis=0, keepdims=True)
        self.d_beta = np.sum(d_out_flat, axis=0, keepdims=True)

        d_x_hat = d_output * self.gamma
        d_input = (1.0 / D) * inv_std * (
            D * d_x_hat
            - np.sum(d_x_hat, axis=-1, keepdims=True)
            - x_hat * np.sum(d_x_hat * x_hat, axis=-1, keepdims=True)
        )

        if learning_rate is not None:
            self.gamma -= learning_rate * self.d_gamma
            self.beta -= learning_rate * self.d_beta

        return d_input


class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019):
    RMSNorm(x) = (x / RMS(x)) * gamma
    where RMS(x) = sqrt((1/D) * sum(x_i^2) + eps)
    Modern standard for LLaMA, Mistral, Gemma, Grok.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones((1, dim))
        self.d_gamma = None
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(X ** 2, axis=-1, keepdims=True) + self.eps)
        x_hat = X / rms
        out = x_hat * self.gamma
        self.cache = (X, x_hat, rms)
        return out

    def backward(self, d_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        X, x_hat, rms = self.cache
        D = self.dim

        d_out_flat = d_output.reshape(-1, D)
        x_hat_flat = x_hat.reshape(-1, D)

        self.d_gamma = np.sum(d_out_flat * x_hat_flat, axis=0, keepdims=True)
        d_x_hat = d_output * self.gamma

        d_input = (1.0 / rms) * (
            d_x_hat - x_hat * np.mean(d_x_hat * x_hat, axis=-1, keepdims=True)
        )

        if learning_rate is not None:
            self.gamma -= learning_rate * self.d_gamma

        return d_input


class Dropout(Module):
    """
    Inverted Dropout regularization:
    During training: drops activations with probability p, scales remaining by 1 / (1 - p).
    During evaluation: identity pass-through.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.training and self.p > 0.0:
            self.mask = (np.random.rand(*X.shape) >= self.p) / (1.0 - self.p)
            return X * self.mask
        return X

    def backward(self, d_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        if self.training and self.p > 0.0 and self.mask is not None:
            return d_output * self.mask
        return d_output


# =====================================================================
# Sequential Container
# =====================================================================

class Sequential(Module):
    """Chains multiple layers together sequentially."""
    def __init__(self, layers: List[Any]):
        super().__init__()
        self.layers = layers

    def train(self) -> None:
        self.training = True
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self) -> None:
        self.training = False
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, d_loss: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        grad = d_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad
