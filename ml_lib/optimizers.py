import numpy as np
from typing import Union, Dict, Any, Optional, List

class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.t = 0  # Global step counter
        self._states: Dict[Any, Dict[str, np.ndarray]] = {}
        self._auto_key_idx = 0

    def zero_grad(self) -> None:
        """Resets any step iteration tracking if needed."""
        self._auto_key_idx = 0

    def _get_key(self, param: Union[np.ndarray, float], param_id: Optional[Any]) -> Any:
        if param_id is not None:
            return param_id
        # If no explicit param_id is given, use sequential index within the step cycle
        key = self._auto_key_idx
        self._auto_key_idx += 1
        return key

class GradientDescent(Optimizer):
    """
    Standard Batch / Stochastic Gradient Descent:
    theta = theta - eta * grad
    """
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate=learning_rate)

    def step(
        self,
        param: Union[np.ndarray, float],
        grad: Union[np.ndarray, float],
        param_id: Optional[Any] = None
    ) -> Union[np.ndarray, float]:
        """Updates a parameter array in the opposite direction of its gradient."""
        return param - self.learning_rate * grad


class SGDMomentum(Optimizer):
    """
    SGD with Polyak Momentum:
    v_{t+1} = beta * v_t + grad
    theta_{t+1} = theta_t - eta * v_{t+1}
    Accelerates gradient vectors in the right direction, dampening oscillations.
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate=learning_rate)
        self.momentum = momentum

    def step(
        self,
        param: Union[np.ndarray, float],
        grad: Union[np.ndarray, float],
        param_id: Optional[Any] = None
    ) -> Union[np.ndarray, float]:
        key = self._get_key(param, param_id)
        if key not in self._states:
            self._states[key] = {"v": np.zeros_like(param, dtype=float)}

        v = self._states[key]["v"]
        v = self.momentum * v + grad
        self._states[key]["v"] = v

        return param - self.learning_rate * v


class RMSprop(Optimizer):
    """
    Root Mean Square Propagation (Geoffrey Hinton, 2012):
    Maintains a moving average of squared gradients to normalize step sizes:
    s_{t+1} = beta * s_t + (1 - beta) * grad^2
    theta_{t+1} = theta_t - (eta / (sqrt(s_{t+1}) + eps)) * grad
    """
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.99, epsilon: float = 1e-8):
        super().__init__(learning_rate=learning_rate)
        self.beta = beta
        self.epsilon = epsilon

    def step(
        self,
        param: Union[np.ndarray, float],
        grad: Union[np.ndarray, float],
        param_id: Optional[Any] = None
    ) -> Union[np.ndarray, float]:
        key = self._get_key(param, param_id)
        if key not in self._states:
            self._states[key] = {"s": np.zeros_like(param, dtype=float)}

        s = self._states[key]["s"]
        s = self.beta * s + (1.0 - self.beta) * (grad ** 2)
        self._states[key]["s"] = s

        step_update = (self.learning_rate / (np.sqrt(s) + self.epsilon)) * grad
        return param - step_update


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Kingma & Ba, 2014):
    Combines first moment (momentum) and second moment (RMSprop) with bias correction:
    m_{t+1} = beta1 * m_t + (1 - beta1) * grad
    v_{t+1} = beta2 * v_t + (1 - beta2) * grad^2
    m_hat = m_{t+1} / (1 - beta1^t)
    v_hat = v_{t+1} / (1 - beta2^t)
    theta_{t+1} = theta_t - (eta / (sqrt(v_hat) + eps)) * m_hat
    """
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep: Dict[Any, int] = {}

    def step(
        self,
        param: Union[np.ndarray, float],
        grad: Union[np.ndarray, float],
        param_id: Optional[Any] = None
    ) -> Union[np.ndarray, float]:
        key = self._get_key(param, param_id)
        if key not in self._states:
            self._states[key] = {
                "m": np.zeros_like(param, dtype=float),
                "v": np.zeros_like(param, dtype=float)
            }
            self.timestep[key] = 0

        self.timestep[key] += 1
        t = self.timestep[key]

        m = self._states[key]["m"]
        v = self._states[key]["v"]

        # First and second moment updates
        m = self.beta1 * m + (1.0 - self.beta1) * grad
        v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)

        self._states[key]["m"] = m
        self._states[key]["v"] = v

        # Bias corrections
        m_hat = m / (1.0 - (self.beta1 ** t))
        v_hat = v / (1.0 - (self.beta2 ** t))

        step_update = (self.learning_rate / (np.sqrt(v_hat) + self.epsilon)) * m_hat
        return param - step_update


class AdamW(Optimizer):
    """
    Decoupled Weight Decay Adam (Loshchilov & Hutter, 2017):
    In standard Adam with L2 regularization, the weight decay gradient gets distorted
    by the second moment scaling (1 / sqrt(v)).
    AdamW decouples weight decay directly onto the parameter:
    theta_{t+1} = theta_t - eta * weight_decay * theta_t - (eta / (sqrt(v_hat) + eps)) * m_hat
    This is the default optimizer for modern LLMs (GPT-3/4, LLaMA, etc.).
    """
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01
    ):
        super().__init__(learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.timestep: Dict[Any, int] = {}

    def step(
        self,
        param: Union[np.ndarray, float],
        grad: Union[np.ndarray, float],
        param_id: Optional[Any] = None
    ) -> Union[np.ndarray, float]:
        key = self._get_key(param, param_id)
        if key not in self._states:
            self._states[key] = {
                "m": np.zeros_like(param, dtype=float),
                "v": np.zeros_like(param, dtype=float)
            }
            self.timestep[key] = 0

        self.timestep[key] += 1
        t = self.timestep[key]

        m = self._states[key]["m"]
        v = self._states[key]["v"]

        # First and second moment updates
        m = self.beta1 * m + (1.0 - self.beta1) * grad
        v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)

        self._states[key]["m"] = m
        self._states[key]["v"] = v

        # Bias corrections
        m_hat = m / (1.0 - (self.beta1 ** t))
        v_hat = v / (1.0 - (self.beta2 ** t))

        # Decoupled weight decay
        param_decayed = param - self.learning_rate * self.weight_decay * param

        # Adaptive step
        step_update = (self.learning_rate / (np.sqrt(v_hat) + self.epsilon)) * m_hat
        return param_decayed - step_update


def clip_grad_norm(
    grads: Union[np.ndarray, List[np.ndarray]],
    max_norm: float
) -> float:
    """
    Clips gradients by their global L2 norm to prevent exploding gradients:
    total_norm = sqrt(sum(||grad_i||^2))
    if total_norm > max_norm:
        grad_i = grad_i * (max_norm / total_norm)
    Returns the original total_norm.
    """
    if isinstance(grads, np.ndarray):
        grad_list = [grads]
    else:
        grad_list = grads

    total_norm = float(np.sqrt(sum(np.sum(g ** 2) for g in grad_list)))
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for g in grad_list:
            g *= clip_coef

    return total_norm


def clip_grad_value(
    grads: Union[np.ndarray, List[np.ndarray]],
    clip_value: float
) -> None:
    """
    Clips gradients element-wise to [-clip_value, clip_value] in-place.
    """
    if isinstance(grads, np.ndarray):
        np.clip(grads, -clip_value, clip_value, out=grads)
    else:
        for g in grads:
            np.clip(g, -clip_value, clip_value, out=g)
