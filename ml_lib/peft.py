"""Parameter-Efficient Fine-Tuning (PEFT) and LoRA from scratch in pure NumPy.

Includes:
- LoRALinear: Low-Rank Adaptation module injecting trainable rank decomposition
  matrices W = W_0 + (alpha / r) * B @ A while freezing base weights W_0.
- Weight merging and unmerging for zero-latency inference serving.
- SFT loss with prompt masking: computes cross-entropy only on target completion tokens.
"""

from typing import Optional, List, Dict, Tuple
import numpy as np
from ml_lib.nn import Module, Linear
from ml_lib.losses import softmax, cross_entropy_loss, cross_entropy_softmax_gradient


class LoRALinear(Module):
    """Low-Rank Adaptation (LoRA) layer wrapping a base Linear layer.

    Forward:
        y = x @ W_0 + (alpha / r) * (x @ A.T) @ B.T + b
    where:
        W_0 in R^{d_in x d_out} is frozen.
        A in R^{r x d_in} initialized with Gaussian noise N(0, 1 / r).
        B in R^{d_out x r} initialized to zeros (so Delta W = 0 at initialization).
        alpha / r is the constant scaling factor.
    """

    def __init__(
        self,
        base_linear: Linear,
        rank: int = 4,
        alpha: float = 8.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.freeze_base = freeze_base
        self.merged = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # LoRA low-rank matrices
        # A: (rank, in_features)
        # B: (out_features, rank)
        self.lora_A = np.random.randn(rank, self.in_features) * np.sqrt(1.0 / rank)
        self.lora_B = np.zeros((self.out_features, rank))

        # Gradients
        self.grad_lora_A = np.zeros_like(self.lora_A)
        self.grad_lora_B = np.zeros_like(self.lora_B)

        # Cache for backpropagation
        self._cached_x = None
        self._cached_xa = None

    def merge(self) -> None:
        """Merge LoRA weights into base_linear.weights for zero-overhead inference:

        W_new = W_0 + scaling * (B @ A).T
        """
        if not self.merged:
            delta_w = self.scaling * (self.lora_B @ self.lora_A).T
            self.base_linear.weights += delta_w
            self.merged = True

    def unmerge(self) -> None:
        """Unmerge LoRA weights from base_linear.weights to resume training."""
        if self.merged:
            delta_w = self.scaling * (self.lora_B @ self.lora_A).T
            self.base_linear.weights -= delta_w
            self.merged = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass supporting 2D (B, in_f) or 3D (B, T, in_f) inputs."""
        self._cached_x = x

        if self.merged:
            return self.base_linear.forward(x)

        # Base forward
        out_base = self.base_linear.forward(x)

        # LoRA forward: x @ A.T @ B.T * scaling
        # x: (..., in_f)
        # A.T: (in_f, r) -> xa: (..., r)
        xa = np.matmul(x, self.lora_A.T)
        self._cached_xa = xa

        # B.T: (r, out_f) -> lora_out: (..., out_f)
        lora_out = np.matmul(xa, self.lora_B.T) * self.scaling

        return out_base + lora_out

    def backward(self, d_out: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        """Backward pass computing gradients for lora_A, lora_B, and passing dx to input."""
        if self.merged:
            raise RuntimeError("Cannot backward through merged LoRALinear. Call unmerge() first.")

        x = self._cached_x
        xa = self._cached_xa
        scaling = self.scaling

        # Reshape to 2D for clean matrix multiplications
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        xa_2d = xa.reshape(-1, self.rank)
        d_out_2d = d_out.reshape(-1, self.out_features)

        # Gradient w.r.t lora_B:
        # lora_out = xa @ B.T * scaling
        # d_lora_B = (d_out.T @ xa) * scaling
        self.grad_lora_B = np.matmul(d_out_2d.T, xa_2d) * scaling

        # Gradient w.r.t xa:
        # d_xa = (d_out @ B) * scaling
        d_xa_2d = np.matmul(d_out_2d, self.lora_B) * scaling

        # Gradient w.r.t lora_A:
        # xa = x @ A.T
        # d_lora_A = d_xa.T @ x
        self.grad_lora_A = np.matmul(d_xa_2d.T, x_2d)

        # Gradient w.r.t input x through LoRA branch:
        # d_x_lora = d_xa @ A
        d_x_lora_2d = np.matmul(d_xa_2d, self.lora_A)

        # Gradient through base linear
        d_x_base = self.base_linear.backward(
            d_out,
            learning_rate=learning_rate if not self.freeze_base else None
        )
        d_x_base_2d = d_x_base.reshape(-1, self.in_features)

        # Total d_x
        d_x = (d_x_base_2d + d_x_lora_2d).reshape(orig_shape)

        # Parameter update if learning_rate is supplied
        if learning_rate is not None:
            self.lora_A -= learning_rate * self.grad_lora_A
            self.lora_B -= learning_rate * self.grad_lora_B

        return d_x


def apply_lora_to_linear(
    linear_layer: Linear,
    rank: int = 4,
    alpha: float = 8.0,
    freeze_base: bool = True
) -> LoRALinear:
    """Helper to convert a standard Linear layer into a LoRALinear layer."""
    return LoRALinear(linear_layer, rank=rank, alpha=alpha, freeze_base=freeze_base)


def sft_masked_cross_entropy(
    logits: np.ndarray,
    target_ids: np.ndarray,
    prompt_lengths: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Compute Supervised Fine-Tuning (SFT) loss with prompt masking.

    Loss is computed strictly on response/completion tokens:
    position t is ignored if t < prompt_length - 1 (next-token prediction starts
    after the last prompt token).

    Args:
        logits: (B, T, vocab_size) model unnormalized predictions.
        target_ids: (B, T) ground truth token IDs.
        prompt_lengths: (B,) number of tokens in the prompt for each sample.

    Returns:
        loss: Scalar mean cross-entropy over active response tokens.
        grad_logits: (B, T, vocab_size) gradient w.r.t logits.
    """
    B, T, V = logits.shape
    grad_logits = np.zeros_like(logits)
    total_loss = 0.0
    active_tokens = 0

    for b in range(B):
        p_len = prompt_lengths[b]
        # Valid prediction positions: index i predicts target_ids[b, i+1]
        # or if target_ids is aligned with logits[b, t], positions >= p_len
        # Here standard next-token formulation: logits[:, :-1] predicts target_ids[:, 1:]
        for t in range(p_len - 1, T - 1):
            next_token = target_ids[b, t + 1]
            p = softmax(logits[b, t])
            token_loss = -np.log(np.clip(p[next_token], 1e-15, 1.0))
            total_loss += token_loss
            active_tokens += 1

            # Analytic softmax gradient: p - 1 at target, p elsewhere
            grad = p.copy()
            grad[next_token] -= 1.0
            grad_logits[b, t] = grad

    mean_loss = total_loss / max(active_tokens, 1)
    if active_tokens > 0:
        grad_logits /= active_tokens

    return float(mean_loss), grad_logits
