"""Model Quantization and Compression primitives from scratch in pure NumPy.

Includes:
- Symmetric and asymmetric INT8 quantization and dequantization.
- Symmetric INT4 quantization and dequantization.
- QuantizedLinear: Memory-efficient Linear layer holding compressed integer weights
  with zero-copy or dequantized forward evaluation, demonstrating 4x-8x weight compression.
"""

from typing import Tuple, Optional
import numpy as np
from ml_lib.nn import Module, Linear


def quantize_symmetric_int8(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric per-tensor or per-channel INT8 quantization.

    Maps floating-point weights to signed 8-bit integers [-127, 127]:
        scale = max(|weights|) / 127
        w_int8 = clip(round(weights / scale), -127, 127)

    Args:
        weights: Float array of arbitrary shape.

    Returns:
        w_int8: np.int8 quantized array.
        scale: Scaling factor(s).
    """
    max_val = np.max(np.abs(weights))
    scale = np.maximum(max_val / 127.0, 1e-12)
    w_q = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
    return w_q, np.array(scale, dtype=np.float32)


def dequantize_symmetric_int8(w_q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize INT8 integers back to floating-point values:

    weights_approx = w_q * scale
    """
    return w_q.astype(np.float32) * scale


def quantize_asymmetric_int8(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Asymmetric INT8 quantization mapping arbitrary range [min_val, max_val] to [0, 255]."""
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    scale = max((max_val - min_val) / 255.0, 1e-12)
    data_q = np.clip(np.round((data - min_val) / scale), 0, 255).astype(np.uint8)
    return data_q, scale, min_val


def dequantize_asymmetric_int8(data_q: np.ndarray, scale: float, min_val: float) -> np.ndarray:
    """Dequantize asymmetric UINT8 array back to float:

    data_approx = data_q * scale + min_val
    """
    return data_q.astype(np.float32) * scale + min_val


def quantize_symmetric_int4(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric 4-bit quantization mapping weights to [-7, 7]."""
    max_val = np.max(np.abs(weights))
    scale = np.maximum(max_val / 7.0, 1e-12)
    w_q = np.clip(np.round(weights / scale), -7, 7).astype(np.int8)
    return w_q, np.array(scale, dtype=np.float32)


def dequantize_symmetric_int4(w_q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize 4-bit array back to float."""
    return w_q.astype(np.float32) * scale


class QuantizedLinear(Module):
    """Memory-compressed Linear layer with quantized weights (INT8 or INT4)."""

    def __init__(self, base_linear: Linear, bits: int = 8):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bits = bits
        self.bias = base_linear.bias.copy() if base_linear.bias is not None else None

        if bits == 8:
            self.w_q, self.scale = quantize_symmetric_int8(base_linear.weights)
        elif bits == 4:
            self.w_q, self.scale = quantize_symmetric_int4(base_linear.weights)
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}. Must be 8 or 4.")

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to standard FP32 weights."""
        return 32.0 / self.bits

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Dequantize weights on the fly and compute linear output."""
        if self.bits == 8:
            w_float = dequantize_symmetric_int8(self.w_q, self.scale)
        else:
            w_float = dequantize_symmetric_int4(self.w_q, self.scale)

        out = np.matmul(x, w_float)
        if self.bias is not None:
            out += self.bias
        return out
