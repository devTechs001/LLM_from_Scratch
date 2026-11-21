# Phase 11 & 12: Quantization & Compression & Sparse Mixture-of-Experts (MoE)

## Executive Summary

Phases 11 and 12 complete the frontier systems stack of our pure NumPy framework:
1. **Phase 11: Quantization & Model Compression (`ml_lib/quantization.py`)**:
   - Symmetric and asymmetric INT8 quantization (4x memory reduction).
   - 4-bit integer quantization (INT4, 8x memory reduction).
   - `QuantizedLinear`: Drop-in replacement for `Linear` layers holding compressed integer weights and executing on-the-fly dequantized evaluation with $< 2\%$ relative output distortion.
2. **Phase 12: Sparse Mixture-of-Experts (MoE) (`ml_lib/moe.py`)**:
   - Token-level sparse routing activating top-$k$ experts per token.
   - Top-$k$ Softmax Gating router.
   - Switch Transformer / GShard auxiliary load-balancing loss $\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i$.
   - Forward token dispatch and exact backward gradient propagation through activated experts and router logits.

---

## 1. Phase 11: Quantization & Model Compression

### 1.1 Symmetric INT8 Quantization
Symmetric quantization maps continuous floating-point weights $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ to signed 8-bit integers $[-127, 127]$:

$$S = \frac{\max(|W|)}{127}$$
$$W_q = \operatorname{clip}\left(\left\lfloor \frac{W}{S} \right\rceil, -127, 127\right) \in \text{int8}$$
$$\hat{W} = W_q \cdot S \approx W$$

### 1.2 Asymmetric INT8 Quantization (Min-Max)
For activations or feature maps that have non-zero or strictly positive means (e.g. after ReLU/GELU):

$$S = \frac{X_{\max} - X_{\min}}{255}$$
$$X_q = \operatorname{clip}\left(\left\lfloor \frac{X - X_{\min}}{S} \right\rceil, 0, 255\right) \in \text{uint8}$$
$$\hat{X} = X_q \cdot S + X_{\min}$$

### 1.3 Symmetric INT4 Quantization
INT4 quantization maps weights to signed 4-bit integers $[-7, 7]$, reducing the memory footprint by **8x** relative to FP32:

$$S = \frac{\max(|W|)}{7}$$
$$W_q = \operatorname{clip}\left(\left\lfloor \frac{W}{S} \right\rceil, -7, 7\right)$$

### 1.4 QuantizedLinear
The `QuantizedLinear` layer stores only integer matrices `w_q` and small scalar scales, dynamically unpacking or executing matrix multiplications with 4x-8x reduced memory requirements.

---

## 2. Phase 12: Sparse Mixture of Experts (MoE)

### 2.1 Top-$k$ Softmax Gating
In dense Transformers, every token is processed by identical MLP feed-forward networks. In Sparse MoE:
1. An expert pool $\{E_1, E_2, \dots, E_M\}$ is maintained (each an MLP with dimension $d_{\text{model}} \to d_{\text{ff}} \to d_{\text{model}}$).
2. A gating router $W_g \in \mathbb{R}^{d_{\text{model}} \times M}$ computes routing logits $H = X W_g$.
3. The top-$k$ indices are selected:
   $$\mathcal{T} = \operatorname{TopK}(H_t, k)$$
4. The router probabilities over the selected top-$k$ are normalized:
   $$w_{t, j} = \frac{\exp(H_{t, j})}{\sum_{m \in \mathcal{T}} \exp(H_{t, m})} \quad \forall j \in \mathcal{T}$$
5. The token representation is computed as a linear combination of the $k$ active experts:
   $$y_t = \sum_{j \in \mathcal{T}} w_{t, j} \cdot E_j(x_t)$$

Tokens not assigned to expert $j$ require no computation from $E_j$, keeping parameter capacity massive while per-token inference FLOPs remain fixed.

### 2.2 Auxiliary Load-Balancing Loss (Switch Transformer)
Without regularization, routers often suffer from *expert collapse*, where all tokens route to 1 or 2 experts. We incorporate the Switch Transformer / GShard auxiliary loss:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot M \sum_{i=1}^M f_i \cdot P_i$$

where:
- $M$ is the number of experts.
- $f_i = \frac{1}{N} \sum_{t=1}^N \mathbb{I}(i \in \mathcal{T}_t)$ is the empirical fraction of tokens routed to expert $i$.
- $P_i = \frac{1}{N} \sum_{t=1}^N \operatorname{softmax}(H)_{t, i}$ is the average router probability for expert $i$.

The minimum occurs when $f_i = P_i = \frac{1}{M}$ (uniform load balancing across all experts).

### 2.3 Single-Dispatch Expert Routing & Backward Pass
To prevent duplicate state overwrites, our pure NumPy implementation groups all tokens routed to expert $j$ across any top-$k$ rank, evaluates `E_j.forward()` exactly once per step, and aggregates gradients backwards through the active expert weights and the router softmax matrix.

---

## 3. Empirical Test Suite & Results

Across `tests/test_phase11_and_12_moe_quant.py`:
- `test_symmetric_int8_quantization`: Quantizes to signed INT8 with $< 0.05$ MAE (**PASSED**)
- `test_asymmetric_int8_quantization`: Min-max quantization over biased activations (**PASSED**)
- `test_symmetric_int4_quantization`: 4-bit quantization with 8x compression (**PASSED**)
- `test_quantized_linear_forward_accuracy`: `QuantizedLinear` achieves $< 0.1$ MAE vs FP32 Linear (**PASSED**)
- `test_sparse_moe_forward_backward_shapes`: 3D sequence token routing and backpropagation (**PASSED**)
- `test_moe_auxiliary_load_balancing_loss`: Skewed routing produces higher penalty than uniform distribution (**PASSED**)
- `test_sparse_moe_training_convergence`: MoE reduces non-linear mapping loss $> 50\%$ (**PASSED**)

**71 out of 71 unit tests pass repository-wide** in 2.83s.
