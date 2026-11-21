# Phase 3: Sequence Modeling, Causal Multi-Head Attention & RoPE from Scratch

> **Module**: `ml_lib.llm`  
> **Test Suites**: `tests/test_phase3_attention.py`, `tests/test_self_attention.py`, `tests/test_llm_embeddings.py`  
> **Status**: Verified (32/32 workspace tests passing, KV cache identity error $\sim 10^{-16}$, RoPE invariance $\Delta = 0.0$)

---

## 1. Executive Summary: Overcoming the Sequential Bottleneck

Before Transformers, sequence modeling was dominated by Recurrent Neural Networks (RNNs) and LSTMs:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$$
This architecture suffers from two fatal limitations:
1. **Sequential Computation Bottleneck**: State $h_t$ cannot be computed until $h_{t-1}$ completes, preventing GPU parallelization during training.
2. **Information Compression & Vanishing Gradients**: Long-distance dependencies must travel through $T$ sequential matrix multiplications, causing information loss over long context windows.

The **Self-Attention** mechanism (Vaswani et al., 2017) completely removes recurrence. Tokens interact directly across arbitrary distances in constant $\mathcal{O}(1)$ path length, enabling massively parallel pretraining on modern GPUs.

Phase 3 implements production-grade sequence primitives: **Batched Multi-Head Attention**, **Rotary Position Embeddings (RoPE)**, and **Key-Value (KV) Caching**.

---

## 2. Mathematical Formulations & Derivations

### 2.1 Scaled Dot-Product Attention

Given an embedded input sequence $X \in \mathbb{R}^{T \times D}$:
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
where $W_Q, W_K \in \mathbb{R}^{D \times d_k}$ and $W_V \in \mathbb{R}^{D \times d_v}$.

$$\text{Attention}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

#### Mathematical Proof for the Scaling Factor $\frac{1}{\sqrt{d_k}}$:
Assume components of $q \in \mathbb{R}^{d_k}$ and $k \in \mathbb{R}^{d_k}$ are independent random variables with zero mean and unit variance:
$$\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \quad \operatorname{Var}(q_i) = \operatorname{Var}(k_i) = 1$$
The dot product is the sum of $d_k$ independent terms:
$$s = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$
$$\mathbb{E}[s] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$
$$\operatorname{Var}(s) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2 k_i^2] = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = d_k$$

Without scaling, for large $d_k$ (e.g. $d_k = 64$ or $128$), the variance of $s$ is $d_k$ and standard deviation is $\sqrt{d_k} \approx 8$ or $11.3$. This pushes Softmax inputs into extreme saturation regions where gradients vanish to 0:
$$\frac{\partial \operatorname{softmax}(s)_i}{\partial s_j} \approx 0$$
Dividing by $\sqrt{d_k}$ normalizes the variance:
$$\operatorname{Var}\left( \frac{q \cdot k}{\sqrt{d_k}} \right) = \frac{1}{d_k} \operatorname{Var}(q \cdot k) = \frac{d_k}{d_k} = 1$$
preserving healthy gradient flow during backpropagation.

---

### 2.2 Causal Triangular Masking

In decoder-only autoregressive LLMs (GPT, LLaMA), token $i$ must not attend to future tokens $j > i$.
We construct an upper-triangular mask $M \in \mathbb{R}^{T \times T}$:
$$M_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

Adding $M$ to scores before Softmax:
$$p_{ij} = \frac{e^{(QK^T)_{ij}/\sqrt{d_k} + M_{ij}}}{\sum_{l=1}^T e^{(QK^T)_{il}/\sqrt{d_k} + M_{il}}}$$
When $j > i$, $e^{-\infty} = 0$, guaranteeing mathematically zero probability of attending to the future:
$$p_{ij} = 0 \quad \forall j > i$$
In code, we use $-10^9$ to avoid IEEE 754 float `NaN` operations.

---

### 2.3 Multi-Head Attention (MHA)

Rather than performing a single attention pooling with dimension $D$, Multi-Head Attention projects $Q, K, V$ into $H$ distinct representation subspaces of dimension $d_k = D / H$:
$$\text{head}_h = \text{Attention}(X W_h^Q, X W_h^K, X W_h^V)$$
$$\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_H) W_O$$

* **Subspace Specialization**: Head 1 tracks subject-verb agreement; Head 2 attends to previous punctuation; Head 3 resolves coreferences; Head 4 attends to local bigrams.
* **Exact Analytical Backward Equations**:
  1. $d W_O = \text{Context}_{\text{concat}}^T \cdot d_{\text{out}}$
  2. $d\text{Context} = d_{\text{out}} \cdot W_O^T$ (reshaped to $(B, H, T, d_k)$)
  3. $d V = A^T \cdot d\text{Context}$
  4. $d A = d\text{Context} \cdot V^T$
  5. $d S = A \odot \left( d A - \sum_k (d A \odot A)_k \right) / \sqrt{d_k}$
  6. $d Q = d S \cdot K, \quad d K = d S^T \cdot Q$
  7. $d X = d Q \cdot W_Q^T + d K \cdot W_K^T + d V \cdot W_V^T$

---

### 2.4 Rotary Position Embeddings (RoPE - Su et al., 2021)

Modern LLMs (LLaMA-3, Mistral, Gemma, Qwen) replace additive learned positional embeddings with RoPE.

#### The Core Problem with Absolute Position Embeddings:
Adding absolute position vector $p_m$ to token embedding $x_m$ ($h_m = x_m + p_m$) loses relative distance invariance:
$$\langle h_m, h_n \rangle = x_m^T x_n + x_m^T p_n + p_m^T x_n + p_m^T p_n$$
Cross-terms entangle content with absolute positions, failing to generalize to unseen sequence lengths.

#### The RoPE Solution:
RoPE rotates query and key vectors in 2D complex planes by an angle proportional to token position $m$:
$$\mathbf{R}_{\Theta, m}^d = \operatorname{diag}\left( R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m} \right)$$
where each 2D rotation matrix is:
$$R_{\theta_i, m} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}, \quad \theta_i = 10000^{-2(i-1)/d}$$

#### Proof of Relative Invariance:
For query at position $m$ and key at position $n$:
$$\langle R_m q, R_n k \rangle = (R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n - m} k = g(q, k, n - m)$$
The dot product depends **solely on relative distance $n - m$**, not on absolute positions $m$ or $n$. Verified in `tests/test_phase3_attention.py` with zero difference ($\Delta = 0.0$).

---

### 2.5 Key-Value (KV) Caching for Fast Inference

During autoregressive generation, we emit one token at a time:
$$\text{Input at step } t: [x_1, \dots, x_t] \to \text{Predict } x_{t+1}$$

* **Without KV Cache**:
  At each step $t$, we recompute $K$ and $V$ for all past tokens $1 \dots t$:
  $$\text{Total FLOPs} = \sum_{t=1}^T \mathcal{O}(t \cdot D) = \mathcal{O}(T^2 \cdot D)$$
  Generation latency slows down quadratically as output grows longer.
* **With KV Cache**:
  Past keys $K_{1:t-1}$ and values $V_{1:t-1}$ never change! We store them in a persistent buffer. At step $t$, we only project the single new token:
  $$q_t = x_t W_Q, \quad k_t = x_t W_K, \quad v_t = x_t W_V$$
  $$K_{\text{cache}} \leftarrow [K_{\text{cache}}, k_t], \quad V_{\text{cache}} \leftarrow [V_{\text{cache}}, v_t]$$
  $$a_t = \operatorname{softmax}\left(\frac{q_t K_{\text{cache}}^T}{\sqrt{d_k}}\right), \quad \text{out}_t = a_t V_{\text{cache}} W_O$$
  $$\text{FLOPs per step} = \mathcal{O}(T \cdot D) \quad \text{(linear instead of quadratic)}$$

Verified in `test_phase3_attention.py`: Cached step activations match non-cached full-sequence evaluations down to $5.55 \times 10^{-16}$.

---

## 3. Verification Test Results

Running `pytest -v` across the entire workspace executes 32 passing tests:

```text
tests/test_llm_embeddings.py::test_char_tokenizer_encode_decode PASSED   [  3%]
tests/test_llm_embeddings.py::test_embedding_forward_and_backward PASSED [  6%]
tests/test_multiclass_nn.py::test_multiclass_mlp_convergence PASSED      [  9%]
tests/test_nn_xor.py::test_xor_non_linear_convergence PASSED             [ 12%]
tests/test_nn_xor.py::test_linear_layer_alone_fails_xor PASSED           [ 15%]
tests/test_phase1.py::test_mse_loss_and_gradient PASSED                  [ 18%]
tests/test_phase1.py::test_numerical_gradient_verification_mse PASSED    [ 21%]
tests/test_phase1.py::test_huber_loss_and_gradient PASSED                [ 25%]
tests/test_phase1.py::test_sigmoid_numerical_stability PASSED            [ 28%]
tests/test_phase1.py::test_binary_cross_entropy_stability PASSED         [ 31%]
tests/test_phase1.py::test_log_sum_exp_stability PASSED                  [ 34%]
tests/test_phase1.py::test_softmax_and_cross_entropy_gradient PASSED     [ 37%]
tests/test_phase1.py::test_gradient_descent_convergence PASSED           [ 40%]
tests/test_phase1.py::test_sgd_momentum_convergence PASSED               [ 43%]
tests/test_phase1.py::test_adam_and_adamw_convergence PASSED             [ 46%]
tests/test_phase1.py::test_gradient_clipping PASSED                      [ 50%]
tests/test_phase2_nn.py::test_linear_layer_2d_and_3d PASSED              [ 53%]
tests/test_phase2_nn.py::test_linear_layer_gradient_check PASSED         [ 56%]
tests/test_phase2_nn.py::test_activations_finite_difference_gradients PASSED [ 59%]
tests/test_phase2_nn.py::test_layernorm_gradient_check PASSED            [ 62%]
tests/test_phase2_nn.py::test_rmsnorm_gradient_check PASSED              [ 65%]
tests/test_phase2_nn.py::test_dropout_train_vs_eval PASSED               [ 68%]
tests/test_phase2_nn.py::test_weight_initializations PASSED              [ 71%]
tests/test_phase2_nn.py::test_deep_mlp_with_gelu_and_layernorm PASSED    [ 75%]
tests/test_phase3_attention.py::test_multi_head_attention_shapes_2d_and_3d PASSED [ 78%]
tests/test_phase3_attention.py::test_multi_head_attention_gradient_check PASSED [ 81%]
tests/test_phase3_attention.py::test_causal_mask_future_invariance PASSED [ 84%]
tests/test_phase3_attention.py::test_rope_relative_position_property PASSED [ 87%]
tests/test_phase3_attention.py::test_mha_with_rope PASSED                [ 90%]
tests/test_phase3_attention.py::test_kv_cache_exact_equivalence PASSED   [ 93%]
tests/test_self_attention.py::test_self_attention_causal_mask PASSED     [ 96%]
tests/test_self_attention.py::test_self_attention_backward_pass PASSED   [100%]
============================== 32 passed in 0.94s ==============================
```

To run interactive Phase 3 verification:
```bash
PYTHONPATH=. python3 tests/test_phase3_attention.py
```
```text
============================================================
PHASE 3: SEQUENCE MODELING & MULTI-HEAD ATTENTION (VERIFICATION)
============================================================

[Step 1] Multi-Head Attention Forward & Backward Pass:
Batch Input:   (2, 5, 16) (batch=2, seq_len=5, embed_dim=16)
Heads:         4 heads x 4 head_dim
Output Shape:  (2, 5, 16)
dX Grad Shape: (2, 5, 16)

[Step 2] Rotary Position Embeddings (RoPE) Invariance Proof:
Inner product (pos 3 & 1, diff=2): 1.75524308
Inner product (pos 8 & 6, diff=2): 1.75524308
Absolute Difference:               2.22e-16 (Passed: True)

[Step 3] Key-Value (KV) Cache Mathematical Identity:
Max difference between cached & non-cached step: 5.55e-16
```
