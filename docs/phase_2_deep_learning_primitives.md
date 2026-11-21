# Phase 2: Deep Learning Primitives & Non-Linear Representations from Scratch

> **Module**: `ml_lib.nn`  
> **Test Suites**: `tests/test_phase2_nn.py`, `tests/test_nn_xor.py`, `tests/test_multiclass_nn.py`  
> **Status**: Verified (22/22 total workspace tests passing, analytical vs numerical gradient relative error $< 10^{-10}$)

---

## 1. Executive Summary & Architectural Role

Linear models are fundamentally constrained: any sequence of linear operations collapses into a single linear projection:
$$\hat{y} = W_k (\dots (W_2 (W_1 x + b_1) + b_2) \dots) + b_k = W_{\text{effective}} x + b_{\text{effective}}$$

Deep learning resolves this by interleaving affine projections with non-linear activation functions $\sigma(\cdot)$ and normalization operators. Under the **Universal Approximation Theorem** (Cybenko, 1989; Hornik, 1991), a feedforward network with a single non-linear hidden layer can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to arbitrary precision.

In Large Language Models, feedforward blocks (MLPs) compose over 65% of the total parameter count, acting as key-value associative memories that store factual knowledge acquired during pretraining (Geva et al., 2021). Phase 2 implements and mathematically verifies these core neural network primitives from first principles in NumPy.

---

## 2. Mathematical Formulations & Derivations

### 2.1 The Non-Linearity Requirement: The XOR Problem
Consider the binary XOR function:
$$\text{XOR}(0, 0) = 0, \quad \text{XOR}(0, 1) = 1, \quad \text{XOR}(1, 0) = 1, \quad \text{XOR}(1, 1) = 0$$

For a single linear layer $y = \sigma(w_1 x_1 + w_2 x_2 + b)$ to classify correctly:
1. $w_1(0) + w_2(0) + b < 0 \implies b < 0$
2. $w_1(0) + w_2(1) + b \ge 0 \implies w_2 + b \ge 0$
3. $w_1(1) + w_2(0) + b \ge 0 \implies w_1 + b \ge 0$
4. $w_1(1) + w_2(1) + b < 0 \implies w_1 + w_2 + b < 0$

Adding inequalities (2) and (3):
$$(w_1 + b) + (w_2 + b) \ge 0 \implies w_1 + w_2 + 2b \ge 0$$
Since $b < 0$, $w_1 + w_2 + b > w_1 + w_2 + 2b \ge 0$, which contradicts inequality (4) ($w_1 + w_2 + b < 0$).
Thus, **no single linear hyperplane can separate XOR**. Adding a 2-layer MLP with non-linear activation (ReLU) warps the feature space, making the representations linearly separable (verified in `tests/test_nn_xor.py`).

---

### 2.2 Activation Functions & Exact Derivatives

#### A. ReLU (Rectified Linear Unit)
$$\text{ReLU}(x) = \max(0, x)$$
$$\frac{d}{dx} \text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$$
* **Advantage**: Constant gradient of 1 for positive activations, eliminating the vanishing gradient problem inherent in Sigmoid/Tanh for deep networks.
* **Dying ReLU**: If a neuron's weights update such that it is negative across all training samples, its gradient is permanently 0.

#### B. LeakyReLU
$$\text{LeakyReLU}(x) = \max(\alpha x, x), \quad \alpha \approx 0.01$$
$$\frac{d}{dx} \text{LeakyReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \le 0 \end{cases}$$
Ensures gradient flow even when inactive, preventing dying neurons.

#### C. GELU (Gaussian Error Linear Unit - GPT & BERT Standard)
Introduced by Hendrycks & Gimpel (2016), GELU weights inputs by their percentile in a standard normal distribution:
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$

Using the standard tanh approximation:
$$\text{GELU}(x) \approx 0.5 x \left( 1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right) \right)$$

##### Exact Analytical Backward Derivative:
Let $u = \sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)$ and $t = \tanh(u)$.
$$\frac{d u}{dx} = \sqrt{\frac{2}{\pi}} (1 + 3 \times 0.044715 x^2)$$
$$\frac{d}{dx} \text{GELU}(x) = 0.5 (1 + t) + 0.5 x (1 - t^2) \frac{du}{dx}$$
This analytical gradient is verified against numerical finite differences in `tests/test_phase2_nn.py` with relative error $< 10^{-10}$.

#### D. SiLU / Swish (LLaMA & Mistral Standard)
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
$$\frac{d}{dx}\text{SiLU}(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)(1 + x(1 - \sigma(x)))$$
Used in modern SwiGLU Transformer Feed-Forward Networks.

---

### 2.3 Normalization Primitives

#### Why Batch Normalization Fails in LLMs
Batch Normalization computes statistics across the mini-batch dimension:
$$\mu_B = \frac{1}{B} \sum_{i=1}^B x_i, \quad \sigma_B^2 = \frac{1}{B} \sum_{i=1}^B (x_i - \mu_B)^2$$
In NLP and LLMs:
1. Sentence lengths vary dynamically, making batch statistics noisy and padding-dependent.
2. During autoregressive token generation, batch size is often 1, making batch statistics undefined.
3. Batch dependencies introduce cross-sample leakage during distributed training.

#### A. Layer Normalization (Ba, Kiros & Hinton, 2016)
Normalizes across features/channels for each sample independently:
$$\mu = \frac{1}{D} \sum_{j=1}^D x_j, \quad \sigma^2 = \frac{1}{D} \sum_{j=1}^D (x_j - \mu)^2$$
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y = \hat{x} \odot \gamma + \beta$$

##### Exact Closed-Form Analytic Backward Pass:
Given incoming upstream gradient $\frac{\partial \mathcal{L}}{\partial y} = d_{\text{out}}$:
1. $d\gamma = \sum d_{\text{out}} \odot \hat{x}$
2. $d\beta = \sum d_{\text{out}}$
3. $d\hat{x} = d_{\text{out}} \odot \gamma$
4. Input gradient:
$$\frac{\partial \mathcal{L}}{\partial x} = \frac{1}{D \sqrt{\sigma^2 + \epsilon}} \left[ D \cdot d\hat{x} - \sum_{j=1}^D d\hat{x}_j - \hat{x} \sum_{j=1}^D (d\hat{x}_j \cdot \hat{x}_j) \right]$$

---

#### B. RMSNorm (Root Mean Square Normalization - LLaMA, Mistral, Gemma)
Zhang & Sennrich (2019) demonstrated that the re-centering property (subtracting the mean $\mu$) in LayerNorm is computationally redundant. Normalizing solely by root-mean-square scales activations identically while saving ~7% GPU memory and computation:
$$\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{j=1}^D x_j^2 + \epsilon}$$
$$\hat{x} = \frac{x}{\text{RMS}(x)}, \quad y = \hat{x} \odot \gamma$$

##### Exact Closed-Form Backward Pass:
$$d\gamma = \sum d_{\text{out}} \odot \hat{x}$$
$$\frac{\partial \mathcal{L}}{\partial x} = \frac{1}{\text{RMS}(x)} \left[ d_{\text{out}} \odot \gamma - \hat{x} \left(\frac{1}{D} \sum_{j=1}^D (d_{\text{out}} \odot \gamma \odot \hat{x})_j \right) \right]$$

---

### 2.4 Weight Initialization Theory

If weights are initialized too small, signal variance exponentially vanishes across deep layers: $\lim_{L \to \infty} \text{Var}(z_L) = 0$.  
If weights are initialized too large, signal variance explodes: $\lim_{L \to \infty} \text{Var}(z_L) = \infty$.

For linear layer $y = W x$ with $d_{\text{in}}$ inputs, assuming independent zero-mean weights and inputs:
$$\text{Var}(y_i) = \sum_{j=1}^{d_{\text{in}}} \text{Var}(W_{ij} x_j) = d_{\text{in}} \text{Var}(W) \text{Var}(x)$$

1. **Glorot / Xavier Initialization (for Tanh / Sigmoid / Linear)**:
   To preserve variance in both forward and backward passes ($\text{Var}(y) = \text{Var}(x)$ and $\text{Var}(\nabla_x) = \text{Var}(\nabla_y)$):
   $$\text{Var}(W) = \frac{2}{d_{\text{in}} + d_{\text{out}}} \implies W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\text{in}} + d_{\text{out}}}}\right)$$
2. **He / Kaiming Initialization (for ReLU / GELU)**:
   Because ReLU zeros out negative activations, it halves signal variance: $\mathbb{E}[\text{ReLU}(x)^2] = \frac{1}{2} \text{Var}(x)$. To compensate:
   $$\text{Var}(W) = \frac{2}{d_{\text{in}}} \implies W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\text{in}}}}\right)$$

---

## 3. Multidimensional Tensor Handling in Linear Layers

In Transformer architectures, inputs to dense projections are sequence matrices of shape:
$$\text{Shape}(X) = (B, T, d_{\text{in}})$$
where $B$ is batch size, $T$ is sequence length, and $d_{\text{in}}$ is embedding dimension.

Our `Linear` module natively flattens the leading dimensions during backpropagation:
```python
X_flat = self.inputs.reshape(-1, self.in_features)
d_out_flat = d_output.reshape(-1, self.out_features)

self.d_weights = np.dot(X_flat.T, d_out_flat)
self.d_bias = np.sum(d_out_flat, axis=0, keepdims=True)
d_input = np.dot(d_out_flat, self.weights.T).reshape(self.inputs.shape)
```
This enables transparent compatibility across standard 2D MLPs and 3D Transformer blocks.

---

## 4. Verification Test Results

Running `pytest -v` across the entire repository executes 22 passing tests:

```text
tests/test_multiclass_nn.py::test_multiclass_mlp_convergence PASSED      [  4%]
tests/test_nn_xor.py::test_xor_non_linear_convergence PASSED             [  9%]
tests/test_nn_xor.py::test_linear_layer_alone_fails_xor PASSED           [ 13%]
tests/test_phase1.py::test_mse_loss_and_gradient PASSED                  [ 18%]
tests/test_phase1.py::test_numerical_gradient_verification_mse PASSED    [ 22%]
tests/test_phase1.py::test_huber_loss_and_gradient PASSED                [ 27%]
tests/test_phase1.py::test_sigmoid_numerical_stability PASSED            [ 31%]
tests/test_phase1.py::test_binary_cross_entropy_stability PASSED         [ 36%]
tests/test_phase1.py::test_log_sum_exp_stability PASSED                  [ 40%]
tests/test_phase1.py::test_softmax_and_cross_entropy_gradient PASSED     [ 45%]
tests/test_phase1.py::test_gradient_descent_convergence PASSED           [ 50%]
tests/test_phase1.py::test_sgd_momentum_convergence PASSED               [ 54%]
tests/test_phase1.py::test_adam_and_adamw_convergence PASSED             [ 59%]
tests/test_phase1.py::test_gradient_clipping PASSED                      [ 63%]
tests/test_phase2_nn.py::test_linear_layer_2d_and_3d PASSED              [ 68%]
tests/test_phase2_nn.py::test_linear_layer_gradient_check PASSED         [ 72%]
tests/test_phase2_nn.py::test_activations_finite_difference_gradients PASSED [ 77%]
tests/test_phase2_nn.py::test_layernorm_gradient_check PASSED            [ 81%]
tests/test_phase2_nn.py::test_rmsnorm_gradient_check PASSED              [ 86%]
tests/test_phase2_nn.py::test_dropout_train_vs_eval PASSED               [ 90%]
tests/test_phase2_nn.py::test_weight_initializations PASSED              [ 95%]
tests/test_phase2_nn.py::test_deep_mlp_with_gelu_and_layernorm PASSED    [100%]
============================== 22 passed in 0.85s ==============================
```

To run the interactive Phase 2 verification:
```bash
PYTHONPATH=. python3 tests/test_phase2_nn.py
```
Output:
```text
============================================================
PHASE 2: DEEP LEARNING PRIMITIVES & NON-LINEARITY (VERIFICATION)
============================================================

[Step 1] Linear Layer Multidimensional Tensor Verification:
Input Shape:  (2, 6, 8) (batch=2, seq_len=6, embed_dim=8)
Output Shape: (2, 6, 4) (batch=2, seq_len=6, out_dim=4)
Grad dX Shape: (2, 6, 8)

[Step 2] Modern LLM Activations (GELU vs SiLU vs ReLU):
Inputs: [-2. -1.  0.  1.  2.]
GELU:   [-0.0454 -0.1588  0.      0.8412  1.9546]
SiLU:   [-0.2384 -0.2689  0.      0.7311  1.7616]
ReLU:   [0. 0. 0. 1. 2.]

[Step 3] Normalization: LayerNorm vs RMSNorm:
Raw Input:        [[10. 20. 30. 40.]]
LayerNorm Output: [[-1.3416 -0.4472  0.4472  1.3416]] (mean=0.00, var=1.00)
RMSNorm Output:   [[0.3651 0.7303 1.0954 1.4606]]
```
