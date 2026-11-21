# Phase 1: Mathematical Foundations, Optimization & Numerical Stability from Scratch

> **Module**: `ml_lib.losses`, `ml_lib.optimizers`  
> **Test Suite**: `tests/test_phase1.py`  
> **Status**: Verified (11/11 tests passing, analytical vs numerical gradient relative error $< 10^{-10}$)

---

## 1. Executive Summary & Role in AI Systems

Every artificial intelligence system—from a simple linear regressor to trillion-parameter autoregressive Large Language Models (LLMs)—relies on two foundational pillars:
1. **Objective (Loss) Functions**: Scalar functions $\mathcal{L}(\theta)$ quantifying prediction error and establishing the optimization landscape.
2. **First-Order Optimization Algorithms**: Rules that update parameters $\theta \leftarrow \theta + \Delta \theta$ by following negative gradients $-\nabla_\theta \mathcal{L}$ through high-dimensional non-convex manifolds.

In modern LLMs, training stability is dominated by numerical precision: preventing floating-point overflow ($e^{z} \to \infty$), preventing $\log(0) \to -\infty$ (producing `NaN`), and clipping exploding gradients. Phase 1 constructs these foundational primitives entirely from scratch in pure NumPy with machine-precision verification.

---

## 2. Mathematical Formulations & Derivations

### 2.1 Mean Squared Error (MSE)
Used for continuous target regression and basic parameter alignment:
$$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

#### Analytic Gradient with respect to Predictions $\hat{y}$:
Applying the chain rule:
$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{\partial}{\partial \hat{y}_i} \left[ \frac{1}{n} (y_i - \hat{y}_i)^2 \right] = \frac{2}{n} (\hat{y}_i - y_i) (-1) \cdot (-1) = \frac{2}{n} (\hat{y}_i - y_i)$$

In vectorized form:
$$\nabla_{\hat{y}} \mathcal{L}_{\text{MSE}} = \frac{2}{n} (\hat{y} - y)$$

---

### 2.2 Huber Loss (Smooth $L_1$)
Real-world datasets contain anomalies and outliers. While MSE squares errors (making it sensitive to extreme points), Huber loss transitions quadratically near zero and linearly in the tails:
$$\mathcal{L}_\delta(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^n \ell_\delta(y_i - \hat{y}_i)$$
$$\ell_\delta(r) = \begin{cases} 
\frac{1}{2} r^2 & \text{if } |r| \le \delta \\
\delta (|r| - \frac{1}{2}\delta) & \text{if } |r| > \delta
\end{cases}$$

#### Gradient:
$$\frac{\partial \ell_\delta}{\partial \hat{y}_i} = \begin{cases} 
\frac{1}{n} (\hat{y}_i - y_i) & \text{if } |\hat{y}_i - y_i| \le \delta \\
\frac{\delta}{n} \operatorname{sgn}(\hat{y}_i - y_i) & \text{if } |\hat{y}_i - y_i| > \delta
\end{cases}$$

---

### 2.3 Logistic Sigmoid & Binary Cross-Entropy (BCE)

#### Logistic Sigmoid:
Maps arbitrary real-valued logits $z \in \mathbb{R}$ to valid probabilities $p \in (0, 1)$:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Numerical Safeguard**: If $z < -709$ in IEEE 754 64-bit float, $e^{-z} \to \infty$ causing overflow. We bound $z \in [-250, 250]$:
$$\sigma_{\text{stable}}(z) = \frac{1}{1 + e^{-\operatorname{clip}(z, -250, 250)}}$$

#### Binary Cross-Entropy Loss:
Derived from the negative log-likelihood of a Bernoulli distribution:
$$\mathcal{L}_{\text{BCE}}(y, p) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

**Numerical Safeguard**: If $p_i = 0$ or $p_i = 1$, $\ln(0)$ evaluates to $-\infty$ and yields `NaN` gradients. We clip predictions into $[\epsilon, 1 - \epsilon]$ where $\epsilon = 10^{-15}$.

---

### 2.4 Softmax & Categorical Cross-Entropy (Next-Token Loss in LLMs)

In language models, the output layer emits a vector of raw unnormalized logits $z \in \mathbb{R}^V$ across a vocabulary of size $V$. The probability of token $k$ is given by Softmax:
$$p_k = \frac{e^{z_k}}{\sum_{j=1}^V e^{z_j}}$$

#### The Log-Sum-Exp Trick:
Computing $e^{z_k}$ directly for logits $> 700$ results in floating-point overflow (`inf / inf = NaN`).
Let $c = \max_j z_j$:
$$p_k = \frac{e^{z_k - c} \cdot e^c}{\sum_{j=1}^V e^{z_j - c} \cdot e^c} = \frac{e^{z_k - c}}{\sum_{j=1}^V e^{z_j - c}}$$
Because $z_k - c \le 0$ for all $k$, the exponent is strictly $\le 0$, completely eliminating floating-point overflow.

#### Categorical Cross-Entropy Loss:
For true one-hot token indicator $y$ with target index $t$ ($y_t = 1$, $y_{j \ne t} = 0$):
$$\mathcal{L} = -\sum_{j=1}^V y_j \ln(p_j) = -\ln(p_t) = -\ln\left(\frac{e^{z_t}}{\sum_j e^{z_j}}\right) = \ln\left(\sum_{j=1}^V e^{z_j}\right) - z_t$$

#### The Elegant Combined Softmax + Cross-Entropy Gradient:
Differentiating $\mathcal{L}$ directly with respect to logit $z_k$:
$$\frac{\partial \mathcal{L}}{\partial z_k} = \frac{\partial}{\partial z_k} \left[ \ln\left(\sum_{j=1}^V e^{z_j}\right) - z_t \right] = \frac{e^{z_k}}{\sum_{j=1}^V e^{z_j}} - \mathbb{I}(k = t) = p_k - y_k$$

In batch form over $N$ tokens:
$$\nabla_z \mathcal{L} = \frac{1}{N} (P - Y)$$
This remarkable cancellation means we never calculate $\frac{\partial \mathcal{L}}{\partial p} \times \frac{\partial p}{\partial z}$ separately, eliminating division by tiny probabilities.

---

## 3. Mathematical Gradient Verification: Central Finite Differences

To ensure analytical backpropagation code is 100% bug-free, we employ two-sided central difference approximation:
$$g_{\text{numeric}, i} = \frac{\mathcal{L}(\theta + \epsilon e_i) - \mathcal{L}(\theta - \epsilon e_i)}{2\epsilon} = \frac{\partial \mathcal{L}}{\partial \theta_i} + \mathcal{O}(\epsilon^2)$$

Using Taylor series expansion:
$$\mathcal{L}(\theta + \epsilon) = \mathcal{L}(\theta) + \epsilon \mathcal{L}'(\theta) + \frac{\epsilon^2}{2}\mathcal{L}''(\theta) + \frac{\epsilon^3}{6}\mathcal{L}'''(\theta) + \dots$$
$$\mathcal{L}(\theta - \epsilon) = \mathcal{L}(\theta) - \epsilon \mathcal{L}'(\theta) + \frac{\epsilon^2}{2}\mathcal{L}''(\theta) - \frac{\epsilon^3}{6}\mathcal{L}'''(\theta) + \dots$$
Subtracting the two equations cancels even-order terms:
$$\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon) = 2\epsilon \mathcal{L}'(\theta) + \frac{\epsilon^3}{3}\mathcal{L}'''(\theta) + \dots$$
$$\frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon} = \mathcal{L}'(\theta) + \mathcal{O}(\epsilon^2)$$

The relative error metric:
$$\text{RelError}(g_a, g_n) = \frac{\|g_a - g_n\|_2}{\max(\|g_a\|_2 + \|g_n\|_2, \epsilon_0)}$$

In `test_phase1.py`, our analytical gradient achieved **$\text{RelError} = 1.53 \times 10^{-11}$** against numeric gradients, verifying exact mathematical correctness.

---

## 4. Optimization Algorithms

### 4.1 Vanilla Gradient Descent (GD)
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$
* **Limitation**: Suffers in ravines where surface curves much more steeply in one dimension than another, causing high-frequency oscillations.

### 4.2 SGD with Polyak Momentum
Introduces velocity $v_t$ that accumulates past gradients with friction coefficient $\beta \in [0, 1)$:
$$v_{t+1} = \beta v_t + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$
* **Physics Intuition**: Models a heavy ball rolling down a potential energy surface; momentum dampens oscillations perpendicular to the valley while accelerating along the slope.

### 4.3 RMSprop
Normalizes gradient step by running average of squared gradients $s_t$:
$$s_{t+1} = \beta s_t + (1 - \beta) (\nabla_\theta \mathcal{L})^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_{t+1}} + \epsilon} \nabla_\theta \mathcal{L}$$

### 4.4 Adam (Adaptive Moment Estimation)
Combines momentum (first raw moment $m_t$) and RMSprop (second uncentered moment $v_t$) with bias correction for initialization at zero:
$$m_{t+1} = \beta_1 m_t + (1 - \beta_1) g_t$$
$$v_{t+1} = \beta_2 v_t + (1 - \beta_2) g_t^2$$

#### Bias Correction Derivation:
Expanding $m_t$ assuming $g_i$ come from a stationary distribution with mean $\mathbb{E}[g]$:
$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$
$$\mathbb{E}[m_t] = \mathbb{E}\left[(1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i\right] = \mathbb{E}[g] (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[g] (1 - \beta_1^t)$$
Therefore, the unbiased estimator is:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 4.5 AdamW (Decoupled Weight Decay for LLMs)
In standard $L_2$ regularization, the objective is $\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{1}{2} \lambda \|\theta\|^2$.
In Adam, the gradient $g + \lambda \theta$ gets scaled by $\frac{1}{\sqrt{\hat{v}}}$. Parameters with large historical gradients experience suppressed weight decay, while parameters with small gradients get excessively penalized.

**AdamW** (Loshchilov & Hutter, 2017) decouples weight decay directly from gradient updates:
$$\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
This is the standard optimizer used to train GPT-3/4, LLaMA-1/2/3, Mistral, and Claude.

---

## 5. Real-World Production Techniques

### 5.1 Global Gradient Norm Clipping
During LLM pretraining, rare token combinations or high learning rates can induce sudden spikes in loss ("loss spikes"), sending gradient magnitudes to infinity. Global norm clipping rescales the full gradient vector:
$$\|g\|_2 = \sqrt{\sum_{i} \|g_i\|_2^2}$$
$$\tilde{g}_i = \begin{cases}
g_i & \text{if } \|g\|_2 \le M \\
g_i \cdot \frac{M}{\|g\|_2 + 10^{-6}} & \text{if } \|g\|_2 > M
\end{cases}$$
This preserves gradient direction while enforcing a strict maximum Euclidean step length $M$.

---

## 6. Verification Test Results

Running `pytest tests/test_phase1.py` executes 11 rigorous test cases:

```text
tests/test_phase1.py::test_mse_loss_and_gradient PASSED                  [  9%]
tests/test_phase1.py::test_numerical_gradient_verification_mse PASSED    [ 18%]
tests/test_phase1.py::test_huber_loss_and_gradient PASSED                [ 27%]
tests/test_phase1.py::test_sigmoid_numerical_stability PASSED            [ 36%]
tests/test_phase1.py::test_binary_cross_entropy_stability PASSED         [ 45%]
tests/test_phase1.py::test_log_sum_exp_stability PASSED                  [ 54%]
tests/test_phase1.py::test_softmax_and_cross_entropy_gradient PASSED     [ 63%]
tests/test_phase1.py::test_gradient_descent_convergence PASSED           [ 72%]
tests/test_phase1.py::test_sgd_momentum_convergence PASSED               [ 81%]
tests/test_phase1.py::test_adam_and_adamw_convergence PASSED             [ 90%]
tests/test_phase1.py::test_gradient_clipping PASSED                      [100%]
```

To run both the visual step-by-step progress and the automated test suite:
```bash
PYTHONPATH=. python3 tests/test_phase1.py
```
Output:
```text
============================================================
PHASE 1: FOUNDATIONS OF ML & OPTIMIZATION (VERIFICATION)
============================================================

[Step 1] Initial State:
Target:      [10. 20. 30.]
Prediction:  [2. 2. 2.]
Initial MSE: 390.6667
 Step  1 -> Loss: 340.3141 | Pred: [2.53 3.2  3.87]
 Step  5 -> Loss: 195.9630 | Pred: [ 4.33  7.25 10.17]
 Step 10 -> Loss: 98.2974 | Pred: [ 5.99 10.97 15.95]

Final MSE Loss after 10 steps: 98
Final predictions: [ 5.99 10.97 15.95]

[Step 2] Finite Difference Analytic Gradient Verification:
Analytic Gradient: [-1.333333 -2.       -2.      ]
Numeric Gradient:  [-1.333333 -2.       -2.      ]
Relative Error:    1.53e-11 (Passed: True)

[Step 3] Advanced Optimizers: AdamW Convergence Check:
Target Weights:   [ 5. -3.]
Learned Weights:  [ 5.25 -2.83]
```
