# Phase 4: Transformer Architecture & MiniGPT Pre-training from Scratch

> **Module**: `ml_lib.llm` (`TransformerBlock`, `TransformerLM`, `MiniGPT`)  
> **Test Suites**: `tests/test_phase4_transformer.py`, `tests/test_minigpt.py`  
> **Status**: Verified (38/38 total repository tests passing, 100% bit-for-bit KV-cache generation match)

---

## 1. Executive Summary: The Modern Decoder-Only Architecture

State-of-the-art Large Language Models (GPT-4, LLaMA-3, Mistral, Claude, DeepSeek) are **decoder-only autoregressive Transformers**. The model processes a sequence of discrete token IDs $x_{1:T}$, projects them into dense vector space, routes representations through stacked Transformer blocks, and emits probability distributions over the vocabulary for the next token:
$$P(x_{t} \mid x_{1:t-1}) = \operatorname{softmax}\left( \text{LM\_Head}(h_t^{(L)}) \right)$$

Phase 4 constructs this complete architecture from first principles in pure NumPy, featuring **Pre-LayerNorm/Pre-RMSNorm**, **Rotary Position Embeddings (RoPE)**, **Residual Skip Connections**, **GELU Feed-Forward Networks**, and **Multi-Layer KV Caching**.

---

## 2. Mathematical Formulations & Architectural Derivations

### 2.1 The Pre-LN Gradient Highway

The original Transformer (Vaswani et al., 2017) utilized **Post-LN**:
$$x_{l+1} = \operatorname{Norm}(x_l + \operatorname{SubLayer}(x_l))$$

In deep networks ($L \ge 12$), Post-LN normalizes the residual sum at every layer. The normalization scale suppresses the identity gradient, causing backpropagated gradients to vanish in early layers unless delicate learning rate warm-up schedules are used.

Modern LLMs utilize **Pre-LN** (or Pre-RMSNorm):
$$h_l = x_l + \operatorname{Attention}(\operatorname{Norm}_1(x_l))$$
$$x_{l+1} = h_l + \operatorname{FFN}(\operatorname{Norm}_2(h_l))$$

#### Mathematical Proof of the Unimpeded Identity Gradient Highway:
Expanding the recurrence across $L$ layers:
$$x_L = x_0 + \sum_{l=0}^{L-1} \left[ \operatorname{Attention}(\operatorname{Norm}_1(x_l)) + \operatorname{FFN}(\operatorname{Norm}_2(h_l)) \right]$$

Differentiating the final representation $x_L$ directly with respect to input embedding $x_0$:
$$\frac{\partial x_L}{\partial x_0} = \mathbf{I} + \sum_{l=0}^{L-1} \frac{\partial}{\partial x_0} \left[ \operatorname{Attention}(\operatorname{Norm}_1(x_l)) + \operatorname{FFN}(\operatorname{Norm}_2(h_l)) \right]$$

The identity matrix $\mathbf{I}$ guarantees that gradients flow directly from layer $L$ to layer $0$ without vanishing, regardless of network depth. In `tests/test_phase4_transformer.py`, we verify that even when sublayer weights are scaled down to near-zero, $\frac{\partial x_L}{\partial x_0} \approx \mathbf{I}$ with relative error $< 0.05$.

---

### 2.2 Feed-Forward Networks as Associative Key-Value Memories

While self-attention routes information *between* tokens, the Feed-Forward Network (FFN) operates on each token position *independently*:
$$\operatorname{FFN}(x) = W_2 \cdot \operatorname{GELU}(W_1 x + b_1) + b_2$$
where $W_1 \in \mathbb{R}^{D \times 4D}$ and $W_2 \in \mathbb{R}^{4D \times D}$.

* **Memory Interpretation** (Geva et al., 2021):
  * $W_1$ acts as a bank of **keys** detecting specific patterns or concepts.
  * The non-linear activation acts as a threshold indicator.
  * $W_2$ acts as a bank of **values** retrieving factual attributes associated with activated keys.
* FFN parameters compose approximately **$\frac{2}{3}$ of the total weights** in standard Transformer architectures.

---

### 2.3 Weight Tying (Press & Wolf, 2017)

In language modeling, the input embedding maps token ID $k \to e_k \in \mathbb{R}^D$, while the final linear language model head projects hidden states $h \in \mathbb{R}^D \to z_k = h \cdot w_k^T \in \mathbb{R}^V$.

Under weight tying:
$$W_{\text{lm\_head}} = W_{\text{embedding}}^T$$
* **Semantic Duality**: The vector representation representing token $k$ as an input is identical to the vector representation representing token $k$ as a prediction target.
* **Parameter Efficiency**: Reduces parameter count by $V \times D$ (e.g. for vocabulary $V = 32,000$ and $D = 4,096$, this saves over 131 million parameters).

---

### 2.4 Pre-Training Objective: Causal Language Modeling

The pre-training objective minimizes Categorical Cross-Entropy across sequence tokens (next-token prediction):
$$\mathcal{L}(\theta) = -\frac{1}{T-1} \sum_{t=1}^{T-1} \ln P_\theta(x_{t+1} \mid x_1, \dots, x_t)$$

Combined with **AdamW** and decoupled weight decay $\lambda$:
$$\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
and global gradient norm clipping $\|\mathbf{g}\|_2 \le M$, preventing loss divergence.

---

### 2.5 Multi-Layer Key-Value (KV) Caching

During autoregressive generation, each Transformer block maintains its own persistent $(K, V)$ cache:
```
Prompt:  [x_0, x_1, x_2] -> Prefill phase -> populate k_cache, v_cache across layers 1..L
Step 1:  [x_3]           -> Incremental decode -> append k_3, v_3 to cache -> emit x_4
Step 2:  [x_4]           -> Incremental decode -> append k_4, v_4 to cache -> emit x_5
```

In `tests/test_phase4_transformer.py`, we verify that the multi-layer KV-cached output matches full non-cached generation **bit-for-bit**:
```python
cached_gen = model.generate(start_tokens, max_new_tokens=6, use_cache=True)
non_cached_gen = model.generate(start_tokens, max_new_tokens=6, use_cache=False)
assert np.array_equal(cached_gen, non_cached_gen)  # Exactly identical
```

---

## 3. Verification Test Results

Running `pytest -v` across the entire workspace executes all 38 passing unit tests:

```text
tests/test_llm_embeddings.py::test_char_tokenizer_encode_decode PASSED   [  2%]
tests/test_llm_embeddings.py::test_embedding_forward_and_backward PASSED [  5%]
tests/test_minigpt.py::test_minigpt_convergence_and_generation PASSED    [  7%]
tests/test_multiclass_nn.py::test_multiclass_mlp_convergence PASSED      [ 10%]
tests/test_nn_xor.py::test_xor_non_linear_convergence PASSED             [ 13%]
tests/test_nn_xor.py::test_linear_layer_alone_fails_xor PASSED           [ 15%]
tests/test_phase1.py::test_mse_loss_and_gradient PASSED                  [ 18%]
tests/test_phase1.py::test_numerical_gradient_verification_mse PASSED    [ 21%]
tests/test_phase1.py::test_huber_loss_and_gradient PASSED                [ 23%]
tests/test_phase1.py::test_sigmoid_numerical_stability PASSED            [ 26%]
tests/test_phase1.py::test_binary_cross_entropy_stability PASSED         [ 28%]
tests/test_phase1.py::test_log_sum_exp_stability PASSED                  [ 31%]
tests/test_phase1.py::test_softmax_and_cross_entropy_gradient PASSED     [ 34%]
tests/test_phase1.py::test_gradient_descent_convergence PASSED           [ 36%]
tests/test_phase1.py::test_sgd_momentum_convergence PASSED               [ 39%]
tests/test_phase1.py::test_adam_and_adamw_convergence PASSED             [ 42%]
tests/test_phase1.py::test_gradient_clipping PASSED                      [ 44%]
tests/test_phase2_nn.py::test_linear_layer_2d_and_3d PASSED              [ 47%]
tests/test_phase2_nn.py::test_linear_layer_gradient_check PASSED         [ 50%]
tests/test_phase2_nn.py::test_activations_finite_difference_gradients PASSED [ 52%]
tests/test_phase2_nn.py::test_layernorm_gradient_check PASSED            [ 55%]
tests/test_phase2_nn.py::test_rmsnorm_gradient_check PASSED              [ 57%]
tests/test_phase2_nn.py::test_dropout_train_vs_eval PASSED               [ 60%]
tests/test_phase2_nn.py::test_weight_initializations PASSED              [ 63%]
tests/test_phase2_nn.py::test_deep_mlp_with_gelu_and_layernorm PASSED    [ 65%]
tests/test_phase3_attention.py::test_multi_head_attention_shapes_2d_and_3d PASSED [ 68%]
tests/test_phase3_attention.py::test_multi_head_attention_gradient_check PASSED [ 71%]
tests/test_phase3_attention.py::test_causal_mask_future_invariance PASSED [ 73%]
tests/test_phase3_attention.py::test_rope_relative_position_property PASSED [ 76%]
tests/test_phase3_attention.py::test_mha_with_rope PASSED                [ 78%]
tests/test_phase3_attention.py::test_kv_cache_exact_equivalence PASSED   [ 81%]
tests/test_phase4_transformer.py::test_transformer_block_forward_backward_shapes PASSED [ 84%]
tests/test_phase4_transformer.py::test_residual_identity_gradient_highway PASSED [ 86%]
tests/test_phase4_transformer.py::test_transformer_lm_multi_layer_rope PASSED [ 89%]
tests/test_phase4_transformer.py::test_transformer_lm_kv_cache_generation PASSED [ 92%]
tests/test_phase4_transformer.py::test_transformer_pretraining_convergence PASSED [ 94%]
tests/test_self_attention.py::test_self_attention_causal_mask PASSED     [ 97%]
tests/test_self_attention.py::test_self_attention_backward_pass PASSED   [100%]
============================== 38 passed in 1.99s ==============================
```

To run interactive Phase 4 pre-training and text generation:
```bash
PYTHONPATH=. python3 tests/test_phase4_transformer.py
```
```text
============================================================
PHASE 4: TRANSFORMER ARCHITECTURE & PRE-TRAINING (VERIFICATION)
============================================================

[Step 1] Pre-training Corpus:
Corpus:     'attention is all you need for deep learning transformers from scratch'
Vocab Size: 19 unique characters
Sequence:   69 tokens

[Step 2] Initializing 2-Layer TransformerLM (Pre-RMSNorm + RoPE):

[Step 3] Pre-training Convergence:
 Epoch   1 | Cross-Entropy Loss: 3.5018
 Epoch  50 | Cross-Entropy Loss: 0.8130
 Epoch 100 | Cross-Entropy Loss: 0.9388
 Epoch 150 | Cross-Entropy Loss: 0.4352
 Epoch 200 | Cross-Entropy Loss: 0.5562

[Step 4] Fast KV-Cached Text Generation:
Prompt:    'attention'
Generated: 'attention norn ion all s alneeen nhad fhtn s n ansean '
```
