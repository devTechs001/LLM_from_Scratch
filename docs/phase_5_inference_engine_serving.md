# Phase 5: Inference Engine, Advanced Sampling & High-Throughput Serving from Scratch

> **Module**: `ml_lib.inference` (`sample_next_token`, `GenerationPipeline`)  
> **Test Suite**: `tests/test_phase5_inference.py`  
> **Status**: Verified (45/45 total repository tests passing across all 5 phases)

---

## 1. Executive Summary & The Serving Challenge

Training an LLM represents an upfront capital expenditure; **inference represents the ongoing operational expenditure (OpEx)**. In production, serving an autoregressive LLM presents fundamental computational and memory challenges:
1. **Memory-Bandwidth Bound**: Generating each token requires reading all model weights from GPU High-Bandwidth Memory (HBM) to SRAM to process a single token vector ($B=1$), resulting in low Arithmetic Intensity (FLOPs / Byte).
2. **KV Cache Memory Footprint**: Past keys and values must be preserved in memory for every concurrent user request.
3. **Degeneration vs Hallucination in Sampling**: Selecting how to sample next tokens dictates whether output is repetitive, hallucinatory, or coherent.

Phase 5 implements a full inference engine from first principles in NumPy, covering **Temperature Scaling**, **Top-$k$ Truncation**, **Top-$p$ (Nucleus) Sampling**, **Repetition Penalties**, **Streaming Generators**, and systems concepts like **PagedAttention** and **Speculative Decoding**.

---

## 2. Mathematical Formulations & Sampling Theory

Given raw model logits $z \in \mathbb{R}^V$ emitted by the language model head:

### 2.1 Temperature Scaling & Shannon Entropy

Temperature scaling parameterizes the sharpness of the probability distribution by dividing logits by scalar $T > 0$:
$$p_k(T) = \frac{e^{z_k / T}}{\sum_{j=1}^V e^{z_j / T}}$$

#### Information Entropy Analysis:
Shannon entropy measures uncertainty:
$$H(p) = -\sum_{k=1}^V p_k \ln p_k$$
* **$T \to 0$ (Greedy Decoding / Argmax)**:
  $$\lim_{T \to 0} p_k(T) = \begin{cases} 1 & \text{if } k = \arg\max_j z_j \\ 0 & \text{otherwise} \end{cases}$$
  Entropy $H(p) \to 0$. Output is strictly deterministic, but prone to repetitive loops and degenerate repetition.
* **$T = 1.0$**:
  Preserves raw model calibration.
* **$T \to \infty$**:
  All logits $z_k / T \to 0$, producing the maximum-entropy uniform distribution:
  $$p_k \to \frac{1}{V}, \quad H(p) \to \ln V$$

---

### 2.2 Top-$k$ Filtering (Fan et al., 2018)

Language models assign small non-zero probabilities to thousands of irrelevant or nonsensical tokens in the vocabulary tail. Top-$k$ truncates the candidate set to the $k$ most probable tokens:
$$V^{(k)} = \left\{ i \in V \mid \operatorname{rank}(z_i) \le k \right\}$$
$$\tilde{z}_i = \begin{cases} z_i & \text{if } i \in V^{(k)} \\ -\infty & \text{otherwise} \end{cases}$$
$$p_i = \operatorname{softmax}(\tilde{z})_i$$
* **Limitation of Fixed $k$**: When the model is highly confident (e.g. predicting the next word after `"The Eiffel Tower is in"` $\to$ `"Paris"`), $k=50$ permits low-quality tokens into the candidate pool. When the model faces high ambiguity, $k=50$ may prematurely exclude valid candidates.

---

### 2.3 Top-$p$ (Nucleus) Sampling (Holtzman et al., 2019)

Nucleus sampling dynamically adjusts the candidate pool size $M$ based on the model's certainty. Tokens are sorted by descending probability $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(V)}$. We define the nucleus $V^{(p)}$ as the smallest subset whose cumulative probability mass meets or exceeds threshold $p \in (0, 1]$:
$$V^{(p)} = \left\{ (1), (2), \dots, (M) \right\} \quad \text{where } M = \min \left\{ m \mid \sum_{i=1}^m p_{(i)} \ge p \right\}$$

Probabilities outside the nucleus are zeroed out and renormalized:
$$\tilde{p}_i = \begin{cases} \frac{p_i}{\sum_{j \in V^{(p)}} p_j} & \text{if } i \in V^{(p)} \\ 0 & \text{otherwise} \end{cases}$$
* **Dynamic Behavior**:
  * Confident predictions $\implies$ peaked distribution $\implies M$ shrinks to 1 or 2 tokens.
  * Uncertain predictions $\implies$ flat distribution $\implies M$ expands to hundreds of candidate tokens.

---

### 2.4 Repetition Penalty (Keskar et al., 2019)

Neural text generators often get stuck in repetitive cycles (e.g., repeating the same sentence or phrase indefinitely). The repetition penalty discounts the logits of tokens that have already appeared in the generated history $\mathcal{H}$:
$$\tilde{z}_k = \begin{cases} 
z_k / \theta & \text{if } z_k > 0 \text{ and } k \in \mathcal{H} \\ 
z_k \cdot \theta & \text{if } z_k < 0 \text{ and } k \in \mathcal{H} \\
z_k & \text{if } k \notin \mathcal{H}
\end{cases} \quad \text{where } \theta \ge 1.0$$
Because dividing a positive logit by $\theta > 1$ reduces it, and multiplying a negative logit by $\theta > 1$ makes it more negative, token $k$ is demoted in both regimes. Verified in `tests/test_phase5_inference.py`.

---

## 3. High-Throughput Serving & Systems Architecture

### 3.1 KV Cache Memory Footprint Analysis

For a model with $L$ layers, $H$ heads, head dimension $d_k$, sequence length $T$, batch size $B$, stored in FP16 (2 bytes per element):
$$\text{Memory}_{\text{KV}} = 2 \times (\text{Key} + \text{Value}) \times L \times H \times d_k \times B \times T \text{ bytes} = 4 \cdot L \cdot D \cdot B \cdot T \text{ bytes}$$

#### Concrete Example: LLaMA-70B
* $L = 80$, $D = 8,192$, $T = 8,192$, $B = 32$
* $\text{Memory}_{\text{KV}} = 4 \times 80 \times 8,192 \times 32 \times 8,192 \text{ bytes} \approx \mathbf{687.2 \text{ GB}}$
* The KV cache alone exceeds the memory footprint of the 140 GB model weights!

---

### 3.2 Advanced Serving Technologies

1. **Continuous Batching (Orca / vLLM)**:
   Traditional batching waits for the slowest request in a batch to finish (request-level batching), wasting GPU cycles on finished slots. Continuous batching schedules iterations at the token level: completed requests exit immediately and new prompts enter the batch at every decoding step.
2. **PagedAttention (Kwon et al., 2023)**:
   Standard memory allocation reserves contiguous memory for each request up to maximum context length, wasting 60–80% of memory due to fragmentation and over-reservation. PagedAttention divides the KV cache into fixed-size virtual blocks mapped to non-contiguous physical pages in DRAM (analogous to virtual memory in operating systems), reducing memory waste to $< 4\%$.
3. **Speculative Decoding (Leviathan et al., 2023)**:
   A fast small "draft" model (e.g. 1B) predicts $K$ candidate tokens sequentially. The large target model (e.g. 70B) evaluates all $K$ tokens simultaneously in a single forward pass ($O(1)$ verification) using causal attention. Valid tokens are accepted; invalid tokens are rejected. Yields a $2\times - 3\times$ latency reduction with mathematically identical output distribution.
4. **Model Quantization**:
   * **Weight-Only (AWQ / GPTQ)**: Compresses weights from FP16 (16-bit) to INT4 (4-bit), reducing memory footprint by $4\times$ and speeding up memory-bound inference.
   * **Weight-Activation (SmoothQuant / FP8)**: Scales both weights and activations to 8-bit integers/floats, utilizing specialized Tensor Core matrix engines.

---

## 4. Verification Test Results

Running `pytest -v` across the entire workspace verifies all 45 passing unit tests:

```text
tests/test_llm_embeddings.py::test_char_tokenizer_encode_decode PASSED   [  2%]
tests/test_llm_embeddings.py::test_embedding_forward_and_backward PASSED [  4%]
tests/test_minigpt.py::test_minigpt_convergence_and_generation PASSED    [  6%]
tests/test_multiclass_nn.py::test_multiclass_mlp_convergence PASSED      [  8%]
tests/test_nn_xor.py::test_xor_non_linear_convergence PASSED             [ 11%]
tests/test_nn_xor.py::test_linear_layer_alone_fails_xor PASSED           [ 13%]
tests/test_phase1.py::test_mse_loss_and_gradient PASSED                  [ 15%]
tests/test_phase1.py::test_numerical_gradient_verification_mse PASSED    [ 17%]
tests/test_phase1.py::test_huber_loss_and_gradient PASSED                [ 20%]
tests/test_phase1.py::test_sigmoid_numerical_stability PASSED            [ 22%]
tests/test_phase1.py::test_binary_cross_entropy_stability PASSED         [ 24%]
tests/test_phase1.py::test_log_sum_exp_stability PASSED                  [ 26%]
tests/test_phase1.py::test_softmax_and_cross_entropy_gradient PASSED     [ 28%]
tests/test_phase1.py::test_gradient_descent_convergence PASSED           [ 31%]
tests/test_phase1.py::test_sgd_momentum_convergence PASSED               [ 33%]
tests/test_phase1.py::test_adam_and_adamw_convergence PASSED             [ 35%]
tests/test_phase1.py::test_gradient_clipping PASSED                      [ 37%]
tests/test_phase2_nn.py::test_linear_layer_2d_and_3d PASSED              [ 40%]
tests/test_phase2_nn.py::test_linear_layer_gradient_check PASSED         [ 42%]
tests/test_phase2_nn.py::test_activations_finite_difference_gradients PASSED [ 44%]
tests/test_phase2_nn.py::test_layernorm_gradient_check PASSED            [ 46%]
tests/test_phase2_nn.py::test_rmsnorm_gradient_check PASSED              [ 48%]
tests/test_phase2_nn.py::test_dropout_train_vs_eval PASSED               [ 51%]
tests/test_phase2_nn.py::test_weight_initializations PASSED              [ 53%]
tests/test_phase2_nn.py::test_deep_mlp_with_gelu_and_layernorm PASSED    [ 55%]
tests/test_phase3_attention.py::test_multi_head_attention_shapes_2d_and_3d PASSED [ 57%]
tests/test_phase3_attention.py::test_multi_head_attention_gradient_check PASSED [ 60%]
tests/test_phase3_attention.py::test_causal_mask_future_invariance PASSED [ 62%]
tests/test_phase3_attention.py::test_rope_relative_position_property PASSED [ 64%]
tests/test_phase3_attention.py::test_mha_with_rope PASSED                [ 66%]
tests/test_phase3_attention.py::test_kv_cache_exact_equivalence PASSED   [ 68%]
tests/test_phase4_transformer.py::test_transformer_block_forward_backward_shapes PASSED [ 71%]
tests/test_phase4_transformer.py::test_residual_identity_gradient_highway PASSED [ 73%]
tests/test_phase4_transformer.py::test_transformer_lm_multi_layer_rope PASSED [ 75%]
tests/test_phase4_transformer.py::test_transformer_lm_kv_cache_generation PASSED [ 77%]
tests/test_phase4_transformer.py::test_transformer_pretraining_convergence PASSED [ 80%]
tests/test_phase5_inference.py::test_greedy_sampling_deterministic PASSED [ 82%]
tests/test_phase5_inference.py::test_temperature_entropy_effect PASSED   [ 84%]
tests/test_phase5_inference.py::test_top_k_filtering PASSED              [ 86%]
tests/test_phase5_inference.py::test_top_p_nucleus_filtering PASSED      [ 88%]
tests/test_phase5_inference.py::test_repetition_penalty PASSED           [ 91%]
tests/test_phase5_inference.py::test_generation_pipeline_streaming_and_stop_words PASSED [ 93%]
tests/test_phase5_inference.py::test_generation_pipeline_full_generation PASSED [ 95%]
tests/test_self_attention.py::test_self_attention_causal_mask PASSED     [ 97%]
tests/test_self_attention.py::test_self_attention_backward_pass PASSED   [100%]
============================== 45 passed in 1.99s ==============================
```

To run the interactive Phase 5 demonstration:
```bash
PYTHONPATH=. python3 tests/test_phase5_inference.py
```
```text
============================================================
PHASE 5: INFERENCE ENGINE & ADVANCED SAMPLING (VERIFICATION)
============================================================

[Step 1] Token Sampling Verification:
Logits: [ 2.   5.   4.8  0.5 -3. ]
Greedy Argmax (T=0):  Index 1
Top-K (k=2, T=1.0):   Index 1
Top-P (p=0.8, T=1.0): Index 1
Repetition Penalty:   Index 2 (demoted index 1)

[Step 2] Generation Pipeline Streaming Demo:
Streaming generation for prompt 'deep':
Prompt: 'deep' -> Output: llelggogcearrreierraggaar
```
