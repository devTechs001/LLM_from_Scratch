# Phase 6: Spatial Representations, Vision Transformers & Multimodal Systems from Scratch

> **Module**: `ml_lib.vision` (`PatchEmbedding`, `VisionTransformer`, `ContrastiveLoss`, `l2_normalize`)  
> **Test Suite**: `tests/test_phase6_vision.py`  
> **Status**: Verified (50/50 total repository tests passing across all 6 phases)

---

## 1. Executive Summary: The Convergence of Modalities

Historically, machine learning was fragmented across incompatible architectural paradigms:
* **Computer Vision**: Convolutional Neural Networks (CNNs: AlexNet, ResNet).
* **Natural Language Processing**: Recurrent Networks (LSTMs, GRUs).
* **Audio & Signals**: 1D Convolutions and Spectrogram WaveNets.

The discovery of the **Vision Transformer (ViT)** (Dosovitskiy et al., 2020) and **Contrastive Language-Image Pre-training (CLIP)** (Radford et al., 2021) demonstrated that **all data modalities can be projected into sequences of tokens processed by the exact same Transformer architecture**.

Phase 6 implements this unified spatial and multimodal architecture from first principles in pure NumPy, covering **Patch Embeddings**, **Vision Transformers**, and **Symmetric Contrastive InfoNCE Alignment**.

---

## 2. Mathematical Formulations & Architectural Derivations

### 2.1 Patch Embedding: Tokenizing Continuous Spatial Pixels

Images exist as continuous 2D spatial grids $X \in \mathbb{R}^{B \times C \times H \times W}$. Standard Self-Attention computes $\mathcal{O}(T^2)$ pairwise interactions. Applying self-attention directly to individual pixels (e.g. $224 \times 224 = 50,176$ pixels) would require a prohibitively massive attention matrix of $(50,176)^2 \approx 2.5 \times 10^9$ elements per sample.

ViT solves this by slicing images into non-overlapping patches of size $P \times P$:
$$N = \left( \frac{H}{P} \right) \times \left( \frac{W}{P} \right)$$
Flattening each patch into a vector of dimension $D_{\text{patch}} = C \cdot P^2$, we obtain:
$$X_{\text{patches}} \in \mathbb{R}^{B \times N \times (C \cdot P^2)}$$

Each patch vector is projected into the Transformer's hidden embedding dimension $D$ via a trainable linear projection $W_{\text{proj}} \in \mathbb{R}^{(C \cdot P^2) \times D}$:
$$X_{\text{emb}} = X_{\text{patches}} \cdot W_{\text{proj}} + b_{\text{proj}} \in \mathbb{R}^{B \times N \times D}$$

#### Exact Backpropagation Inversion (`unpatchify`):
Given upstream gradient $d_{\text{patches}} \in \mathbb{R}^{B \times N \times (C \cdot P^2)}$, we invert the patch rearrangement to reconstruct pixel-space gradients:
$$d_{\text{images}} = \operatorname{reshape}\left( \operatorname{transpose}\left( \operatorname{reshape}(d_{\text{patches}}, [B, \frac{H}{P}, \frac{W}{P}, C, P, P]), [0, 3, 1, 4, 2, 5] \right), [B, C, H, W] \right)$$
Verified in `tests/test_phase6_vision.py`.

---

### 2.2 The Vision Transformer (ViT) Architecture

$$\text{Image } (B, C, H, W) \to \text{Patch Embedding } (B, N, D) \to \text{Prepend } [CLS] \to \text{Add Pos Embed} \to L \times \text{TransformerBlocks} \to \text{Head}$$

1. **Classification Token $[CLS]$**:
   A learnable vector $x_{\text{cls}} \in \mathbb{R}^{1 \times 1 \times D}$ is prepended to the patch sequence:
   $$X_0 = \left[ x_{\text{cls}}; X_{\text{emb}} \right] \in \mathbb{R}^{B \times (N + 1) \times D}$$
   Because attention is bidirectional, the $[CLS]$ token interacts with all spatial patches across all layers, aggregating global visual context.
2. **1D Spatial Position Embeddings**:
   Since self-attention is permutation-equivariant, spatial order is injected via learnable positional embeddings $E_{\text{pos}} \in \mathbb{R}^{1 \times (N + 1) \times D}$:
   $$Z_0 = X_0 + E_{\text{pos}}$$
3. **Bidirectional Self-Attention**:
   Unlike autoregressive LLMs, vision is non-causal: patches attend to all other patches in all directions simultaneously ($\operatorname{causal} = \text{False}$).
4. **Classification Readout**:
   The output corresponding to the $[CLS]$ token at index 0 is passed to LayerNorm and a linear classifier:
   $$\hat{y} = \operatorname{Linear}(\operatorname{LayerNorm}(Z_L[:, 0, :]))$$

In `tests/test_phase6_vision.py`, ViT trained on synthetic visual patterns (horizontal vs vertical stripes) achieves **100% accuracy**.

---

### 2.3 Multimodal Contrastive Learning: CLIP (Radford et al., 2021)

CLIP aligns images and text into a shared joint embedding space $S^{D-1}$ using a two-tower architecture.

#### Mathematical Formulation:
Given a batch of $B$ (image, text) pairs:
1. Compute image embeddings $I = f_{\text{img}}(x_{\text{img}}) \in \mathbb{R}^{B \times D}$.
2. Compute text embeddings $T = f_{\text{txt}}(x_{\text{txt}}) \in \mathbb{R}^{B \times D}$.
3. Project onto the unit hypersphere via $L_2$ normalization:
   $$\hat{I}_i = \frac{I_i}{\|I_i\|_2}, \quad \hat{T}_j = \frac{T_j}{\|T_j\|_2}$$
4. Compute the $B \times B$ pairwise cosine similarity matrix:
   $$S_{ij} = \frac{\hat{I}_i \cdot \hat{T}_j}{\tau}$$
   where $\tau$ is a learned temperature parameter.

#### Symmetric InfoNCE Loss:
Treating the diagonal pairs $(i, i)$ as positive matches and all off-diagonal pairs $(i, j)$ ($j \ne i$) as negatives:
$$\mathcal{L}_{\text{image}} = -\frac{1}{B} \sum_{i=1}^B \ln\left( \frac{e^{S_{ii}}}{\sum_{j=1}^B e^{S_{ij}}} \right)$$
$$\mathcal{L}_{\text{text}} = -\frac{1}{B} \sum_{j=1}^B \ln\left( \frac{e^{S_{jj}}}{\sum_{i=1}^B e^{S_{ij}}} \right)$$
$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2} (\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}})$$

#### Zero-Shot Classification Mechanics:
To classify an image without task-specific fine-tuning:
1. Construct textual prompts for all candidate classes: `"a photo of a [class]"`.
2. Compute normalized text vectors $\hat{T}_c$ for all classes $c \in \{1, \dots, C\}$.
3. Compute normalized image vector $\hat{I}$.
4. Predict class with highest cosine similarity:
   $$\hat{c} = \arg\max_c (\hat{I} \cdot \hat{T}_c)$$

---

## 3. The Multimodal Roadmap: Future Phases

```
[Phase 1-5: Completed Text LLM System]
  ├── Foundations & Optimizers (AdamW)
  ├── Deep Learning Layers (GELU/RMSNorm)
  ├── Multi-Head Attention & RoPE
  ├── Pre-LN TransformerLM
  └── High-Throughput Inference Engine
         │
         ▼
[Phase 6: Spatial & Multimodal Foundations] (Completed)
  ├── Patch Embedding (Continuous Image -> Discrete Vision Tokens)
  ├── Vision Transformer (ViT) with [CLS] Readout
  └── CLIP / InfoNCE Symmetric Contrastive Alignment
         │
         ▼
[Phase 7: Universal Multimodal Dataset & Signal Pipelines]
  ├── Unified Multimodal Tensor Dataset (Batching, Collation, Augmentation)
  ├── 1D Audio & Waveform Tokenization (Mel-Spectrograms, Discrete Codecs)
  └── Tabular & Continuous Signal Embeddings
         │
         ▼
[Phase 8: Vision-Language Model (VLM / LLaVA Architecture)]
  ├── Vision Encoder (ViT)
  ├── Multimodal Projection Connector (Linear / MLP Adapter)
  ├── Autoregressive LLM Decoder (TransformerLM)
  └── Visual Question Answering & Image Captioning Pre-training
```

---

## 4. Verification Test Results

Running `pytest -v` across the entire workspace verifies all 50 passing unit tests:

```text
tests/test_llm_embeddings.py::test_char_tokenizer_encode_decode PASSED   [  2%]
tests/test_llm_embeddings.py::test_embedding_forward_and_backward PASSED [  4%]
tests/test_minigpt.py::test_minigpt_convergence_and_generation PASSED    [  6%]
tests/test_multiclass_nn.py::test_multiclass_mlp_convergence PASSED      [  8%]
tests/test_nn_xor.py::test_xor_non_linear_convergence PASSED             [ 10%]
tests/test_nn_xor.py::test_linear_layer_alone_fails_xor PASSED           [ 12%]
tests/test_phase1.py::test_mse_loss_and_gradient PASSED                  [ 14%]
tests/test_phase1.py::test_numerical_gradient_verification_mse PASSED    [ 16%]
tests/test_phase1.py::test_huber_loss_and_gradient PASSED                [ 18%]
tests/test_phase1.py::test_sigmoid_numerical_stability PASSED            [ 20%]
tests/test_phase1.py::test_binary_cross_entropy_stability PASSED         [ 22%]
tests/test_phase1.py::test_log_sum_exp_stability PASSED                  [ 24%]
tests/test_phase1.py::test_softmax_and_cross_entropy_gradient PASSED     [ 26%]
tests/test_phase1.py::test_gradient_descent_convergence PASSED           [ 28%]
tests/test_phase1.py::test_sgd_momentum_convergence PASSED               [ 30%]
tests/test_phase1.py::test_adam_and_adamw_convergence PASSED             [ 32%]
tests/test_phase1.py::test_gradient_clipping PASSED                      [ 34%]
tests/test_phase2_nn.py::test_linear_layer_2d_and_3d PASSED              [ 36%]
tests/test_phase2_nn.py::test_linear_layer_gradient_check PASSED         [ 38%]
tests/test_phase2_nn.py::test_activations_finite_difference_gradients PASSED [ 40%]
tests/test_phase2_nn.py::test_layernorm_gradient_check PASSED            [ 42%]
tests/test_phase2_nn.py::test_rmsnorm_gradient_check PASSED              [ 44%]
tests/test_phase2_nn.py::test_dropout_train_vs_eval PASSED               [ 46%]
tests/test_phase2_nn.py::test_weight_initializations PASSED              [ 48%]
tests/test_phase2_nn.py::test_deep_mlp_with_gelu_and_layernorm PASSED    [ 50%]
tests/test_phase3_attention.py::test_multi_head_attention_shapes_2d_and_3d PASSED [ 52%]
tests/test_phase3_attention.py::test_multi_head_attention_gradient_check PASSED [ 54%]
tests/test_phase3_attention.py::test_causal_mask_future_invariance PASSED [ 56%]
tests/test_phase3_attention.py::test_rope_relative_position_property PASSED [ 58%]
tests/test_phase3_attention.py::test_mha_with_rope PASSED                [ 60%]
tests/test_phase3_attention.py::test_kv_cache_exact_equivalence PASSED   [ 62%]
tests/test_phase4_transformer.py::test_transformer_block_forward_backward_shapes PASSED [ 64%]
tests/test_phase4_transformer.py::test_residual_identity_gradient_highway PASSED [ 66%]
tests/test_phase4_transformer.py::test_transformer_lm_multi_layer_rope PASSED [ 68%]
tests/test_phase4_transformer.py::test_transformer_lm_kv_cache_generation PASSED [ 70%]
tests/test_phase4_transformer.py::test_transformer_pretraining_convergence PASSED [ 72%]
tests/test_phase5_inference.py::test_greedy_sampling_deterministic PASSED [ 74%]
tests/test_phase5_inference.py::test_temperature_entropy_effect PASSED   [ 76%]
tests/test_phase5_inference.py::test_top_k_filtering PASSED              [ 78%]
tests/test_phase5_inference.py::test_top_p_nucleus_filtering PASSED      [ 80%]
tests/test_phase5_inference.py::test_repetition_penalty PASSED           [ 82%]
tests/test_phase5_inference.py::test_generation_pipeline_streaming_and_stop_words PASSED [ 84%]
tests/test_phase5_inference.py::test_generation_pipeline_full_generation PASSED [ 86%]
tests/test_phase6_vision.py::test_patch_embedding_shapes_and_inversion PASSED [ 88%]
tests/test_phase6_vision.py::test_vision_transformer_forward_backward PASSED [ 90%]
tests/test_phase6_vision.py::test_vision_transformer_image_classification_training PASSED [ 92%]
tests/test_phase6_vision.py::test_l2_normalize_unit_hypersphere PASSED   [ 94%]
tests/test_phase6_vision.py::test_contrastive_loss_clip PASSED           [ 96%]
tests/test_self_attention.py::test_self_attention_causal_mask PASSED     [ 98%]
tests/test_self_attention.py::test_self_attention_backward_pass PASSED   [100%]
============================== 50 passed in 2.08s ==============================
```

To run the interactive Phase 6 demonstration:
```bash
PYTHONPATH=. python3 tests/test_phase6_vision.py
```
```text
============================================================
PHASE 6: VISION TRANSFORMERS & MULTIMODAL SYSTEMS (VERIFICATION)
============================================================

[Step 1] Image Patching Demo:
Input Images: (2, 3, 16, 16) (batch=2, channels=3, 16x16 pixels)
Patch Tokens: (2, 16, 32) (batch=2, 16 patches, 32 embed_dim)

[Step 2] Training Vision Transformer on Visual Patterns:
 Epoch  1 | Cross-Entropy Loss: 1.7299
 Epoch 20 | Cross-Entropy Loss: 0.0182
 Epoch 40 | Cross-Entropy Loss: 0.0094
Training Accuracy: 100.0%

[Step 3] Multimodal CLIP Alignment Demo:
Contrastive Loss: 0.0001
Diagonal Sim (Positive pairs): [0.949 0.926 0.941 0.932]
```
