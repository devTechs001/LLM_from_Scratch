# Phase 7 & 8: Universal Data Pipelines & Vision-Language Model (VLM / LLaVA)

## Executive Summary

Phases 7 and 8 expand our pure NumPy deep learning library from unimodal text and vision representations into a universal multimodal foundation model. We implement:
1. **Universal Data & Signal Tokenizers (Phase 7)**:
   - Flexible batching primitives (`Dataset`, `TensorDataset`, `DataLoader`).
   - Signal tokenizers for continuous 1D audio waveforms (`AudioSpectrogramTokenizer`) and continuous tabular/sensor signals (`ContinuousFeatureEncoder`).
   - Image augmentation operators (`random_horizontal_flip`, `random_erasing`).
2. **Vision-Language Model (Phase 8 - LLaVA Architecture)**:
   - A unified architecture combining spatial patch visual encoding, cross-modal MLP projection, and autoregressive causal language modeling.
   - Multimodal prefix attention: visual tokens act as prefix prompt contexts for the language decoder.
   - End-to-end backpropagation flowing from cross-entropy token predictions through the decoder, projector, and patch projection back to the raw input pixels.
   - Autoregressive image-conditioned generation for visual question answering and image captioning.

All primitives are written in pure NumPy without third-party deep learning frameworks.

---

## Architecture Diagram

```
[Raw Image (B, C, H, W)]                   [Text Tokens (B, T)]
           │                                         │
    PatchEmbedding (ViT)                    Embedding (Char/BPE)
           │                                         │
 [Visual Patches (B, N, D_v)]                [Word Embeddings (B, T, D_t)]
           │                                         │
  MLP Adapter Projector                              │
           │                                         │
 [Projected Visual Tokens (B, N, D)]                 │
           └───────────────────┬─────────────────────┘
                               │ Concatenate along Sequence Dim
                     [Z_multimodal (B, N + T, D)]
                               │
               TransformerLM (Pre-LN Causal Blocks)
                               │
                      [Logits (B, N + T, Vocab)]
                               │
                 Autoregressive Next-Token Loss
```

---

## 1. Phase 7: Universal Data & Signal Tokenizers

### 1.1 Dataset and DataLoader (`ml_lib/data.py`)
To feed multimodal models, we built PyTorch-style batching abstractions:
- **`Dataset`**: Abstract base class defining `__len__()` and `__getitem__(index)`.
- **`TensorDataset`**: Encapsulates multiple NumPy arrays indexed synchronously along dimension 0.
- **`DataLoader`**: Provides minibatches, optional shuffling via index permutation (`np.random.permutation`), and drop-last or remainder handling.

### 1.2 Audio Spectrogram Tokenizer
Audio waveforms $x \in \mathbb{R}^{B \times L}$ are continuous 1D time-series signals. To tokenize them for attention architectures:
1. **Window Framing (STFT analogue)**: The signal of length $L$ is partitioned into overlapping windows of length $F$ (`frame_size`) with stride $S$ (`hop_size`):
   $$N_{\text{frames}} = \left\lfloor \frac{L - F}{S} \right\rfloor + 1$$
   $$\mathbf{f}_t = x[:, t \cdot S : t \cdot S + F] \in \mathbb{R}^{B \times F}$$
2. **Linear Spectral Token Projection**:
   $$Z_{\text{audio}} = \mathbf{f}_t W_{\text{proj}} + \mathbf{b} \in \mathbb{R}^{B \times N_{\text{frames}} \times D}$$

### 1.3 Continuous Feature Encoder
For sensor data, tabular metrics, or continuous embeddings:
1. **Z-Score Normalization**:
   $$\hat{x} = \frac{x - \mu}{\sigma + \epsilon}$$
2. **Dense Feature Projection**:
   $$Z_{\text{feature}} = \hat{x} W_{\text{feat}} + \mathbf{b} \in \mathbb{R}^{B \times 1 \times D}$$

### 1.4 NumPy Data Augmentations
- `random_horizontal_flip(images, p=0.5)`: Vectorized spatial axis reversal (`images[:, :, :, ::-1]`).
- `random_erasing(images, p=0.5, erase_size=4)`: Random spatial cutout occlusion simulating object truncation.

---

## 2. Phase 8: Vision-Language Model (VLM / LLaVA)

### 2.1 Cross-Modal Token Prefix Fusion
In contemporary multimodal architectures (e.g., LLaVA, Flamingo, BLIP-2), visual representations are projected directly into the input embedding space of an autoregressive language decoder.

Given:
- Image: $X_{\text{img}} \in \mathbb{R}^{B \times C \times H \times W}$
- Text tokens: $X_{\text{text}} \in \mathbb{R}^{B \times T}$

The forward sequence is constructed as follows:
1. **Patch Extraction**:
   $$Z_{\text{vis}} = \operatorname{PatchEmbedding}(X_{\text{img}}) \in \mathbb{R}^{B \times N \times D_{\text{vis}}}, \quad N = \frac{H \cdot W}{P^2}$$
2. **Cross-Modal Adapter Projection**:
   An adapter projects visual tokens to the language embedding space $D$:
   $$Z_{\text{proj}} = \operatorname{GELU}(Z_{\text{vis}} W_1 + b_1) W_2 + b_2 \in \mathbb{R}^{B \times N \times D}$$
3. **Text Token Embedding**:
   $$Z_{\text{txt}} = \operatorname{Embedding}(X_{\text{text}}) \in \mathbb{R}^{B \times T \times D}$$
4. **Sequence Prefix Concatenation**:
   $$Z = \begin{bmatrix} Z_{\text{proj}} & Z_{\text{txt}} \end{bmatrix} \in \mathbb{R}^{B \times (N + T) \times D}$$

### 2.2 Causal Prefix Attention
Under standard autoregressive causal attention:
- Visual patch tokens attend to all preceding visual tokens (and self).
- Text tokens at positions $t \ge N$ attend to **all visual tokens** $1 \dots N$ as well as preceding text tokens $N \dots t$.
- Future text tokens are strictly masked out.

This allows the language decoder to naturally condition its next-token probabilities on the full visual context:
$$P(w_{t+1} \mid X_{\text{img}}, w_1, \dots, w_t)$$

### 2.3 End-to-End Multimodal Backpropagation
Gradients flow seamlessly from the token-prediction cross-entropy loss through the entire pipeline:
1. $\nabla_{Z} \mathcal{L} \in \mathbb{R}^{B \times (N+T) \times D}$ computed by `TransformerLM.backward()`.
2. Split along the sequence axis:
   $$\nabla_{Z_{\text{proj}}} \mathcal{L} = \nabla_{Z} \mathcal{L}[:, :N, :], \quad \nabla_{Z_{\text{txt}}} \mathcal{L} = \nabla_{Z} \mathcal{L}[:, N:, :]$$
3. Backpropagation through the adapter projector:
   $$\nabla_{Z_{\text{vis}}} \mathcal{L} = \operatorname{Projector.backward}(\nabla_{Z_{\text{proj}}} \mathcal{L})$$
4. Unpatchify pixel reconstruction via `PatchEmbedding.backward()`:
   $$\nabla_{X_{\text{img}}} \mathcal{L} \in \mathbb{R}^{B \times C \times H \times W}$$

### 2.4 Image-Conditioned Generation
The `VisionLanguageModel.generate` method implements autoregressive inference:
1. Projects the input image into $N$ visual prefix tokens.
2. Embeds the user prompt tokens.
3. In each generation step:
   - Evaluates the joint sequence $[Z_{\text{proj}}, Z_{\text{txt}}]$.
   - Extracts the logits at the final position.
   - Samples the next token using greedy, temperature, or top-$k$/top-$p$ sampling.
   - Appends the generated token ID to $Z_{\text{txt}}$ until `max_new_tokens` is reached.

---

## 3. Empirical Verification & Test Suite

The test suite in `tests/test_phase7_and_8_multimodal.py` verifies all multimodal capabilities:

| Test Case | Description | Result |
|---|---|---|
| `test_dataloader_shuffling_and_collation` | Batch collation and epoch shuffling | **PASSED** |
| `test_audio_spectrogram_tokenizer` | 1D waveform windowing into spectral tokens | **PASSED** |
| `test_continuous_feature_encoder` | Tabular z-score normalization and projection | **PASSED** |
| `test_image_augmentations` | Vectorized horizontal flips and spatial erasing | **PASSED** |
| `test_vision_language_model_forward_backward` | End-to-end forward pass and pixel gradient propagation | **PASSED** |
| `test_vision_language_model_generation` | Autoregressive visual-caption generation | **PASSED** |
| `test_vlm_multimodal_training_convergence` | Joint training on image-caption pairs reduces loss > 70% | **PASSED** |

### Complete Repository Verification
Across all 8 phases, **57 out of 57** unit tests pass cleanly:
```bash
$ PYTHONPATH=. pytest -v
============================== 57 passed in 2.05s ==============================
```

---

## 4. Code Example

```python
import numpy as np
from ml_lib.multimodal import VisionLanguageModel
from ml_lib.llm import CharTokenizer

# 1. Initialize tokenizer and VLM
corpus = "Describe the image: A bright horizontal pattern."
tokenizer = CharTokenizer(corpus)

vlm = VisionLanguageModel(
    vocab_size=tokenizer.vocab_size,
    image_size=16,
    patch_size=4,
    in_channels=1,
    embed_dim=32,
    num_layers=2,
    num_heads=4,
    use_mlp_projector=True
)

# 2. Input image and prompt
image = np.random.randn(1, 16, 16)
prompt_ids = tokenizer.encode("Describe:")

# 3. Autoregressive multimodal generation
output_ids = vlm.generate(
    image=image,
    prompt_ids=prompt_ids,
    max_new_tokens=20,
    temperature=0.7
)
print("Generated text:", tokenizer.decode(output_ids))
```
