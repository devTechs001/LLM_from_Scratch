# LLM & Universal Neural System from Scratch: Pure NumPy

A ground-up implementation of machine learning, deep learning primitives, numerical optimization, multi-head attention, decoder-only Transformers, high-throughput inference serving, Vision Transformers (ViT), multimodal contrastive learning (CLIP), universal signal tokenizers (Audio/Continuous), and Vision-Language Models (VLM / LLaVA) using only Python and pure NumPy.

---

## Complete 12-Phase Architecture & Documentation

| Phase | Description | Module | Documentation | Test Suite |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Mathematical Foundations & Optimizers | `ml_lib.losses`, `ml_lib.optimizers` | [`docs/phase_1_foundations.md`](docs/phase_1_foundations.md) | `tests/test_phase1.py` (11 tests) |
| **Phase 2** | Deep Learning Primitives & Non-Linearity | `ml_lib.nn` | [`docs/phase_2_deep_learning_primitives.md`](docs/phase_2_deep_learning_primitives.md) | `tests/test_phase2_nn.py` (8 tests) |
| **Phase 3** | Multi-Head Attention & RoPE | `ml_lib.llm` | [`docs/phase_3_sequence_modeling_attention.md`](docs/phase_3_sequence_modeling_attention.md) | `tests/test_phase3_attention.py` (6 tests) |
| **Phase 4** | Pre-LN Transformer & Language Model | `ml_lib.llm` | [`docs/phase_4_transformer_pretraining.md`](docs/phase_4_transformer_pretraining.md) | `tests/test_phase4_transformer.py` (5 tests) |
| **Phase 5** | Inference Engine, Sampling & Serving | `ml_lib.inference` | [`docs/phase_5_inference_engine_serving.md`](docs/phase_5_inference_engine_serving.md) | `tests/test_phase5_inference.py` (7 tests) |
| **Phase 6** | Vision Transformers & Multimodal Systems | `ml_lib.vision` | [`docs/phase_6_vision_and_multimodal.md`](docs/phase_6_vision_and_multimodal.md) | `tests/test_phase6_vision.py` (5 tests) |
| **Phase 7** | Universal Data Pipelines & Signal Tokenizers | `ml_lib.data` | [`docs/phase_7_and_8_multimodal_vlm.md`](docs/phase_7_and_8_multimodal_vlm.md) | `tests/test_phase7_and_8_multimodal.py` (4 tests) |
| **Phase 8** | Vision-Language Model (VLM / LLaVA) | `ml_lib.multimodal` | [`docs/phase_7_and_8_multimodal_vlm.md`](docs/phase_7_and_8_multimodal_vlm.md) | `tests/test_phase7_and_8_multimodal.py` (3 tests) |
| **Phase 9** | Parameter-Efficient Fine-Tuning (LoRA & SFT) | `ml_lib.peft` | [`docs/phase_9_and_10_peft_and_alignment.md`](docs/phase_9_and_10_peft_and_alignment.md) | `tests/test_phase9_and_10_alignment.py` (4 tests) |
| **Phase 10** | Preference Optimization & Alignment (DPO & RM) | `ml_lib.alignment` | [`docs/phase_9_and_10_peft_and_alignment.md`](docs/phase_9_and_10_peft_and_alignment.md) | `tests/test_phase9_and_10_alignment.py` (3 tests) |
| **Phase 11** | Model Quantization & Compression (INT8 & INT4) | `ml_lib.quantization` | [`docs/phase_11_and_12_moe_and_quantization.md`](docs/phase_11_and_12_moe_and_quantization.md) | `tests/test_phase11_and_12_moe_quant.py` (4 tests) |
| **Phase 12** | Sparse Mixture-of-Experts (MoE) & Top-k Routing | `ml_lib.moe` | [`docs/phase_11_and_12_moe_and_quantization.md`](docs/phase_11_and_12_moe_and_quantization.md) | `tests/test_phase11_and_12_moe_quant.py` (3 tests) |

---

## Repository Structure

```
.
├── docs/
│   ├── phase_1_foundations.md                # Losses (MSE/BCE/Softmax), GD, SGD, Adam, AdamW, & gradient clipping
│   ├── phase_2_deep_learning_primitives.md    # Linear (2D/3D), GELU, SiLU, LayerNorm, RMSNorm, Dropout, XOR Proof
│   ├── phase_3_sequence_modeling_attention.md # Multi-Head Attention, RoPE rotary embeddings, & KV Caching
│   ├── phase_4_transformer_pretraining.md    # Pre-LN TransformerBlock, TransformerLM, & Language Model Pre-training
│   ├── phase_5_inference_engine_serving.md   # Top-p, Top-k, Temperature, Repetition Penalty, & Streaming Engine
│   ├── phase_6_vision_and_multimodal.md      # PatchEmbedding, ViT, [CLS] token, & CLIP InfoNCE Contrastive Loss
│   ├── phase_7_and_8_multimodal_vlm.md       # AudioSpectrogramTokenizer, ContinuousEncoder, VLM / LLaVA prefix fusion
│   ├── phase_9_and_10_peft_and_alignment.md  # LoRA low-rank adaptation, SFT prompt masking, RewardModel, & DPO
│   └── phase_11_and_12_moe_and_quantization.md # INT8/INT4 quantization, QuantizedLinear, SparseMoE & Switch Aux Loss
├── ml_lib/
│   ├── __init__.py                           # Complete package exports
│   ├── losses.py                             # Losses, stability safeguards, analytic & numeric finite-difference gradients
│   ├── optimizers.py                         # GD, SGDMomentum, RMSprop, Adam, AdamW, clip_grad_norm
│   ├── linear_model.py                       # LinearRegression & LogisticRegression
│   ├── nn.py                                 # Linear (2D/3D), ReLU, GELU, SiLU, LayerNorm, RMSNorm, Dropout
│   ├── llm.py                                # CharTokenizer, Embedding, MultiHeadAttention, RoPE, TransformerLM, MiniGPT
│   ├── inference.py                          # sample_next_token, GenerationPipeline (streaming & caching)
│   ├── vision.py                             # PatchEmbedding, VisionTransformer, l2_normalize, ContrastiveLoss
│   ├── data.py                               # Dataset, DataLoader, AudioSpectrogramTokenizer, ContinuousFeatureEncoder, Augmentations
│   ├── multimodal.py                         # VisionLanguageModel (LLaVA architecture with visual prefix & text decoding)
│   ├── peft.py                               # LoRALinear, weight merge/unmerge, SFT masked cross-entropy
│   ├── alignment.py                          # RewardModel, Bradley-Terry pairwise loss, DPO loss & policy gradients
│   ├── quantization.py                       # INT8 & INT4 symmetric/asymmetric quantization & QuantizedLinear
│   └── moe.py                                # ExpertMLP, SparseMoE with top-k gating & Switch aux loss
├── tests/
│   ├── test_phase1.py                        # 11 unit tests for losses, numerical stability & optimizers
│   ├── test_phase2_nn.py                     # 8 unit tests for deep learning layers, activations, & LayerNorm/RMSNorm
│   ├── test_phase3_attention.py              # 6 unit tests for Multi-Head Attention, RoPE, & KV cache identity
│   ├── test_phase4_transformer.py            # 5 unit tests for Pre-LN Transformer & language model training
│   ├── test_phase5_inference.py              # 7 unit tests for token samplers, repetition penalty, & streaming
│   ├── test_phase6_vision.py                 # 5 unit tests for Vision Transformer & CLIP contrastive loss
│   ├── test_phase7_and_8_multimodal.py       # 7 unit tests for Audio/Continuous tokenizers & VLM forward/backward/generate
│   ├── test_phase9_and_10_alignment.py       # 7 unit tests for LoRA, weight merging, SFT masking, RewardModel & DPO
│   ├── test_phase11_and_12_moe_quant.py      # 7 unit tests for INT8/INT4 quantization & SparseMoE routing/aux loss
│   ├── test_minigpt.py                       # MiniGPT end-to-end training & generation
│   ├── test_nn_xor.py                        # XOR non-linear separation & single-layer failure proof
│   ├── test_multiclass_nn.py                 # Multi-class MLP cluster convergence
│   ├── test_linear_regression.py             # Verifies linear regression convergence
│   ├── test_logistic_regression.py           # Verifies binary classification
│   ├── test_llm_embeddings.py                # Verifies token & position embeddings
│   └── test_self_attention.py                # Verifies causal attention & gradient flow
├── requirements.txt
└── README.md
```

---

## Quickstart & Verification

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run Full Test Suite (71 Unit Tests):
```bash
PYTHONPATH=. pytest -v
```

### Run Phase-by-Phase Interactive Demos:
```bash
# Phase 1: Mathematical Foundations & Optimizers
PYTHONPATH=. python3 tests/test_phase1.py

# Phase 2: Deep Learning Layers & Normalization
PYTHONPATH=. python3 tests/test_phase2_nn.py

# Phase 3: Multi-Head Attention, RoPE & KV Cache
PYTHONPATH=. python3 tests/test_phase3_attention.py

# Phase 4: Full Multi-layer TransformerLM Pre-training
PYTHONPATH=. python3 tests/test_phase4_transformer.py

# Phase 5: Inference Engine, Sampling & Streaming Pipeline
PYTHONPATH=. python3 tests/test_phase5_inference.py

# Phase 6: Vision Transformer & Multimodal Contrastive Alignment
PYTHONPATH=. python3 tests/test_phase6_vision.py

# Phases 7 & 8: Multimodal Signal Tokenizers & Vision-Language Model
PYTHONPATH=. python3 tests/test_phase7_and_8_multimodal.py

# Phases 9 & 10: Parameter-Efficient Fine-Tuning (LoRA) & Alignment (DPO)
PYTHONPATH=. python3 tests/test_phase9_and_10_alignment.py

# Phases 11 & 12: Model Quantization (INT8/INT4) & Sparse Mixture-of-Experts (MoE)
PYTHONPATH=. python3 tests/test_phase11_and_12_moe_quant.py
```

### Train Baseline MiniGPT and Generate Text:
```bash
PYTHONPATH=. python3 tests/test_minigpt.py
```
