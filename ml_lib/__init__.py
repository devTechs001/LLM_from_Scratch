"""ml_lib: A pure NumPy deep learning, LLM & Multimodal AI framework built from scratch."""

from ml_lib.losses import (
    mse_loss,
    mse_gradient,
    huber_loss,
    huber_gradient,
    sigmoid,
    binary_cross_entropy,
    bce_gradient,
    log_sum_exp,
    softmax,
    cross_entropy_loss,
    cross_entropy_softmax_gradient,
    compute_numerical_gradient,
    relative_error,
)
from ml_lib.optimizers import (
    Optimizer,
    GradientDescent,
    SGDMomentum,
    RMSprop,
    Adam,
    AdamW,
    clip_grad_norm,
    clip_grad_value,
)
from ml_lib.linear_model import LinearRegression, LogisticRegression
from ml_lib.nn import (
    Module,
    Linear,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
    LayerNorm,
    RMSNorm,
    Dropout,
    Sequential,
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
)
from ml_lib.llm import (
    CharTokenizer,
    Embedding,
    precompute_freqs_cis,
    apply_rotary_emb,
    SelfAttention,
    MultiHeadAttention,
    TransformerBlock,
    TransformerLM,
    MiniGPT,
)
from ml_lib.inference import (
    sample_next_token,
    GenerationPipeline,
)
from ml_lib.vision import (
    PatchEmbedding,
    VisionTransformer,
    l2_normalize,
    ContrastiveLoss,
)
from ml_lib.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    AudioSpectrogramTokenizer,
    ContinuousFeatureEncoder,
    random_horizontal_flip,
    random_erasing,
)
from ml_lib.multimodal import (
    VisionLanguageModel,
)
from ml_lib.peft import (
    LoRALinear,
    apply_lora_to_linear,
    sft_masked_cross_entropy,
)
from ml_lib.alignment import (
    RewardModel,
    pairwise_reward_loss,
    compute_sequence_log_probs,
    dpo_loss_and_gradients,
)
from ml_lib.quantization import (
    quantize_symmetric_int8,
    dequantize_symmetric_int8,
    quantize_asymmetric_int8,
    dequantize_asymmetric_int8,
    quantize_symmetric_int4,
    dequantize_symmetric_int4,
    QuantizedLinear,
)
from ml_lib.moe import (
    ExpertMLP,
    SparseMoE,
)

__all__ = [
    # Losses & Gradients
    "mse_loss",
    "mse_gradient",
    "huber_loss",
    "huber_gradient",
    "sigmoid",
    "binary_cross_entropy",
    "bce_gradient",
    "log_sum_exp",
    "softmax",
    "cross_entropy_loss",
    "cross_entropy_softmax_gradient",
    "compute_numerical_gradient",
    "relative_error",
    # Optimizers
    "Optimizer",
    "GradientDescent",
    "SGDMomentum",
    "RMSprop",
    "Adam",
    "AdamW",
    "clip_grad_norm",
    "clip_grad_value",
    # Models & Layers
    "LinearRegression",
    "LogisticRegression",
    "Module",
    "Linear",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "GELU",
    "SiLU",
    "LayerNorm",
    "RMSNorm",
    "Dropout",
    "Sequential",
    # Initializations
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    # LLM, Attention & RoPE
    "CharTokenizer",
    "Embedding",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "SelfAttention",
    "MultiHeadAttention",
    "TransformerBlock",
    "TransformerLM",
    "MiniGPT",
    # Inference & Serving
    "sample_next_token",
    "GenerationPipeline",
    # Vision & Contrastive Alignment
    "PatchEmbedding",
    "VisionTransformer",
    "l2_normalize",
    "ContrastiveLoss",
    # Universal Data & Signal Tokenizers
    "Dataset",
    "TensorDataset",
    "DataLoader",
    "AudioSpectrogramTokenizer",
    "ContinuousFeatureEncoder",
    "random_horizontal_flip",
    "random_erasing",
    # Multimodal Vision-Language Model
    "VisionLanguageModel",
    # Parameter-Efficient Fine-Tuning & SFT
    "LoRALinear",
    "apply_lora_to_linear",
    "sft_masked_cross_entropy",
    # Alignment & Preference Optimization
    "RewardModel",
    "pairwise_reward_loss",
    "compute_sequence_log_probs",
    "dpo_loss_and_gradients",
    # Quantization & Compression
    "quantize_symmetric_int8",
    "dequantize_symmetric_int8",
    "quantize_asymmetric_int8",
    "dequantize_asymmetric_int8",
    "quantize_symmetric_int4",
    "dequantize_symmetric_int4",
    "QuantizedLinear",
    # Sparse Mixture-of-Experts
    "ExpertMLP",
    "SparseMoE",
]
