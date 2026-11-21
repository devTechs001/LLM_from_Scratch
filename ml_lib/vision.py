import numpy as np
from typing import Tuple, Optional, Any
from ml_lib.nn import Linear, LayerNorm, RMSNorm
from ml_lib.llm import TransformerBlock
from ml_lib.losses import softmax, cross_entropy_loss, cross_entropy_softmax_gradient

# =====================================================================
# Patch Embedding (ViT Vision Tokenizer)
# =====================================================================

class PatchEmbedding:
    """
    Splits images into non-overlapping patches and projects each patch
    into a dense embedding vector:
      (B, C, H, W) -> (B, N, C * P * P) -> (B, N, D)
    where N = (H / P) * (W / P) is the sequence length of vision tokens.
    """
    def __init__(
        self,
        image_size: int = 16,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 32
    ):
        assert image_size % patch_size == 0, f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = in_channels * patch_size * patch_size

        self.proj = Linear(self.patch_dim, embed_dim)
        self.cache = {}

    def forward(self, images: np.ndarray) -> np.ndarray:
        """
        images: shape (B, C, H, W)
        returns patches: shape (B, num_patches, embed_dim)
        """
        B, C, H, W = images.shape
        P = self.patch_size
        num_h, num_w = self.num_patches_h, self.num_patches_w

        # 1. Reshape to grid of patches
        x = images.reshape(B, C, num_h, P, num_w, P)
        # 2. Transpose to (B, num_h, num_w, C, P, P)
        x = x.transpose(0, 2, 4, 1, 3, 5)
        # 3. Flatten each patch into a vector: (B, N, C * P * P)
        patches_flat = x.reshape(B, self.num_patches, self.patch_dim)

        # 4. Linear projection into embedding space: (B, N, D)
        embeddings = self.proj.forward(patches_flat)

        self.cache = {"images_shape": images.shape, "patches_flat": patches_flat}
        return embeddings

    def backward(self, d_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        """
        Backpropagates gradients through linear projection and reconstructs
        gradient with respect to input images: (B, N, D) -> (B, C, H, W).
        """
        B, C, H, W = self.cache["images_shape"]
        P = self.patch_size
        num_h, num_w = self.num_patches_h, self.num_patches_w

        # 1. Backprop through linear projection
        d_patches_flat = self.proj.backward(d_output, learning_rate)

        # 2. Invert patch transpose & reshape: (B, num_h, num_w, C, P, P) -> (B, C, H, W)
        d_grid = d_patches_flat.reshape(B, num_h, num_w, C, P, P)
        d_images = d_grid.transpose(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return d_images


# =====================================================================
# Vision Transformer (ViT - Dosovitskiy et al., 2020)
# =====================================================================

class VisionTransformer:
    """
    Vision Transformer (ViT) implemented from scratch:
      - Patch Embedding tokenization
      - Learnable [CLS] classification token prepended to patch sequence
      - Learnable 1D spatial position embeddings
      - Stack of bidirectional Pre-LN Transformer blocks (causal=False)
      - Final LayerNorm + Linear classification head
    """
    def __init__(
        self,
        image_size: int = 16,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 32,
        depth: int = 2,
        num_heads: int = 2,
        norm_type: str = "layernorm",
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # 1. Patch extraction & projection
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 2. Learnable [CLS] classification token & 1D Positional Embeddings
        self.cls_token = np.random.randn(1, 1, embed_dim) * 0.02
        self.pos_embed = np.random.randn(1, num_patches + 1, embed_dim) * 0.02

        # 3. Bidirectional Transformer Blocks (causal=False)
        self.blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                use_rope=False,
                causal=False,
                norm_type=norm_type
            )
            for _ in range(depth)
        ]

        # 4. Final normalization & Classification Head
        norm_cls = LayerNorm if norm_type == "layernorm" else RMSNorm
        self.norm = norm_cls(embed_dim)
        self.head = Linear(embed_dim, num_classes)
        self.cache = {}

    def forward(self, images: np.ndarray) -> np.ndarray:
        """
        images: shape (B, C, H, W)
        returns: logits of shape (B, num_classes)
        """
        B = images.shape[0]

        # 1. Project patches: (B, N, D)
        patches = self.patch_embed.forward(images)

        # 2. Prepend [CLS] token: (B, N+1, D)
        cls_tokens = np.repeat(self.cls_token, B, axis=0)
        x_seq = np.concatenate([cls_tokens, patches], axis=1)

        # 3. Add learned spatial position embeddings
        x_seq = x_seq + self.pos_embed

        # 4. Pass through bidirectional Transformer blocks
        for block in self.blocks:
            x_seq = block.forward(x_seq)

        # 5. Extract [CLS] token at position 0
        cls_out = x_seq[:, 0:1, :]
        cls_norm = self.norm.forward(cls_out)[:, 0, :]

        # 6. Linear classification head
        logits = self.head.forward(cls_norm)

        self.cache = {
            "images": images,
            "x_seq": x_seq,
            "cls_norm": cls_norm,
            "B": B
        }
        return logits

    def backward(self, d_logits: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backpropagates through classification head, final norm, transformer blocks,
        [CLS] token, positional embeddings, and patch projection to raw pixel gradients.
        """
        B = self.cache["B"]

        # 1. Head backward
        d_cls_norm = self.head.backward(d_logits, learning_rate)
        d_cls_norm_3d = np.expand_dims(d_cls_norm, axis=1)

        # 2. Final norm backward
        d_cls_out = self.norm.backward(d_cls_norm_3d, learning_rate)

        # 3. Scatter [CLS] gradient into full sequence gradient (B, N+1, D)
        d_seq = np.zeros_like(self.cache["x_seq"])
        d_seq[:, 0:1, :] = d_cls_out

        # 4. Backward through bidirectional blocks
        for block in reversed(self.blocks):
            d_seq = block.backward(d_seq, learning_rate)

        # 5. Positional embedding gradient
        d_pos = np.sum(d_seq, axis=0, keepdims=True)
        self.pos_embed -= learning_rate * d_pos

        # 6. [CLS] token gradient
        d_cls = np.sum(d_seq[:, 0:1, :], axis=0, keepdims=True)
        self.cls_token -= learning_rate * d_cls

        # 7. Patch embedding backward to pixel space
        d_patches = d_seq[:, 1:, :]
        d_images = self.patch_embed.backward(d_patches, learning_rate)
        return d_images


# =====================================================================
# Multimodal Contrastive Alignment (CLIP / InfoNCE)
# =====================================================================

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Projects vectors onto the unit hypersphere: x / ||x||_2."""
    norm = np.sqrt(np.sum(x ** 2, axis=axis, keepdims=True) + eps)
    return x / norm


class ContrastiveLoss:
    """
    Symmetric InfoNCE / CLIP Contrastive Loss (Radford et al., 2021):
      S = tau * (I_emb @ T_emb.T)
      Loss_I = CrossEntropy(S, targets)
      Loss_T = CrossEntropy(S.T, targets)
      Loss = (Loss_I + Loss_T) / 2
    Maximizes cosine similarity of positive pairs while repelling negative pairs.
    """
    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def forward(
        self,
        image_embeds: np.ndarray,
        text_embeds: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        image_embeds: (B, D) L2-normalized image representations
        text_embeds:  (B, D) L2-normalized text representations
        returns: (scalar loss, similarity matrix S)
        """
        B = image_embeds.shape[0]
        # Cosine similarity matrix scaled by inverse temperature
        similarity = np.dot(image_embeds, text_embeds.T) / self.temperature

        # Ground-truth diagonal targets: sample i matches sample i
        targets = np.arange(B)

        probs_i2t = softmax(similarity)
        loss_i2t = cross_entropy_loss(targets, probs_i2t)

        probs_t2i = softmax(similarity.T)
        loss_t2i = cross_entropy_loss(targets, probs_t2i)

        loss = float(0.5 * (loss_i2t + loss_t2i))
        return loss, similarity
