import numpy as np
from typing import Optional, List, Tuple
from ml_lib.vision import PatchEmbedding
from ml_lib.llm import TransformerLM, CharTokenizer
from ml_lib.nn import Linear, GELU
from ml_lib.losses import softmax

# =====================================================================
# Vision-Language Model (VLM / LLaVA Architecture - Liu et al., 2023)
# =====================================================================

class VisionLanguageModel:
    """
    Multimodal Vision-Language Model (VLM):
      1. Vision Encoder: PatchEmbedding extracts visual patch tokens: (B, N, D_vis)
      2. Multimodal Adapter Projector: Linear/MLP projects visual tokens into
         the language model's hidden representation space: (B, N, D_text)
      3. Autoregressive Language Model: TransformerLM
    The visual tokens act as soft prefix prompts. Under causal self-attention,
    the model reads the image prefix and autoregressively generates text explanations,
    image captions, or answers visual queries.
    """
    def __init__(
        self,
        vocab_size: int,
        image_size: int = 16,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
        max_seq_len: int = 128,
        use_mlp_projector: bool = True
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_mlp_projector = use_mlp_projector

        # 1. Vision Patch Encoder
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        # 2. Multimodal Projection Adapter (maps vision space to text space)
        if use_mlp_projector:
            self.proj_w1 = Linear(embed_dim, embed_dim * 2)
            self.proj_act = GELU()
            self.proj_w2 = Linear(embed_dim * 2, embed_dim)
        else:
            self.proj_w1 = Linear(embed_dim, embed_dim)
            self.proj_act = None
            self.proj_w2 = None

        # 3. Autoregressive Language Model Decoder
        self.lm = TransformerLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            use_rope=False
        )
        self.cache = {}

    def _project_visual_tokens(self, v_tokens: np.ndarray) -> np.ndarray:
        if self.use_mlp_projector:
            mid = self.proj_act.forward(self.proj_w1.forward(v_tokens))
            return self.proj_w2.forward(mid)
        return self.proj_w1.forward(v_tokens)

    def _backward_projector(self, d_vis: np.ndarray, lr: Optional[float] = None) -> np.ndarray:
        if self.use_mlp_projector:
            d_mid = self.proj_w2.backward(d_vis, lr)
            d_act = self.proj_act.backward(d_mid)
            return self.proj_w1.backward(d_act, lr)
        return self.proj_w1.backward(d_vis, lr)

    def forward(
        self,
        images: np.ndarray,
        text_ids: np.ndarray
    ) -> np.ndarray:
        """
        images: (B, C, H, W)
        text_ids: (B, T) token IDs
        returns: logits of shape (B, N_patches + T, vocab_size)
        """
        B = images.shape[0]

        # 1. Vision tokens: (B, N, D)
        v_tokens = self.patch_embed.forward(images)
        v_proj = self._project_visual_tokens(v_tokens)

        # 2. Text token embeddings: (B, T, D)
        t_emb = self.lm.tok_emb.forward(text_ids)

        # 3. Concatenate visual tokens and text tokens into unified sequence
        combined = np.concatenate([v_proj, t_emb], axis=1)  # (B, N + T, D)
        total_len = combined.shape[1]

        # 4. Add positional embeddings
        positions = np.arange(total_len)
        x = combined + self.lm.pos_emb.forward(positions)

        # 5. Pass through Transformer decoder blocks
        for block in self.lm.blocks:
            x = block.forward(x)

        # 6. Final normalization & LM Head
        x_norm = self.lm.ln_f.forward(x)
        logits = self.lm.lm_head.forward(x_norm)

        self.cache = {
            "images": images,
            "text_ids": text_ids,
            "N": v_tokens.shape[1],
            "T": text_ids.shape[1],
            "x_norm": x_norm,
            "B": B
        }
        return logits

    def backward(self, d_logits: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        End-to-end backpropagation:
        Loss -> LM Head -> Final Norm -> Decoder Blocks -> [Split Vision & Text]
             -> Multimodal Projector -> Vision Patch Encoder -> Pixel Gradients.
        """
        N = self.cache["N"]
        T = self.cache["T"]

        # 1. LM Head & Final Norm backward
        d_x_norm = self.lm.lm_head.backward(d_logits, learning_rate)
        d_x = self.lm.ln_f.backward(d_x_norm, learning_rate)

        # 2. Reverse through Decoder Blocks
        for block in reversed(self.lm.blocks):
            d_x = block.backward(d_x, learning_rate)

        # 3. Positional embedding backward
        d_pos = np.mean(d_x, axis=0) if d_x.ndim == 3 else d_x
        self.lm.pos_emb.backward(d_pos, learning_rate)

        # 4. Split gradients: visual tokens vs text tokens
        d_vis = d_x[:, :N, :]
        d_txt = d_x[:, N:, :]

        # 5. Text embedding backward
        self.lm.tok_emb.backward(d_txt, learning_rate)

        # 6. Multimodal projector backward
        d_proj = self._backward_projector(d_vis, learning_rate)

        # 7. Vision patch encoder backward to raw pixels
        d_images = self.patch_embed.backward(d_proj, learning_rate)
        return d_images

    def generate(
        self,
        images: np.ndarray,
        prompt_ids: List[int],
        max_new_tokens: int = 20,
        temperature: float = 0.7
    ) -> List[int]:
        """
        Autoregressively generates text conditioned on the input image.
        """
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)

        # Project image into visual prefix tokens: (1, N, D)
        v_tokens = self.patch_embed.forward(images)
        v_proj = self._project_visual_tokens(v_tokens)

        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            t_ids = np.array([generated], dtype=int)
            t_emb = self.lm.tok_emb.forward(t_ids)
            combined = np.concatenate([v_proj, t_emb], axis=1)
            total_len = combined.shape[1]

            pos = np.arange(total_len)
            x = combined + self.lm.pos_emb.forward(pos)
            for block in self.lm.blocks:
                x = block.forward(x)

            logits = self.lm.lm_head.forward(self.lm.ln_f.forward(x))
            last_logits = logits[0, -1]

            if temperature <= 1e-5:
                next_tok = int(np.argmax(last_logits))
            else:
                probs = softmax(last_logits / temperature)
                next_tok = int(np.random.choice(len(probs), p=probs))

            generated.append(next_tok)

        return generated
