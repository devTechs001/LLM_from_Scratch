import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from ml_lib.nn import Linear, ReLU, GELU, Sequential, LayerNorm, RMSNorm
from ml_lib.losses import softmax, cross_entropy_loss, cross_entropy_softmax_gradient

# =====================================================================
# Tokenizer
# =====================================================================

class CharTokenizer:
    """Converts raw text into integer token IDs and decodes IDs back to text."""
    def __init__(self, text: str):
        # Extract sorted unique characters as our vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> np.ndarray:
        """Converts a string into a 1D NumPy array of token IDs."""
        return np.array([self.char_to_id[ch] for ch in text], dtype=int)

    def decode(self, token_ids: np.ndarray) -> str:
        """Converts a 1D array/list of token IDs back into a string."""
        return "".join([self.id_to_char[int(i)] for i in token_ids])


# =====================================================================
# Embedding Layer
# =====================================================================

class Embedding:
    """Lookup table that maps integer indices to dense vectors."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Small random initialization for embeddings
        self.weights = np.random.randn(num_embeddings, embedding_dim) * 0.02
        self.last_indices = None

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        indices: shape (seq_len,) or (batch_size, seq_len)
        returns: shape (..., embedding_dim)
        """
        self.last_indices = indices
        return self.weights[indices]

    def backward(self, d_output: np.ndarray, learning_rate: float) -> None:
        """Accumulates gradients only for the specific embedding rows used in forward()."""
        d_weights = np.zeros_like(self.weights)
        # np.add.at safely adds gradients even if the same token appears multiple times
        np.add.at(d_weights, self.last_indices, d_output)
        self.weights -= learning_rate * d_weights


# =====================================================================
# Rotary Position Embedding (RoPE - Su et al., 2021)
# =====================================================================

def precompute_freqs_cis(dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precomputes sinusoidal frequency matrices for Rotary Position Embedding:
    theta_i = 1 / (theta ** (2 * (i - 1) / dim))
    Returns (cos, sin) arrays of shape (max_seq_len, dim // 2).
    """
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = np.arange(max_seq_len)
    angles = np.outer(t, freqs)
    return np.cos(angles), np.sin(angles)


def apply_rotary_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Applies Rotary Position Embedding (RoPE) to queries or keys:
    Rotates consecutive pairs: [x0, x1] -> [x0 * cos - x1 * sin, x0 * sin + x1 * cos]
    x shape: (..., seq_len, head_dim)
    cos, sin shape: (seq_len, head_dim // 2)
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    
    out = np.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


# =====================================================================
# Single-Head Self-Attention (Baseline)
# =====================================================================

class SelfAttention:
    """Single-head Causal Self-Attention mechanism (GPT-style baseline)."""
    def __init__(self, embed_dim: int, head_dim: int):
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        scale = np.sqrt(2.0 / embed_dim)
        
        # Projection weights for Query, Key, Value, and Output
        self.W_q = np.random.randn(embed_dim, head_dim) * scale
        self.W_k = np.random.randn(embed_dim, head_dim) * scale
        self.W_v = np.random.randn(embed_dim, head_dim) * scale
        self.W_o = np.random.randn(head_dim, embed_dim) * scale
        
        self.cache = {}

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: shape (seq_len, embed_dim)
        returns: shape (seq_len, embed_dim)
        """
        seq_len, _ = x.shape
        
        # 1. Project input into Q, K, V
        Q = np.dot(x, self.W_q)  # (seq_len, head_dim)
        K = np.dot(x, self.W_k)  # (seq_len, head_dim)
        V = np.dot(x, self.W_v)  # (seq_len, head_dim)
        
        # 2. Compute scaled dot-product attention scores
        scores = np.dot(Q, K.T) / np.sqrt(self.head_dim)  # (seq_len, seq_len)
        
        # 3. Apply causal mask so tokens only attend to past/current tokens
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        masked_scores = scores + mask
        
        # 4. Convert scores to attention probabilities
        attn_weights = self._softmax(masked_scores)  # (seq_len, seq_len)
        
        # 5. Weighted sum of Values + output projection
        context = np.dot(attn_weights, V)            # (seq_len, head_dim)
        out = np.dot(context, self.W_o)              # (seq_len, embed_dim)
        
        self.cache = {
            "x": x, "Q": Q, "K": K, "V": V,
            "attn_weights": attn_weights, "context": context
        }
        return out

    def backward(self, d_out: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backpropagates gradients through output projection, softmax, and Q/K/V."""
        x = self.cache["x"]
        Q, K, V = self.cache["Q"], self.cache["K"], self.cache["V"]
        attn = self.cache["attn_weights"]
        context = self.cache["context"]
        
        # 1. Backprop through output projection (out = context @ W_o)
        d_W_o = np.dot(context.T, d_out)
        d_context = np.dot(d_out, self.W_o.T)
        
        # 2. Backprop through context = attn @ V
        d_attn = np.dot(d_context, V.T)
        d_V = np.dot(attn.T, d_context)
        
        # 3. Backprop through row-wise Softmax
        sum_term = np.sum(d_attn * attn, axis=-1, keepdims=True)
        d_scores = attn * (d_attn - sum_term) / np.sqrt(self.head_dim)
        
        # 4. Backprop through scores = Q @ K.T
        d_Q = np.dot(d_scores, K)
        d_K = np.dot(d_scores.T, Q)
        
        # 5. Backprop through Q, K, V linear projections
        d_W_q = np.dot(x.T, d_Q)
        d_W_k = np.dot(x.T, d_K)
        d_W_v = np.dot(x.T, d_V)
        
        d_x = (
            np.dot(d_Q, self.W_q.T)
            + np.dot(d_K, self.W_k.T)
            + np.dot(d_V, self.W_v.T)
        )
        
        # 6. Update weights
        self.W_o -= learning_rate * d_W_o
        self.W_q -= learning_rate * d_W_q
        self.W_k -= learning_rate * d_W_k
        self.W_v -= learning_rate * d_W_v
        
        return d_x


# =====================================================================
# Multi-Head Causal Attention with RoPE & KV Caching
# =====================================================================

class MultiHeadAttention:
    """
    Multi-Head Causal Self-Attention (Vaswani et al., 2017):
    MultiHead(Q, K, V) = Concat(head_1, ..., head_H) @ W_o
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_rope: bool = False,
        causal: bool = True,
        max_seq_len: int = 2048
    ):
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        self.causal = causal
        self.max_seq_len = max_seq_len

        scale = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.randn(embed_dim, embed_dim) * scale
        self.W_k = np.random.randn(embed_dim, embed_dim) * scale
        self.W_v = np.random.randn(embed_dim, embed_dim) * scale
        self.W_o = np.random.randn(embed_dim, embed_dim) * scale

        # RoPE precomputations
        if self.use_rope:
            self.cos, self.sin = precompute_freqs_cis(self.head_dim, max_seq_len=max_seq_len)
        else:
            self.cos, self.sin = None, None

        # KV Cache buffers
        self.k_cache: Optional[np.ndarray] = None
        self.v_cache: Optional[np.ndarray] = None
        self.cache = {}

    def reset_cache(self) -> None:
        """Clears past KV cache for a new autoregressive sequence."""
        self.k_cache = None
        self.v_cache = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(
        self,
        x: np.ndarray,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> np.ndarray:
        is_2d = (x.ndim == 2)
        if is_2d:
            x_3d = np.expand_dims(x, axis=0)  # (1, seq_len, embed_dim)
        else:
            x_3d = x

        B, T, D = x_3d.shape
        H, Dh = self.num_heads, self.head_dim

        # 1. Linear projections: (B, T, D)
        Q = np.dot(x_3d, self.W_q)
        K = np.dot(x_3d, self.W_k)
        V = np.dot(x_3d, self.W_v)

        # 2. Reshape to multi-head: (B, H, T, Dh)
        Q = Q.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        # 3. Apply RoPE if enabled
        if self.use_rope:
            cos_slice = self.cos[start_pos : start_pos + T]
            sin_slice = self.sin[start_pos : start_pos + T]
            Q = apply_rotary_emb(Q, cos_slice, sin_slice)
            K = apply_rotary_emb(K, cos_slice, sin_slice)

        # 4. KV Caching for fast generation
        if use_cache:
            if self.k_cache is not None:
                K = np.concatenate([self.k_cache, K], axis=2)
                V = np.concatenate([self.v_cache, V], axis=2)
            self.k_cache = K
            self.v_cache = V
            scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(Dh)
            total_k_len = K.shape[2]
            if T > 1:
                # Causal mask for multi-token prompt prefill
                mask = np.triu(np.ones((T, total_k_len)), k=1 + (total_k_len - T)) * -1e9
                attn_weights = self._softmax(scores + mask)
            else:
                attn_weights = self._softmax(scores)
        else:
            scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(Dh)
            if self.causal:
                mask = np.triu(np.ones((T, T)), k=1) * -1e9
                attn_weights = self._softmax(scores + mask)
            else:
                attn_weights = self._softmax(scores)

        # 5. Context accumulation: (B, H, T, Dh)
        context = np.matmul(attn_weights, V)

        # 6. Merge heads: (B, T, D)
        context_concat = context.transpose(0, 2, 1, 3).reshape(B, T, D)

        # 7. Output projection: (B, T, D)
        out = np.dot(context_concat, self.W_o)

        self.cache = {
            "x": x_3d, "Q": Q, "K": K, "V": V,
            "attn_weights": attn_weights, "context_concat": context_concat,
            "is_2d": is_2d
        }

        if is_2d:
            return out[0]
        return out

    def backward(self, d_out: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        if self.cache["is_2d"] and d_out.ndim == 2:
            d_out_3d = np.expand_dims(d_out, axis=0)
        else:
            d_out_3d = d_out

        x = self.cache["x"]
        Q, K, V = self.cache["Q"], self.cache["K"], self.cache["V"]
        attn = self.cache["attn_weights"]
        context_concat = self.cache["context_concat"]
        B, T, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # 1. Output projection backward
        d_out_flat = d_out_3d.reshape(-1, D)
        ctx_flat = context_concat.reshape(-1, D)
        d_W_o = np.dot(ctx_flat.T, d_out_flat)
        d_context_concat = np.dot(d_out_3d, self.W_o.T)

        # 2. Split into heads: (B, H, T, Dh)
        d_context = d_context_concat.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        # 3. Backprop through context = attn @ V
        d_V = np.matmul(attn.transpose(0, 1, 3, 2), d_context)
        d_attn = np.matmul(d_context, V.transpose(0, 1, 3, 2))

        # 4. Softmax backward: d_scores = attn * (d_attn - sum(d_attn * attn)) / sqrt(Dh)
        sum_term = np.sum(d_attn * attn, axis=-1, keepdims=True)
        d_scores = attn * (d_attn - sum_term) / np.sqrt(Dh)

        # 5. Backprop through scores = Q @ K.T
        d_Q = np.matmul(d_scores, K)
        d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)

        # 6. Flatten heads back to (B, T, D)
        d_Q_flat = d_Q.transpose(0, 2, 1, 3).reshape(-1, D)
        d_K_flat = d_K.transpose(0, 2, 1, 3).reshape(-1, D)
        d_V_flat = d_V.transpose(0, 2, 1, 3).reshape(-1, D)
        x_flat = x.reshape(-1, D)

        # 7. Linear projections backward
        d_W_q = np.dot(x_flat.T, d_Q_flat)
        d_W_k = np.dot(x_flat.T, d_K_flat)
        d_W_v = np.dot(x_flat.T, d_V_flat)

        dx = (
            np.dot(d_Q_flat, self.W_q.T)
            + np.dot(d_K_flat, self.W_k.T)
            + np.dot(d_V_flat, self.W_v.T)
        ).reshape(B, T, D)

        # 8. Update weights
        if learning_rate is not None:
            self.W_o -= learning_rate * d_W_o
            self.W_q -= learning_rate * d_W_q
            self.W_k -= learning_rate * d_W_k
            self.W_v -= learning_rate * d_W_v

        if self.cache["is_2d"]:
            return dx[0]
        return dx


# =====================================================================
# Modern Pre-LN Transformer Block
# =====================================================================

class TransformerBlock:
    """
    Modern Decoder Transformer Block using Pre-LayerNorm / Pre-RMSNorm:
      x1 = x + MultiHeadAttention(Norm1(x))
      x2 = x1 + FFN(Norm2(x1))
    Where FFN is: Linear(D, 4D) -> GELU -> Linear(4D, D)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_rope: bool = False,
        causal: bool = True,
        norm_type: str = "rmsnorm",
        max_seq_len: int = 2048,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm_type = norm_type

        norm_cls = RMSNorm if norm_type == "rmsnorm" else LayerNorm
        self.norm1 = norm_cls(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, use_rope=use_rope, causal=causal, max_seq_len=max_seq_len)
        self.norm2 = norm_cls(embed_dim)

        self.ffn_w1 = Linear(embed_dim, embed_dim * 4)
        self.ffn_act = GELU()
        self.ffn_w2 = Linear(embed_dim * 4, embed_dim)
        self.cache = {}

    def reset_cache(self) -> None:
        self.attn.reset_cache()

    def forward(self, x: np.ndarray, use_cache: bool = False, start_pos: int = 0) -> np.ndarray:
        # 1. Pre-norm attention + residual
        norm1_x = self.norm1.forward(x)
        attn_out = self.attn.forward(norm1_x, use_cache=use_cache, start_pos=start_pos)
        h = x + attn_out

        # 2. Pre-norm FFN + residual
        norm2_h = self.norm2.forward(h)
        ffn_mid = self.ffn_act.forward(self.ffn_w1.forward(norm2_h))
        ffn_out = self.ffn_w2.forward(ffn_mid)
        out = h + ffn_out

        self.cache = {"x": x, "h": h}
        return out

    def backward(self, d_out: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        # Backward through FFN + residual 2 (out = h + ffn_out)
        d_ffn_out = d_out.copy()
        d_h_residual = d_out.copy()

        d_ffn_mid = self.ffn_w2.backward(d_ffn_out, learning_rate)
        d_ffn_act = self.ffn_act.backward(d_ffn_mid)
        d_norm2_h = self.ffn_w1.backward(d_ffn_act, learning_rate)
        d_h_from_ffn = self.norm2.backward(d_norm2_h, learning_rate)

        d_h = d_h_residual + d_h_from_ffn

        # Backward through Attention + residual 1 (h = x + attn_out)
        d_attn_out = d_h.copy()
        d_x_residual = d_h.copy()

        d_norm1_x = self.attn.backward(d_attn_out, learning_rate)
        d_x_from_attn = self.norm1.backward(d_norm1_x, learning_rate)

        dx = d_x_residual + d_x_from_attn
        return dx


# =====================================================================
# Production-Grade Transformer Language Model (TransformerLM)
# =====================================================================

class TransformerLM:
    """
    Multi-layer autoregressive decoder-only Transformer Language Model:
      - Token Embeddings (+ optional Positional Embeddings or RoPE)
      - Stack of L Pre-Norm TransformerBlocks
      - Final Normalization (RMSNorm or LayerNorm)
      - Unembedding Linear Language Modeling Head (logits projection)
      - Fast KV-cached generation with temperature and greedy decoding
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
        max_seq_len: int = 128,
        use_rope: bool = True,
        norm_type: str = "rmsnorm",
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

        self.tok_emb = Embedding(vocab_size, embed_dim)
        if not use_rope:
            self.pos_emb = Embedding(max_seq_len, embed_dim)
        else:
            self.pos_emb = None

        self.blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                use_rope=use_rope,
                norm_type=norm_type,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ]

        norm_cls = RMSNorm if norm_type == "rmsnorm" else LayerNorm
        self.ln_f = norm_cls(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size)
        self.cache = {}

    def reset_cache(self) -> None:
        """Resets KV caches across all stacked Transformer blocks."""
        for block in self.blocks:
            block.reset_cache()

    def forward(
        self,
        token_ids: np.ndarray,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> np.ndarray:
        """
        token_ids: 1D (seq_len,) or 2D (batch_size, seq_len)
        returns logits: shape (..., vocab_size)
        """
        is_1d = (token_ids.ndim == 1)
        if is_1d:
            token_ids_2d = np.expand_dims(token_ids, axis=0)
        else:
            token_ids_2d = token_ids

        B, T = token_ids_2d.shape

        # 1. Embeddings
        x = self.tok_emb.forward(token_ids_2d)
        if self.pos_emb is not None:
            positions = np.arange(start_pos, start_pos + T)
            x = x + self.pos_emb.forward(positions)

        # 2. Stacked Transformer Blocks
        for block in self.blocks:
            x = block.forward(x, use_cache=use_cache, start_pos=start_pos)

        # 3. Final layer normalization
        x_norm = self.ln_f.forward(x)

        # 4. Project to vocabulary logits
        logits = self.lm_head.forward(x_norm)

        self.cache = {
            "token_ids": token_ids_2d,
            "x_emb": x,
            "x_norm": x_norm,
            "is_1d": is_1d
        }

        if is_1d:
            return logits[0]
        return logits

    def backward(self, d_logits: np.ndarray, learning_rate: float) -> None:
        """Backpropagates through LM head, final norm, blocks, and embeddings."""
        if self.cache["is_1d"] and d_logits.ndim == 2:
            d_logits_3d = np.expand_dims(d_logits, axis=0)
        else:
            d_logits_3d = d_logits

        # 1. LM head backward
        d_x_norm = self.lm_head.backward(d_logits_3d, learning_rate)

        # 2. Final norm backward
        d_x = self.ln_f.backward(d_x_norm, learning_rate)

        # 3. Reverse pass through Transformer blocks
        for block in reversed(self.blocks):
            d_x = block.backward(d_x, learning_rate)

        # 4. Embeddings backward
        self.tok_emb.backward(d_x, learning_rate)
        if self.pos_emb is not None:
            T = self.cache["token_ids"].shape[1]
            positions = np.arange(T)
            d_pos = np.mean(d_x, axis=0) if d_x.ndim == 3 else d_x
            self.pos_emb.backward(d_pos, learning_rate)

    def generate(
        self,
        start_ids: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Autoregressively generates new tokens one by one with optional KV caching.
        """
        generated = list(start_ids)
        self.reset_cache()

        if use_cache:
            # Prefill phase: process initial prompt to fill KV caches
            prompt = np.array(generated, dtype=int)
            logits = self.forward(prompt, use_cache=True, start_pos=0)
            last_logits = logits[-1]
            if temperature > 0.0:
                probs = softmax(last_logits / temperature)
                next_id = int(np.argmax(probs))
            else:
                next_id = int(np.argmax(last_logits))
            generated.append(next_id)

            # Incremental decoding phase: 1 token at a time
            for _ in range(1, max_new_tokens):
                curr_tok = np.array([generated[-1]], dtype=int)
                curr_pos = len(generated) - 1
                logits = self.forward(curr_tok, use_cache=True, start_pos=curr_pos)
                last_logits = logits[0] if logits.ndim == 2 else logits
                if temperature > 0.0:
                    probs = softmax(last_logits / temperature)
                    next_id = int(np.argmax(probs))
                else:
                    next_id = int(np.argmax(last_logits))
                generated.append(next_id)
        else:
            for _ in range(max_new_tokens):
                context = np.array(generated[-self.max_seq_len:], dtype=int)
                logits = self.forward(context, use_cache=False)
                last_logits = logits[-1]
                if temperature > 0.0:
                    probs = softmax(last_logits / temperature)
                    next_id = int(np.argmax(probs))
                else:
                    next_id = int(np.argmax(last_logits))
                generated.append(next_id)

        return np.array(generated, dtype=int)


# =====================================================================
# Backward-Compatible MiniGPT Baseline
# =====================================================================

class MiniGPT:
    """A complete, trainable decoder-only Transformer (GPT) built from scratch."""
    def __init__(self, vocab_size: int, embed_dim: int = 32, max_seq_len: int = 64):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # 1. Embeddings
        self.tok_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(max_seq_len, embed_dim)

        # 2. Transformer Block (Causal Self-Attention + Feed-Forward Network)
        self.attn = SelfAttention(embed_dim=embed_dim, head_dim=embed_dim)
        self.ffn = Sequential([
            Linear(embed_dim, embed_dim * 2),
            ReLU(),
            Linear(embed_dim * 2, embed_dim)
        ])

        # 3. Output projection to vocabulary logits
        self.lm_head = Linear(embed_dim, vocab_size)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        seq_len = len(token_ids)
        positions = np.arange(seq_len)

        x = self.tok_emb.forward(token_ids) + self.pos_emb.forward(positions)
        x = x + self.attn.forward(x)
        x = x + self.ffn.forward(x)
        logits = self.lm_head.forward(x)
        return logits

    def backward(self, d_logits: np.ndarray, learning_rate: float) -> None:
        d_x = self.lm_head.backward(d_logits, learning_rate)
        d_ffn = self.ffn.backward(d_x, learning_rate)
        d_x = d_x + d_ffn
        d_attn = self.attn.backward(d_x, learning_rate)
        d_x = d_x + d_attn
        self.tok_emb.backward(d_x, learning_rate)
        self.pos_emb.backward(d_x, learning_rate)

    def generate(self, start_ids: np.ndarray, max_new_tokens: int) -> np.ndarray:
        generated = list(start_ids)
        for _ in range(max_new_tokens):
            context = np.array(generated[-self.max_seq_len:], dtype=int)
            logits = self.forward(context)
            last_token_logits = logits[-1:]
            probs = softmax(last_token_logits)
            next_id = int(np.argmax(probs))
            generated.append(next_id)
        return np.array(generated, dtype=int)
