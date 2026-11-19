import numpy as np
from ml_lib.nn import Linear, ReLU, Sequential
from ml_lib.losses import softmax, cross_entropy_loss, cross_entropy_softmax_gradient

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

class SelfAttention:
    """Single-head Causal Self-Attention mechanism (GPT-style)."""
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
        # For each row i: d_scores_i = attn_i * (d_attn_i - sum(d_attn_i * attn_i))
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

# ... (keep existing CharTokenizer, Embedding, and SelfAttention classes above) ...

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
        """
        token_ids: 1D array of shape (seq_len,)
        returns logits: shape (seq_len, vocab_size)
        """
        seq_len = len(token_ids)
        positions = np.arange(seq_len)

        # Combine token and positional embeddings
        x = self.tok_emb.forward(token_ids) + self.pos_emb.forward(positions)

        # Self-Attention with residual connection
        x = x + self.attn.forward(x)

        # Feed-Forward Network with residual connection
        x = x + self.ffn.forward(x)

        # Project to vocabulary logits
        logits = self.lm_head.forward(x)
        return logits

    def backward(self, d_logits: np.ndarray, learning_rate: float) -> None:
        """Backpropagates gradients through the entire Transformer."""
        # 1. Language model head
        d_x = self.lm_head.backward(d_logits, learning_rate)

        # 2. Feed-Forward Network + residual skip connection
        d_ffn = self.ffn.backward(d_x, learning_rate)
        d_x = d_x + d_ffn

        # 3. Self-Attention + residual skip connection
        d_attn = self.attn.backward(d_x, learning_rate)
        d_x = d_x + d_attn

        # 4. Token & Positional Embeddings
        self.tok_emb.backward(d_x, learning_rate)
        self.pos_emb.backward(d_x, learning_rate)

    def generate(self, start_ids: np.ndarray, max_new_tokens: int) -> np.ndarray:
        """Autoregressively generates new tokens one by one."""
        generated = list(start_ids)
        for _ in range(max_new_tokens):
            # Crop context to max_seq_len if needed
            context = np.array(generated[-self.max_seq_len:], dtype=int)
            logits = self.forward(context)
            
            # Take logits of the very last token and pick the highest-probability next token
            last_token_logits = logits[-1:]
            probs = softmax(last_token_logits)
            next_id = int(np.argmax(probs))
            generated.append(next_id)
            
        return np.array(generated, dtype=int)
