import numpy as np

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
