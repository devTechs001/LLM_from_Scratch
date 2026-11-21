import numpy as np
from typing import List, Tuple, Optional, Any, Callable

# =====================================================================
# Universal Dataset & DataLoader
# =====================================================================

class Dataset:
    """Base dataset class matching PyTorch / standard ML dataset contracts."""
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError


class TensorDataset(Dataset):
    """Wraps arrays of inputs and targets into an indexed dataset."""
    def __init__(self, *tensors: np.ndarray):
        assert len(tensors) > 0, "At least one tensor required"
        n = len(tensors[0])
        assert all(len(t) == n for t in tensors), "All tensors must have the same leading dimension size"
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ...]:
        return tuple(t[idx] for t in self.tensors)


class DataLoader:
    """
    Mini-batch iterator supporting:
      - Batch collation
      - Dataset shuffling
      - Drop last / remainder batch handling
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn if collate_fn is not None else self._default_collate

    def _default_collate(self, batch: List[Tuple[np.ndarray, ...]]) -> Tuple[np.ndarray, ...]:
        num_fields = len(batch[0])
        return tuple(np.array([item[i] for item in batch]) for i in range(num_fields))

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            batch_indices = indices[start_idx:end_idx]
            batch_items = [self.dataset[i] for i in batch_indices]
            yield self.collate_fn(batch_items)


# =====================================================================
# Continuous 1D Signal & Audio Spectrogram Tokenizer
# =====================================================================

class AudioSpectrogramTokenizer:
    """
    Converts 1D continuous waveform audio signals into short-time spectral
    frames and linearly projects them into Transformer token embeddings:
      Waveform (B, L) -> Spectrogram Frames (B, T_frames, frame_size) -> Tokens (B, T_frames, embed_dim)
    """
    def __init__(
        self,
        frame_size: int = 16,
        hop_size: int = 8,
        embed_dim: int = 32
    ):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.embed_dim = embed_dim
        # Projection from raw spectral frame to embedding dimension
        scale = np.sqrt(2.0 / frame_size)
        self.W_proj = np.random.randn(frame_size, embed_dim) * scale
        self.b_proj = np.zeros((1, embed_dim))

    def forward(self, waveforms: np.ndarray) -> np.ndarray:
        """
        waveforms: (B, L) continuous 1D audio samples
        returns: (B, num_frames, embed_dim) sequence of audio tokens
        """
        B, L = waveforms.shape
        num_frames = (L - self.frame_size) // self.hop_size + 1
        assert num_frames > 0, f"Waveform length ({L}) too short for frame size ({self.frame_size})"

        frames = np.empty((B, num_frames, self.frame_size))
        for t in range(num_frames):
            start = t * self.hop_size
            end = start + self.frame_size
            frames[:, t, :] = waveforms[:, start:end]

        # Apply Hanning window to reduce spectral leakage
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(self.frame_size) / (self.frame_size - 1)))
        windowed_frames = frames * window

        # Linear projection to Transformer embeddings
        tokens = np.dot(windowed_frames, self.W_proj) + self.b_proj
        return tokens


# =====================================================================
# Tabular & Continuous Sensor Feature Encoder
# =====================================================================

class ContinuousFeatureEncoder:
    """
    Normalizes tabular and sensor features and projects them into unified
    Transformer token representations.
    """
    def __init__(self, in_features: int, embed_dim: int):
        self.in_features = in_features
        self.embed_dim = embed_dim
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, embed_dim) * scale
        self.bias = np.zeros((1, embed_dim))
        self.mean = None
        self.std = None

    def fit_normalize(self, X: np.ndarray) -> np.ndarray:
        """Computes feature statistics and applies z-score normalization."""
        self.mean = np.mean(X, axis=0, keepdims=True)
        self.std = np.std(X, axis=0, keepdims=True) + 1e-8
        return (X - self.mean) / self.std

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Projects continuous features into dense token vectors (B, 1, embed_dim)."""
        if self.mean is not None:
            X_norm = (X - self.mean) / self.std
        else:
            X_norm = X
        proj = np.dot(X_norm, self.weights) + self.bias
        return np.expand_dims(proj, axis=1)


# =====================================================================
# Image Augmentation Primitives (Pure NumPy)
# =====================================================================

def random_horizontal_flip(images: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Randomly flips images horizontally along width axis with probability p."""
    out = images.copy()
    for i in range(len(images)):
        if np.random.rand() < p:
            out[i] = np.flip(out[i], axis=-1)
    return out


def random_erasing(images: np.ndarray, p: float = 0.5, erase_size: int = 2) -> np.ndarray:
    """Cutout/Random Erasing: masks out random square regions to prevent feature reliance."""
    out = images.copy()
    B, C, H, W = images.shape
    for i in range(B):
        if np.random.rand() < p:
            top = np.random.randint(0, max(1, H - erase_size))
            left = np.random.randint(0, max(1, W - erase_size))
            out[i, :, top : top + erase_size, left : left + erase_size] = 0.0
    return out
