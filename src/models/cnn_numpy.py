"""Numpy-only forward pass of the trained plagiarism CNN (no torch needed at inference time).

The network is trained with PyTorch in `model_training/`; `model_training/export_numpy.py` copies its weights to
`models/cnn_weights.npz`. Architecture (eval mode, so dropout is off):

    (3, K, K) -> Conv3x3(pad 1)+ReLU -> MaxPool2 -> Conv3x3(pad 1)+ReLU -> AdaptiveAvgPool(4x4)
              -> flatten -> Linear+ReLU -> Linear -> sigmoid = P(plagiarism)

Input channels are [P_A, P_B, P_A - P_B] with *unsmoothed* transition matrices, exactly as in
training (see `model_training/data/build_matrices.py`). The network itself is not symmetric in (A, B), so
`predict_pairs` averages the probability of (A, B) and (B, A).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_WEIGHT_KEYS = ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b")


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 convolution with stride 1 and zero padding 1: (N, C, H, W) -> (N, O, H, W)."""
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    windows = sliding_window_view(padded, (3, 3), axis=(2, 3))  # (N, C, H, W, 3, 3)
    return np.einsum("nchwij,ocij->nohw", windows, w, optimize=True) + b[None, :, None, None]


def _max_pool2(x: np.ndarray) -> np.ndarray:
    """2x2 max pooling with stride 2 (odd trailing rows/columns are dropped, like torch)."""
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    return x[:, :, : h2 * 2, : w2 * 2].reshape(n, c, h2, 2, w2, 2).max(axis=(3, 5))


def _adaptive_avg_pool(x: np.ndarray, size: int) -> np.ndarray:
    """Adaptive average pooling to (size, size), with the same window rule as torch."""
    n, c, h, w = x.shape
    out = np.empty((n, c, size, size), dtype=x.dtype)
    for i in range(size):
        h0, h1 = (i * h) // size, -((-(i + 1) * h) // size)  # floor, ceil
        for j in range(size):
            w0, w1 = (j * w) // size, -((-(j + 1) * w) // size)
            out[:, :, i, j] = x[:, :, h0:h1, w0:w1].mean(axis=(2, 3))
    return out


class NumpyCnnClassifier:
    """Pairwise plagiarism CNN evaluated with numpy (the only classifier the app uses)."""

    def __init__(self, weights: dict[str, np.ndarray], n_states: int, pool_size: int = 4) -> None:
        """Store float32 weights and the geometry needed for the forward pass."""
        missing = [k for k in _WEIGHT_KEYS if k not in weights]
        if missing:
            raise ValueError(f"Missing CNN weights: {missing}")
        self.weights = {k: np.asarray(weights[k], dtype=np.float32) for k in _WEIGHT_KEYS}
        self.n_states = int(n_states)
        self.pool_size = int(pool_size)
        self.in_channels = int(self.weights["conv1_w"].shape[1])
        if self.in_channels != 3:
            raise ValueError(f"Expected a 3-channel CNN, got {self.in_channels} channels")

    @classmethod
    def load(cls, path: str | Path) -> NumpyCnnClassifier:
        """Load weights saved by `model_training/export_numpy.py`."""
        with np.load(path, allow_pickle=False) as data:
            weights = {k: data[k] for k in _WEIGHT_KEYS}
            return cls(weights, n_states=int(data["n_states"]), pool_size=int(data["pool_size"]))

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Run the network on (N, 3, K, K) inputs and return one logit per sample, shape (N,)."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != (self.in_channels, self.n_states, self.n_states):
            raise ValueError(
                f"Expected (N, {self.in_channels}, {self.n_states}, {self.n_states}), got {x.shape}"
            )
        w = self.weights
        x = np.maximum(_conv3x3(x, w["conv1_w"], w["conv1_b"]), 0.0)
        if self.n_states >= 4:  # same rule as the torch model: avoid shrinking tiny matrices
            x = _max_pool2(x)
        x = np.maximum(_conv3x3(x, w["conv2_w"], w["conv2_b"]), 0.0)
        x = _adaptive_avg_pool(x, self.pool_size).reshape(len(x), -1)
        x = np.maximum(x @ w["fc1_w"].T + w["fc1_b"], 0.0)
        return (x @ w["fc2_w"].T + w["fc2_b"]).reshape(-1)

    def predict_proba_tensor(self, x: np.ndarray) -> np.ndarray:
        """P(plagiarism) for already-built (N, 3, K, K) inputs, in the given (A, B) order."""
        logits = self.predict_logits(x).astype(np.float64)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_pairs(self, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        """Symmetric P(plagiarism) for batches of unsmoothed matrices p_a, p_b of shape (N, K, K)."""
        p_a = np.asarray(p_a, dtype=np.float32)
        p_b = np.asarray(p_b, dtype=np.float32)
        ab = np.stack([p_a, p_b, p_a - p_b], axis=1)
        ba = np.stack([p_b, p_a, p_b - p_a], axis=1)
        return 0.5 * (self.predict_proba_tensor(ab) + self.predict_proba_tensor(ba))

    def predict_pair(self, p_a: np.ndarray, p_b: np.ndarray) -> float:
        """Symmetric P(plagiarism) for one pair of (K, K) unsmoothed transition matrices."""
        return float(self.predict_pairs(p_a[np.newaxis], p_b[np.newaxis])[0])
