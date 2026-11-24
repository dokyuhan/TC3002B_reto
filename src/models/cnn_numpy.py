"""NumPy forward pass for the trained pairwise plagiarism CNN.

The network is trained with PyTorch in `model_training/` and exported to
`models/cnn_weights.npz` by `model_training/export_numpy.py`. Architecture in
eval mode (dropout disabled):

    (3, K, K) -> Conv3x3(pad 1)+ReLU -> MaxPool2 -> Conv3x3(pad 1)+ReLU
              -> AdaptiveAvgPool(pool_size) -> flatten -> Linear+ReLU
              -> Linear -> sigmoid = P(plagiarism)

Input channels are `[P_A, P_B, P_A - P_B]` using unsmoothed transition
matrices. Because the underlying network is asymmetric in `(A, B)`,
`predict_pairs` averages `P(A, B)` and `P(B, A)`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import expit

_WEIGHT_KEYS = ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b")


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply 3x3 convolution with stride 1 and zero padding 1: (N, C, H, W) -> (N, O, H, W)."""
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    windows = sliding_window_view(padded, (3, 3), axis=(2, 3))
    return np.einsum("nchwij,ocij->nohw", windows, w, optimize=True) + b[None, :, None, None]


def _max_pool2(x: np.ndarray) -> np.ndarray:
    """Apply 2x2 max pooling with stride 2, dropping odd trailing rows or columns."""
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    return x[:, :, : h2 * 2, : w2 * 2].reshape(n, c, h2, 2, w2, 2).max(axis=(3, 5))


def _adaptive_avg_pool(x: np.ndarray, size: int) -> np.ndarray:
    """Apply adaptive average pooling to `(size, size)` matching PyTorch window bounds."""
    n, c, h, w = x.shape
    out = np.empty((n, c, size, size), dtype=x.dtype)
    for i in range(size):
        h0, h1 = (i * h) // size, -((-(i + 1) * h) // size)
        for j in range(size):
            w0, w1 = (j * w) // size, -((-(j + 1) * w) // size)
            out[:, :, i, j] = x[:, :, h0:h1, w0:w1].mean(axis=(2, 3))
    return out


class NumpyCnnClassifier:
    """Pairwise plagiarism CNN evaluated strictly in NumPy."""

    def __init__(self, weights: dict[str, np.ndarray], n_states: int, pool_size: int = 4) -> None:
        """Validate weight tensor shapes and store geometry for inference."""
        missing = [k for k in _WEIGHT_KEYS if k not in weights]
        if missing:
            raise ValueError(f"Missing CNN weights: {missing}")

        self.weights = {k: np.asarray(weights[k], dtype=np.float32) for k in _WEIGHT_KEYS}
        self.n_states = int(n_states)
        self.pool_size = int(pool_size)
        if self.n_states < 1 or self.pool_size < 1:
            raise ValueError(
                f"n_states and pool_size must be >= 1, got {self.n_states}, {self.pool_size}"
            )

        self._validate_weight_shapes()
        self.in_channels = int(self.weights["conv1_w"].shape[1])

    @classmethod
    def load(cls, path: str | Path) -> NumpyCnnClassifier:
        """Load weights exported by `model_training/export_numpy.py`."""
        with np.load(path, allow_pickle=False) as data:
            missing_weights = [k for k in _WEIGHT_KEYS if k not in data.files]
            if missing_weights:
                raise ValueError(f"Missing CNN weights: {missing_weights}")
            missing_meta = [k for k in ("n_states", "pool_size") if k not in data.files]
            if missing_meta:
                raise ValueError(f"Missing CNN metadata keys: {missing_meta}")
            weights = {k: data[k] for k in _WEIGHT_KEYS}
            return cls(weights, n_states=int(data["n_states"]), pool_size=int(data["pool_size"]))

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Run the CNN on `(N, 3, K, K)` inputs and return logits of shape `(N,)`."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != (self.in_channels, self.n_states, self.n_states):
            raise ValueError(
                f"Expected (N, {self.in_channels}, {self.n_states}, {self.n_states}), got {x.shape}"
            )
        if x.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        w = self.weights
        x = np.maximum(_conv3x3(x, w["conv1_w"], w["conv1_b"]), 0.0)
        if self.n_states >= 4:
            x = _max_pool2(x)
        x = np.maximum(_conv3x3(x, w["conv2_w"], w["conv2_b"]), 0.0)
        x = _adaptive_avg_pool(x, self.pool_size).reshape(len(x), -1)
        x = np.maximum(x @ w["fc1_w"].T + w["fc1_b"], 0.0)
        return (x @ w["fc2_w"].T + w["fc2_b"]).reshape(-1)

    def predict_proba_tensor(self, x: np.ndarray) -> np.ndarray:
        """Return `P(plagiarism)` for `(N, 3, K, K)` inputs in the supplied `(A, B)` order."""
        logits = self.predict_logits(x).astype(np.float64)
        return np.asarray(expit(logits), dtype=np.float64)

    def predict_pairs(self, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        """Return permutation symmetric `P(plagiarism)` for batches of `(N, K, K)` matrices."""
        p_a = np.asarray(p_a, dtype=np.float32)
        p_b = np.asarray(p_b, dtype=np.float32)
        if (
            p_a.ndim != 3
            or p_b.ndim != 3
            or p_a.shape != p_b.shape
            or p_a.shape[1:] != (self.n_states, self.n_states)
        ):
            raise ValueError(
                f"Expected matching (N, {self.n_states}, {self.n_states}) arrays, "
                f"got {p_a.shape} and {p_b.shape}"
            )
        if p_a.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        ab = np.stack([p_a, p_b, p_a - p_b], axis=1)
        ba = np.stack([p_b, p_a, p_b - p_a], axis=1)
        return 0.5 * (self.predict_proba_tensor(ab) + self.predict_proba_tensor(ba))

    def predict_pair(self, p_a: np.ndarray, p_b: np.ndarray) -> float:
        """Return permutation symmetric `P(plagiarism)` for one pair of `(K, K)` matrices."""
        return float(self.predict_pairs(p_a[np.newaxis], p_b[np.newaxis])[0])

    def _validate_weight_shapes(self) -> None:
        w = self.weights
        if w["conv1_w"].ndim != 4 or w["conv1_w"].shape[2:] != (3, 3):
            raise ValueError(f"Invalid conv1_w shape: {w['conv1_w'].shape}")
        c1, in_ch = int(w["conv1_w"].shape[0]), int(w["conv1_w"].shape[1])
        if in_ch != 3:
            raise ValueError(f"Expected a 3-channel CNN, got {in_ch} channels")
        if w["conv1_b"].shape != (c1,):
            raise ValueError(f"Invalid conv1_b shape: expected ({c1},), got {w['conv1_b'].shape}")

        if w["conv2_w"].ndim != 4 or w["conv2_w"].shape[1:] != (c1, 3, 3):
            raise ValueError(
                f"Invalid conv2_w shape: expected (c2, {c1}, 3, 3), got {w['conv2_w'].shape}"
            )
        c2 = int(w["conv2_w"].shape[0])
        if w["conv2_b"].shape != (c2,):
            raise ValueError(f"Invalid conv2_b shape: expected ({c2},), got {w['conv2_b'].shape}")

        flat_dim = c2 * self.pool_size * self.pool_size
        if w["fc1_w"].ndim != 2 or w["fc1_w"].shape[1] != flat_dim:
            raise ValueError(
                f"Invalid fc1_w shape: expected (hidden, {flat_dim}), got {w['fc1_w'].shape}"
            )
        hidden = int(w["fc1_w"].shape[0])
        if w["fc1_b"].shape != (hidden,):
            raise ValueError(f"Invalid fc1_b shape: expected ({hidden},), got {w['fc1_b'].shape}")

        if w["fc2_w"].shape != (1, hidden):
            raise ValueError(f"Invalid fc2_w shape: expected (1, {hidden}), got {w['fc2_w'].shape}")
        if w["fc2_b"].shape != (1,):
            raise ValueError(f"Invalid fc2_b shape: expected (1,), got {w['fc2_b'].shape}")
