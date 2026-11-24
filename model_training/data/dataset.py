"""PyTorch Dataset wrapper for pairwise transition matrix tensors."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class PlagiarismDataset(Dataset):
    """Dataset storing `(N, C, K, K)` tensors and binary targets as float32."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Convert `X` and `y` to contiguous float32 tensors after validating lengths."""
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        """Return dataset sample count."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(features, label)` for sample `idx`."""
        return self.X[idx], self.y[idx]
