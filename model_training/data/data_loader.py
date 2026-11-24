"""Split loading and PyTorch DataLoader builders for CNN training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_training.data.dataset import PlagiarismDataset
from model_training.data.features import validate_features


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    """Read an `.npz` split containing `X`, `y`, and optional `group`."""
    with np.load(path) as f:
        missing = {"X", "y"} - set(f.files)
        if missing:
            raise KeyError(f"{path} is missing required keys {sorted(missing)} (found {f.files})")
        data = {"X": f["X"], "y": f["y"]}
        if "group" in f.files:
            data["group"] = f["group"]
    return data


_LOADERS = {".npz": _load_npz}


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a split, returning float32 `X`, int64 `y`, and optional `group`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    loader = _LOADERS.get(path.suffix)
    if loader is None:
        raise ValueError(f"Unsupported format: '{path.suffix}' (supported: {sorted(_LOADERS)})")
    data = loader(path)

    X, y = data["X"], np.asarray(data["y"])
    validate_features(X)
    if y.shape != (len(X),):
        raise ValueError(f"y must have shape ({len(X)},), got {y.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only binary labels 0 and 1")
    data["X"] = X.astype(np.float32)
    data["y"] = y.astype(np.int64)
    return data


def get_data_info(X: np.ndarray) -> dict[str, int]:
    """Extract `in_channels` and `n_states` from tensor shape `(N, C, K, K)`."""
    return {"in_channels": int(X.shape[1]), "n_states": int(X.shape[2])}


def make_dataloader(
    data: dict[str, np.ndarray],
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """Create a reproducible PyTorch `DataLoader` from a validated split dictionary."""
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        PlagiarismDataset(data["X"], data["y"]),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
