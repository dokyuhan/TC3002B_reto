"""Checkpoint persistence helpers for `PlagiarismCNN`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from model_training.network.cnn_torch import PlagiarismCNN

CHECKPOINT_VERSION = 2

_LEGACY_KEY_MAP = {
    "features.0.weight": "conv1.weight",
    "features.0.bias": "conv1.bias",
    "features.3.weight": "conv2.weight",
    "features.3.bias": "conv2.bias",
    "classifier.2.weight": "fc1.weight",
    "classifier.2.bias": "fc1.bias",
    "classifier.5.weight": "fc2.weight",
    "classifier.5.bias": "fc2.bias",
}


def _plain(obj: Any) -> Any:
    """Force plain JSON serializable Python types for checkpoint metadata."""
    return json.loads(json.dumps(obj))


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map legacy Sequential checkpoint keys to named `PlagiarismCNN` layer keys."""
    return {_LEGACY_KEY_MAP.get(key, key): value for key, value in state_dict.items()}


def save_checkpoint(
    path: str | Path,
    model: PlagiarismCNN,
    threshold: float = 0.5,
    extra: dict | None = None,
) -> None:
    """Save model weights, architecture config, threshold, and extra metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_config": _plain(model.config),
            "threshold": float(threshold),
            "extra": _plain(extra or {}),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[PlagiarismCNN, dict]:
    """Reconstruct `PlagiarismCNN` in eval mode and return `(model, checkpoint)`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = normalize_state_dict(ckpt["state_dict"])
    ckpt["state_dict"] = state_dict
    model = PlagiarismCNN.from_config(ckpt["model_config"])
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, ckpt
