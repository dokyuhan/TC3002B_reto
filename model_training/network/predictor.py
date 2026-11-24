"""PyTorch checkpoint predictor aligned with runtime permutation symmetry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model_training.data.features import validate_features
from model_training.network.checkpoints import load_checkpoint
from model_training.utils import get_device


def _swap_pair_channels(x: np.ndarray) -> np.ndarray:
    """Swap `(P_A, P_B, P_A - P_B)` into `(P_B, P_A, P_B - P_A)` for symmetric inference."""
    swapped = x[:, [1, 0, 2], :, :].copy()
    swapped[:, 2, :, :] = -swapped[:, 2, :, :]
    return swapped


class Predictor:
    """Loaded CNN checkpoint wrapper using checkpoint threshold unless overridden."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        threshold: float | None = None,
    ) -> None:
        """Load checkpoint weights and validate operating threshold in `(0, 1)`."""
        self.device = get_device(device)
        self.model, ckpt = load_checkpoint(checkpoint_path, self.device)
        self.config = ckpt["model_config"]
        self.input_spec = ckpt["extra"].get("input_spec")
        self.threshold = float(ckpt["threshold"] if threshold is None else threshold)
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"Threshold must be in (0, 1), got {self.threshold}")

    @torch.no_grad()
    def predict_proba(
        self,
        X: np.ndarray | torch.Tensor,
        batch_size: int = 256,
        symmetric: bool = True,
    ) -> np.ndarray:
        """Return positive class probabilities of shape `(N,)`, symmetrized by default."""
        arr = self._prepare(X)
        if len(arr) == 0:
            return np.empty((0,), dtype=np.float64)

        probs_ab = self._predict_raw_proba(arr, batch_size=batch_size)
        if not symmetric or arr.shape[1] != 3:
            return probs_ab

        probs_ba = self._predict_raw_proba(_swap_pair_channels(arr), batch_size=batch_size)
        return 0.5 * (probs_ab + probs_ba)

    def predict(
        self,
        X: np.ndarray | torch.Tensor,
        batch_size: int = 256,
        symmetric: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(probabilities, binary_labels)` at `self.threshold`."""
        probs = self.predict_proba(X, batch_size=batch_size, symmetric=symmetric)
        return probs, (probs >= self.threshold).astype(int)

    def _prepare(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None]
        validate_features(arr)
        expected = (self.config["in_channels"], self.config["n_states"], self.config["n_states"])
        if arr.shape[1:] != expected:
            raise ValueError(f"Input shape {arr.shape[1:]} does not match model {expected}")
        return arr

    @torch.no_grad()
    def _predict_raw_proba(self, arr: np.ndarray, batch_size: int) -> np.ndarray:
        out = []
        for i in range(0, len(arr), batch_size):
            batch = torch.from_numpy(arr[i : i + batch_size]).to(self.device)
            out.append(torch.sigmoid(self.model(batch)).cpu().numpy())
        return np.concatenate(out).astype(np.float64)


def predict(
    X: np.ndarray | torch.Tensor,
    checkpoint_path: str | Path,
    device: str = "auto",
    threshold: float | None = None,
    symmetric: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load checkpoint and return `(probabilities, binary_labels)` in one call."""
    return Predictor(checkpoint_path, device, threshold).predict(X, symmetric=symmetric)
