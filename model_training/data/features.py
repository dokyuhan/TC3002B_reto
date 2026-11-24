"""Feature tensor validation for pairwise Markov transition matrices."""

from __future__ import annotations

import numpy as np

from src.vocabulary import DEFAULT_AST_STATES

INPUT_SPEC = {
    "normalization": "row_normalized",
    "smoothing_alpha": 0.0,
    "channels": ["P_A", "P_B", "P_A - P_B"],
    "state_order": list(DEFAULT_AST_STATES),
}

N_MATRIX_CHANNELS = 2


def validate_transition_matrices(P: np.ndarray, atol: float = 1e-4) -> None:
    """Validate that `(..., K, K)` transition matrices are finite, in `[0, 1]`, and row sum to 0 or 1."""
    P = np.asarray(P, dtype=np.float64)
    if not np.isfinite(P).all():
        raise ValueError("Transition matrices contain NaN or infinite values")
    if P.size == 0:
        return
    if P.min() < -atol or P.max() > 1 + atol:
        raise ValueError(f"Transition values outside [0, 1]: min={P.min():.4f}, max={P.max():.4f}")
    sums = P.sum(axis=-1)
    ok = (np.abs(sums - 1.0) <= atol) | (np.abs(sums) <= atol)
    if not ok.all():
        raise ValueError(f"{(~ok).sum()} rows do not sum to 0 or 1 within tolerance")


def validate_features(X: np.ndarray, atol: float = 1e-4) -> None:
    """Validate `(N, C, K, K)` CNN inputs across base transition and derived difference channels."""
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"X must be 4D (N, C, K, K), got {X.shape}")
    if X.shape[2] != X.shape[3]:
        raise ValueError(f"Transition matrices must be square, got {X.shape[2:]}")
    if X.shape[0] == 0:
        return
    for c in range(min(X.shape[1], N_MATRIX_CHANNELS)):
        validate_transition_matrices(X[:, c], atol=atol)
    if X.shape[1] > N_MATRIX_CHANNELS:
        extra = np.asarray(X[:, N_MATRIX_CHANNELS:], dtype=np.float64)
        if not np.isfinite(extra).all() or extra.min() < -1 - atol or extra.max() > 1 + atol:
            raise ValueError("Derived difference channels must be finite and within [-1, 1]")
