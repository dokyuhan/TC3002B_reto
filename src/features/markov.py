"""Markov transition matrix of one AST state sequence."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from src.vocabulary import NUM_STATES, PAD_TOKEN


def clean_sequence(
    seq: Sequence[int] | np.ndarray,
    num_states: int = NUM_STATES,
    pad_token: int = PAD_TOKEN,
) -> np.ndarray:
    """Convert to 1D integer array, strip padding, and validate [0, num_states - 1]."""
    arr = np.asarray(seq, dtype=np.int64).ravel()
    arr = arr[arr != pad_token]

    if arr.size > 0 and (arr.min() < 0 or arr.max() >= num_states):
        raise ValueError(
            f"State IDs out of range [0, {num_states - 1}]: "
            f"min={arr.min()}, max={arr.max()}"
        )
    return arr


def sequence_to_transition_matrix(
    seq: Sequence[int] | np.ndarray,
    num_states: int = NUM_STATES,
    pad_token: int = PAD_TOKEN,
    smoothing_alpha: float = 0.0,
) -> np.ndarray:
    """Build a (K, K) float32 transition matrix with optional Laplace smoothing."""
    if smoothing_alpha < 0.0:
        raise ValueError(f"smoothing_alpha must be >= 0.0, got {smoothing_alpha}")

    arr = clean_sequence(seq, num_states=num_states, pad_token=pad_token)
    counts = np.zeros((num_states, num_states), dtype=np.float64)

    if arr.size >= 2:
        np.add.at(counts, (arr[:-1], arr[1:]), 1.0)

    if smoothing_alpha > 0.0:
        counts += smoothing_alpha

    row_sums = counts.sum(axis=1, keepdims=True)
    # Leave rows with zero outgoing transitions as 0.0 when smoothing_alpha == 0.0.
    matrix = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0.0)
    return matrix.astype(np.float32)
