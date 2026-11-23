"""Markov transition matrix and pairwise tensor computation."""

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


def pair_to_tensor(
    seq_a: Sequence[int] | np.ndarray,
    seq_b: Sequence[int] | np.ndarray,
    num_states: int = NUM_STATES,
    pad_token: int = PAD_TOKEN,
    smoothing_alpha: float = 0.0,
) -> np.ndarray:
    """Build a (3, K, K) tensor with channels [P_A, P_B, |P_A - P_B|]."""
    p_a = sequence_to_transition_matrix(
        seq_a,
        num_states=num_states,
        pad_token=pad_token,
        smoothing_alpha=smoothing_alpha,
    )
    p_b = sequence_to_transition_matrix(
        seq_b,
        num_states=num_states,
        pad_token=pad_token,
        smoothing_alpha=smoothing_alpha,
    )
    return np.stack([p_a, p_b, np.abs(p_a - p_b)], axis=0).astype(np.float32)
