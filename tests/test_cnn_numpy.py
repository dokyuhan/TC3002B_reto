"""Tests for the numpy forward pass of the AI-code detector CNN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features.markov import sequence_to_transition_matrix
from src.models.cnn_numpy import NumpyCnnClassifier, _adaptive_avg_pool, _conv3x3, _max_pool2

K = 12


def random_weights(seed: int = 0, in_channels: int = 1) -> dict[str, np.ndarray]:
    """Random weights with the shapes of the trained network (1 -> 16 -> 32 -> 64 -> 1)."""
    rng = np.random.default_rng(seed)
    shapes = {
        "conv1_w": (16, in_channels, 3, 3), "conv1_b": (16,),
        "conv2_w": (32, 16, 3, 3), "conv2_b": (32,),
        "fc1_w": (64, 32 * 4 * 4), "fc1_b": (64,),
        "fc2_w": (1, 64), "fc2_b": (1,),
    }
    return {k: rng.normal(0.0, 0.1, size=s).astype(np.float32) for k, s in shapes.items()}


def random_matrices(n: int, seed: int) -> np.ndarray:
    """Row-normalized random transition matrices of shape (n, K, K)."""
    return np.random.default_rng(seed).dirichlet(np.ones(K), size=(n, K)).astype(np.float32)


def test_conv_matches_naive_loops() -> None:
    """The vectorized 3x3 convolution equals a direct nested-loop implementation."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 5, 5)).astype(np.float32)
    w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    b = rng.normal(size=4).astype(np.float32)
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    expected = np.zeros((2, 4, 5, 5), dtype=np.float32)
    for n in range(2):
        for o in range(4):
            for i in range(5):
                for j in range(5):
                    expected[n, o, i, j] = (padded[n, :, i : i + 3, j : j + 3] * w[o]).sum() + b[o]
    assert np.allclose(_conv3x3(x, w, b), expected, atol=1e-5)


def test_pooling_shapes_and_values() -> None:
    """Max pool halves 12 -> 6; adaptive average pool maps 6 -> 4 with torch's overlapping windows."""
    x = np.arange(2 * 1 * 6 * 6, dtype=np.float32).reshape(2, 1, 6, 6)
    assert _max_pool2(np.zeros((1, 1, 12, 12), dtype=np.float32)).shape == (1, 1, 6, 6)
    pooled = _adaptive_avg_pool(x, 4)
    assert pooled.shape == (2, 1, 4, 4)
    # Window 0 covers rows/cols 0..1 (ceil(1.5) = 2); window 1 covers 1..2 (floor(1.5) = 1, ceil(3) = 3).
    assert pooled[0, 0, 0, 0] == pytest.approx(x[0, 0, 0:2, 0:2].mean())
    assert pooled[0, 0, 1, 1] == pytest.approx(x[0, 0, 1:3, 1:3].mean())


def test_probabilities_in_range_and_deterministic() -> None:
    """predict_proba returns one probability in [0, 1] per matrix, the same on every call."""
    clf = NumpyCnnClassifier(random_weights(), n_states=K)
    matrices = random_matrices(8, 1)
    probs = clf.predict_proba(matrices)
    assert probs.shape == (8,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs, clf.predict_proba(matrices))
    assert clf.predict_one(matrices[0]) == pytest.approx(probs[0], abs=1e-6)


def test_load_round_trip_and_bad_inputs(tmp_path: Path) -> None:
    """Weights saved as .npz load back identically; wrong shapes or channel counts raise ValueError."""
    weights = random_weights()
    path = tmp_path / "ai_detector_weights.npz"
    np.savez(path, **weights, n_states=np.int64(K), pool_size=np.int64(4))
    loaded = NumpyCnnClassifier.load(path)
    original = NumpyCnnClassifier(weights, n_states=K)
    matrices = random_matrices(3, 3)
    assert np.allclose(loaded.predict_proba(matrices), original.predict_proba(matrices))
    with pytest.raises(ValueError, match="Expected"):
        loaded.predict_logits(np.zeros((1, 1, 5, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="Missing CNN weights"):
        NumpyCnnClassifier({"conv1_w": weights["conv1_w"]}, n_states=K)
    with pytest.raises(ValueError, match="1-channel"):
        NumpyCnnClassifier(random_weights(in_channels=3), n_states=K)


def test_input_matrix_is_unsmoothed() -> None:
    """Inference builds unsmoothed matrices, so a state that never occurs stays an all-zero row (as in training)."""
    seq = np.array([0, 3, 4, 3, 8], dtype=np.int32)
    matrix = sequence_to_transition_matrix(seq, num_states=K)
    assert matrix[11].sum() == 0.0
    assert matrix[3].sum() == pytest.approx(1.0)
