"""Unit tests for the Markov transition matrix of one AST state sequence."""

import numpy as np
import pytest

from src.features.markov import sequence_to_transition_matrix


def test_exact_transition_matrix_and_padding():
    seq = [0, 1, 2, 1, 2, -1, -1]
    P = sequence_to_transition_matrix(seq, num_states=4, smoothing_alpha=0.0)

    expected = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert P.dtype == np.float32
    assert np.allclose(P, expected)


def test_laplace_smoothing_produces_stochastic_rows():
    P = sequence_to_transition_matrix([0, 1], num_states=5, smoothing_alpha=1.0)
    assert np.allclose(P.sum(axis=1), np.ones(5, dtype=np.float32), atol=1e-6)
    assert np.isfinite(P).all()


def test_degenerate_sequences_no_nan():
    for seq in ([], [3], [-1, -1]):
        P = sequence_to_transition_matrix(seq, num_states=4, smoothing_alpha=0.0)
        assert np.all(P == 0.0)
        assert np.isfinite(P).all()


def test_out_of_range_state_raises():
    with pytest.raises(ValueError):
        sequence_to_transition_matrix([0, 1, 12], num_states=12)
