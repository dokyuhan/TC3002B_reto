"""Pruebas unitarias y de contrato para Markov y extracción de features."""

from pathlib import Path
import numpy as np
import pytest

from src.features.extraction import run_pipeline, validate_tensor_batch
from src.features.markov import pair_to_tensor, sequence_to_transition_matrix
from tests.mock_step1 import generate_mock_dataset


def test_exact_transition_matrix_and_padding():
    # Transiciones válidas: 0->1, 1->2, 2->1, 1->2 (ignorando padding -1)
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
    seq = [0, 1]
    P = sequence_to_transition_matrix(seq, num_states=5, smoothing_alpha=1.0)
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


def test_pair_identity_and_symmetry():
    seq_a = [0, 3, 5, 7, 8, 3, 5]
    seq_b = [1, 3, 4, 7, 8, 2]

    t_same = pair_to_tensor(seq_a, seq_a, num_states=12)
    assert t_same.shape == (3, 12, 12)
    assert np.allclose(t_same[2], 0.0)

    t_ab = pair_to_tensor(seq_a, seq_b, num_states=12)
    t_ba = pair_to_tensor(seq_b, seq_a, num_states=12)
    assert np.allclose(t_ab[0], t_ba[1])
    assert np.allclose(t_ab[1], t_ba[0])
    assert np.allclose(t_ab[2], t_ba[2])


@pytest.mark.parametrize("storage_format", ["offsets", "padded"])
def test_end_to_end_pipeline(tmp_path: Path, storage_format: str):
    step1_file = tmp_path / f"mock_{storage_format}.npz"
    step2_file = tmp_path / f"tensors_{storage_format}.npz"

    generate_mock_dataset(
        output_path=step1_file,
        storage_format=storage_format,
        num_states=12,
        n_train=20,
        n_val=10,
        n_test=10,
        seed=7,
    )

    summary = run_pipeline(
        input_path=step1_file,
        output_path=step2_file,
        augment_symmetry_train=True,
    )

    assert summary["train"] == (40, 3, 12, 12)
    assert summary["val"] == (10, 3, 12, 12)
    assert summary["test"] == (10, 3, 12, 12)

    with np.load(step2_file) as out:
        for split in ("train", "val", "test"):
            validate_tensor_batch(out[f"X_{split}"], out[f"y_{split}"], num_states=12)
