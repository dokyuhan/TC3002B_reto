"""Unit and pipeline tests for Markov transition features."""

from pathlib import Path
import numpy as np
import pytest

from src.features.extraction import run_pipeline
from src.features.markov import pair_to_tensor, sequence_to_transition_matrix
from src.preprocessing.normalizer import PAD_TOKEN


def _write_test_archive(path: Path, storage_format: str, num_states: int = 12) -> None:
    """Write a minimal deterministic .npz archive in padded or offsets format."""
    splits = {
        "train": ([[0, 1, 2, 3], [4, 5, 6, 7, 8]], [[0, 1, 2, 4], [1, 2, 3]], [1, 0]),
        "val": ([[0, 2, 4, 6]], [[1, 3, 5, 7]], [0]),
        "test": ([[3, 5, 7, 9]], [[3, 5, 7, 9]], [1]),
    }
    payload: dict[str, np.ndarray] = {"num_states": np.asarray(num_states, dtype=np.int32)}

    for split, (seqs_a, seqs_b, labels) in splits.items():
        payload[f"{split}_y"] = np.asarray(labels, dtype=np.uint8)
        if storage_format == "padded":
            for side, seqs in (("a", seqs_a), ("b", seqs_b)):
                max_len = max(len(s) for s in seqs)
                arr = np.full((len(seqs), max_len), PAD_TOKEN, dtype=np.int32)
                for i, s in enumerate(seqs):
                    arr[i, : len(s)] = s
                payload[f"{split}_seq_{side}"] = arr
        else:
            for side, seqs in (("a", seqs_a), ("b", seqs_b)):
                lengths = np.asarray([len(s) for s in seqs], dtype=np.int64)
                offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
                offsets[1:] = np.cumsum(lengths)
                payload[f"{split}_tokens_{side}"] = np.concatenate(seqs).astype(np.int32)
                payload[f"{split}_offsets_{side}"] = offsets

    np.savez_compressed(path, **payload)


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
    input_file = tmp_path / f"input_{storage_format}.npz"
    output_file = tmp_path / f"tensors_{storage_format}.npz"
    _write_test_archive(input_file, storage_format=storage_format, num_states=12)

    summary = run_pipeline(
        input_path=input_file,
        output_path=output_file,
        augment_symmetry_train=True,
    )

    assert summary["train"] == (4, 3, 12, 12)
    assert summary["val"] == (1, 3, 12, 12)
    assert summary["test"] == (1, 3, 12, 12)

    with np.load(output_file) as out:
        for split in ("train", "val", "test"):
            X = out[f"X_{split}"]
            y = out[f"y_{split}"]
            assert X.dtype == np.float32
            assert X.shape[0] == y.shape[0]
            assert np.isfinite(X).all()
            assert np.allclose(X[:, 2], np.abs(X[:, 0] - X[:, 1]), atol=1e-6)
            for ch in (0, 1):
                row_sums = X[:, ch].sum(axis=-1)
                assert (np.isclose(row_sums, 1.0, atol=1e-5) | np.isclose(row_sums, 0.0, atol=1e-5)).all()
