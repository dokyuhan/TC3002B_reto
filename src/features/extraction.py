"""Batch feature extraction from preprocessed .npz archives to Markov tensors."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from src.features.markov import clean_sequence, pair_to_tensor
from src.vocabulary import DEFAULT_AST_STATES, NUM_STATES, PAD_TOKEN

SPLITS: tuple[str, ...] = ("train", "val", "test")


def _extract_split_pairs(
    data: np.lib.npyio.NpzFile, split: str
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Extract (seq_a, seq_b, y) for a split from padded or offset-packed arrays."""
    y_key = f"{split}_y"
    if y_key not in data:
        raise KeyError(f"Missing required key '{y_key}' in input archive.")
    y = np.asarray(data[y_key], dtype=np.uint8)

    padded_keys = (f"{split}_seq_a", f"{split}_seq_b")
    offset_keys = (
        f"{split}_tokens_a",
        f"{split}_offsets_a",
        f"{split}_tokens_b",
        f"{split}_offsets_b",
    )

    if all(k in data for k in padded_keys):
        seq_a_arr, seq_b_arr = data[padded_keys[0]], data[padded_keys[1]]
        return [seq_a_arr[i] for i in range(len(y))], [seq_b_arr[i] for i in range(len(y))], y

    if all(k in data for k in offset_keys):
        tok_a, off_a = data[offset_keys[0]], data[offset_keys[1]]
        tok_b, off_b = data[offset_keys[2]], data[offset_keys[3]]
        seqs_a = [tok_a[off_a[i] : off_a[i + 1]] for i in range(len(y))]
        seqs_b = [tok_b[off_b[i] : off_b[i + 1]] for i in range(len(y))]
        return seqs_a, seqs_b, y

    raise KeyError(f"Unrecognized format for split '{split}'; expected {padded_keys} or {offset_keys}.")


def build_split_tensors(
    seqs_a: list[np.ndarray],
    seqs_b: list[np.ndarray],
    y: np.ndarray,
    num_states: int = NUM_STATES,
    pad_token: int = PAD_TOKEN,
    smoothing_alpha: float = 0.0,
    drop_degenerate: bool = False,
    augment_symmetry: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a split of sequence pairs into (N, 3, K, K) float32 tensors and uint8 labels."""
    tensors: list[np.ndarray] = []
    labels: list[int] = []

    for seq_a, seq_b, label in zip(seqs_a, seqs_b, y):
        if drop_degenerate:
            clean_a = clean_sequence(seq_a, num_states=num_states, pad_token=pad_token)
            clean_b = clean_sequence(seq_b, num_states=num_states, pad_token=pad_token)
            if clean_a.size < 2 or clean_b.size < 2:
                continue

        tensors.append(
            pair_to_tensor(
                seq_a,
                seq_b,
                num_states=num_states,
                pad_token=pad_token,
                smoothing_alpha=smoothing_alpha,
            )
        )
        labels.append(int(label))

    if not tensors:
        return (
            np.empty((0, 3, num_states, num_states), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
        )

    X = np.stack(tensors, axis=0).astype(np.float32)
    y_out = np.asarray(labels, dtype=np.uint8)

    if augment_symmetry:
        # Channel 2 (|P_A - P_B|) is order-invariant, so only P_A and P_B swap.
        X = np.concatenate([X, X[:, [1, 0, 2], :, :]], axis=0)
        y_out = np.concatenate([y_out, y_out], axis=0)

    return X, y_out


def run_pipeline(
    input_path: Path,
    output_path: Path,
    num_states: int = NUM_STATES,
    pad_token: int = PAD_TOKEN,
    smoothing_alpha: float = 0.0,
    drop_degenerate: bool = False,
    augment_symmetry_train: bool = False,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, tuple[int, ...]]:
    """Transform preprocessed splits into Markov tensor archives and return output shapes."""
    payload: dict[str, np.ndarray] = {}
    summary: dict[str, tuple[int, ...]] = {}

    with np.load(input_path, allow_pickle=False) as data:
        if "num_states" in data:
            num_states = int(data["num_states"])
        state_names = (
            data["state_names"]
            if "state_names" in data
            else np.asarray(DEFAULT_AST_STATES[:num_states])
        )

        for split in splits:
            seqs_a, seqs_b, y = _extract_split_pairs(data, split)
            X_split, y_split = build_split_tensors(
                seqs_a,
                seqs_b,
                y,
                num_states=num_states,
                pad_token=pad_token,
                smoothing_alpha=smoothing_alpha,
                drop_degenerate=drop_degenerate,
                augment_symmetry=(augment_symmetry_train and split == "train"),
            )
            payload[f"X_{split}"] = X_split
            payload[f"y_{split}"] = y_split
            summary[split] = X_split.shape

    payload["num_states"] = np.asarray(num_states, dtype=np.int32)
    payload["smoothing_alpha"] = np.asarray(smoothing_alpha, dtype=np.float32)
    payload["state_names"] = state_names

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return summary


def build_diagnostic_tensors(
    input_path: Path,
    output_path: Path | None = None,
    split: str = "diagnostic",
    num_states: int = NUM_STATES,
    pad_token: int = PAD_TOKEN,
    smoothing_alpha: float = 0.0,
    drop_degenerate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract Markov tensors for a single diagnostic split and optionally save to .npz."""
    with np.load(input_path, allow_pickle=False) as data:
        if "num_states" in data:
            num_states = int(data["num_states"])
        state_names = (
            data["state_names"]
            if "state_names" in data
            else np.asarray(DEFAULT_AST_STATES[:num_states])
        )
        seqs_a, seqs_b, y = _extract_split_pairs(data, split)

    X_diag, y_diag = build_split_tensors(
        seqs_a,
        seqs_b,
        y,
        num_states=num_states,
        pad_token=pad_token,
        smoothing_alpha=smoothing_alpha,
        drop_degenerate=drop_degenerate,
        augment_symmetry=False,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            **{
                f"X_{split}": X_diag,
                f"y_{split}": y_diag,
                "num_states": np.asarray(num_states, dtype=np.int32),
                "smoothing_alpha": np.asarray(smoothing_alpha, dtype=np.float32),
                "state_names": state_names,
            },
        )

    return X_diag, y_diag
