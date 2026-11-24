"""Build 3 channel unsmoothed Markov tensors for CNN training and evaluation.

Each program sequence is mapped via `src.features.markov.sequence_to_transition_matrix`
with `smoothing_alpha=0.0`. Channels are `[P_A, P_B, P_A - P_B]`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.markov import sequence_to_transition_matrix

SPLITS = ("train", "val", "test")


def _matrices(tokens: np.ndarray, offsets: np.ndarray, n_states: int) -> np.ndarray:
    """Slice offset packed token sequences into `(N, K, K)` unsmoothed transition matrices."""
    return np.stack(
        [
            sequence_to_transition_matrix(
                tokens[s:e],
                num_states=n_states,
                smoothing_alpha=0.0,
            )
            for s, e in zip(offsets[:-1], offsets[1:])
        ]
    )


def build_split(npz: np.lib.npyio.NpzFile, split: str) -> dict[str, np.ndarray]:
    """Build `X` `(N, 3, K, K)`, `y`, and `group` arrays for a single split."""
    k = int(npz["num_states"])
    a = _matrices(npz[f"{split}_tokens_a"], npz[f"{split}_offsets_a"], k)
    b = _matrices(npz[f"{split}_tokens_b"], npz[f"{split}_offsets_b"], k)
    return {
        "X": np.stack([a, b, a - b], axis=1),
        "y": npz[f"{split}_y"].astype(np.int64),
        "group": npz[f"{split}_group_ids"],
    }


def convert(sequences_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Convert a preprocessed `*_sequences.npz` archive into `train`, `val`, and `test` splits."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    with np.load(sequences_path) as npz:
        for split in SPLITS:
            if f"{split}_tokens_a" not in npz.files:
                raise KeyError(
                    f"{sequences_path} is missing split '{split}' (found {npz.files[:6]}...)"
                )
            paths[split] = out_dir / f"{split}.npz"
            np.savez(paths[split], **build_split(npz, split))
    return paths


def convert_diagnostic(sequences_path: str | Path, out_path: str | Path) -> Path:
    """Convert the IR Plag `diagnostic` split into an evaluation `.npz` with `levels`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(sequences_path) as npz:
        data = build_split(npz, "diagnostic")
        data["levels"] = npz["diagnostic_levels"]
    np.savez(out_path, **data)
    return out_path
