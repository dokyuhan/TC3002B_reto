"""Generador sintético del entregable del Paso 1 para desarrollo en paralelo."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .schema import DEFAULT_AST_STATES, NUM_STATES, PAD_TOKEN


def _sample_markov_sequence(
    rng: np.random.Generator,
    transition_matrix: np.ndarray,
    length: int,
) -> np.ndarray:
    num_states = transition_matrix.shape[0]
    seq = np.empty(length, dtype=np.int32)
    seq[0] = rng.integers(0, num_states)
    for t in range(1, length):
        seq[t] = rng.choice(num_states, p=transition_matrix[seq[t - 1]])
    return seq


def _perturb_sequence(
    rng: np.random.Generator,
    seq: np.ndarray,
    num_states: int,
    edit_rate: float = 0.15,
) -> np.ndarray:
    """Simula refactorización (sustituciones, inserciones y eliminaciones leves)."""
    tokens = seq.tolist()
    n_edits = max(1, int(len(tokens) * edit_rate))
    for _ in range(n_edits):
        op = rng.choice(["sub", "ins", "del"])
        if op == "sub" and tokens:
            idx = int(rng.integers(0, len(tokens)))
            tokens[idx] = int(rng.integers(0, num_states))
        elif op == "ins":
            idx = int(rng.integers(0, len(tokens) + 1))
            tokens.insert(idx, int(rng.integers(0, num_states)))
        elif op == "del" and len(tokens) > 4:
            idx = int(rng.integers(0, len(tokens)))
            tokens.pop(idx)
    return np.asarray(tokens, dtype=np.int32)


def _generate_split(
    rng: np.random.Generator,
    n_pairs: int,
    num_states: int,
    min_len: int = 25,
    max_len: int = 120,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    seqs_a: list[np.ndarray] = []
    seqs_b: list[np.ndarray] = []
    y = np.zeros(n_pairs, dtype=np.uint8)

    for i in range(n_pairs):
        is_plagiarism = i % 2 == 1
        y[i] = 1 if is_plagiarism else 0

        # Distribuciones dispersas (Dirichlet con concentración baja) simulan programas distintos
        mat_a = rng.dirichlet(np.full(num_states, 0.35), size=num_states)
        len_a = int(rng.integers(min_len, max_len + 1))
        seq_a = _sample_markov_sequence(rng, mat_a, len_a)

        if is_plagiarism:
            seq_b = _perturb_sequence(rng, seq_a, num_states=num_states, edit_rate=0.15)
        else:
            mat_b = rng.dirichlet(np.full(num_states, 0.35), size=num_states)
            len_b = int(rng.integers(min_len, max_len + 1))
            seq_b = _sample_markov_sequence(rng, mat_b, len_b)

        seqs_a.append(seq_a)
        seqs_b.append(seq_b)

    perm = rng.permutation(n_pairs)
    return [seqs_a[i] for i in perm], [seqs_b[i] for i in perm], y[perm]


def _pack_padded(seqs: list[np.ndarray], pad_token: int = PAD_TOKEN) -> np.ndarray:
    max_len = max(len(s) for s in seqs)
    arr = np.full((len(seqs), max_len), pad_token, dtype=np.int32)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return arr


def _pack_offsets(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(s) for s in seqs], dtype=np.int64)
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    tokens = np.concatenate(seqs).astype(np.int32)
    return tokens, offsets


def generate_mock_dataset(
    output_path: Path,
    storage_format: str = "offsets",
    num_states: int = NUM_STATES,
    n_train: int = 600,
    n_val: int = 150,
    n_test: int = 150,
    seed: int = 42,
) -> None:
    if storage_format not in ("padded", "offsets"):
        raise ValueError("storage_format debe ser 'padded' o 'offsets'.")

    rng = np.random.default_rng(seed)
    payload: dict[str, np.ndarray] = {
        "num_states": np.asarray(num_states, dtype=np.int32),
        "state_names": np.asarray(DEFAULT_AST_STATES[:num_states]),
    }

    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        seqs_a, seqs_b, y = _generate_split(rng, count, num_states=num_states)
        payload[f"{split}_y"] = y
        if storage_format == "padded":
            payload[f"{split}_seq_a"] = _pack_padded(seqs_a)
            payload[f"{split}_seq_b"] = _pack_padded(seqs_b)
        else:
            tok_a, off_a = _pack_offsets(seqs_a)
            tok_b, off_b = _pack_offsets(seqs_b)
            payload[f"{split}_tokens_a"] = tok_a
            payload[f"{split}_offsets_a"] = off_a
            payload[f"{split}_tokens_b"] = tok_b
            payload[f"{split}_offsets_b"] = off_b

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un .npz mock del Paso 1")
    parser.add_argument("--output", type=Path, default=Path("data/mock_step1.npz"))
    parser.add_argument("--format", choices=("padded", "offsets"), default="offsets")
    parser.add_argument("--num-states", type=int, default=NUM_STATES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_mock_dataset(
        output_path=args.output,
        storage_format=args.format,
        num_states=args.num_states,
        seed=args.seed,
    )
    print(f"Dataset mock ({args.format}) guardado en {args.output}")


if __name__ == "__main__":
    main()