"""Pipeline por lotes: .npz de preprocesamiento -> tensores (N, 3, K, K) para modelado."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src.features.markov import clean_sequence, pair_to_tensor
from src.preprocessing.normalizer import DEFAULT_AST_STATES, NUM_STATES, PAD_TOKEN

SPLITS: tuple[str, ...] = ("train", "val", "test")


def _extract_split_pairs(
    data: np.lib.npyio.NpzFile, split: str
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Extrae listas de secuencias (A, B) y etiquetas y soportando padding u offsets."""
    y_key = f"{split}_y"
    if y_key not in data:
        raise KeyError(f"Falta la llave obligatoria '{y_key}' en el archivo de entrada.")
    y = np.asarray(data[y_key], dtype=np.uint8)

    padded_keys = (f"{split}_seq_a", f"{split}_seq_b")
    offset_keys = (
        f"{split}_tokens_a",
        f"{split}_offsets_a",
        f"{split}_tokens_b",
        f"{split}_offsets_b",
    )

    if all(k in data for k in padded_keys):
        seq_a_arr = data[padded_keys[0]]
        seq_b_arr = data[padded_keys[1]]
        seqs_a = [seq_a_arr[i] for i in range(len(y))]
        seqs_b = [seq_b_arr[i] for i in range(len(y))]
        return seqs_a, seqs_b, y

    if all(k in data for k in offset_keys):
        tok_a, off_a = data[offset_keys[0]], data[offset_keys[1]]
        tok_b, off_b = data[offset_keys[2]], data[offset_keys[3]]
        seqs_a = [tok_a[off_a[i] : off_a[i + 1]] for i in range(len(y))]
        seqs_b = [tok_b[off_b[i] : off_b[i + 1]] for i in range(len(y))]
        return seqs_a, seqs_b, y

    raise KeyError(
        f"Formato no reconocido para split '{split}'. "
        f"Se esperaba {padded_keys} o {offset_keys}."
    )


def validate_tensor_batch(
    X: np.ndarray,
    y: np.ndarray,
    num_states: int,
    smoothing_alpha: float = 0.0,
) -> None:
    """Verifica todos los invariantes matemáticos y de forma del contrato extracción -> modelado."""
    if X.ndim != 4 or X.shape[1:] != (3, num_states, num_states):
        raise ValueError(
            f"Forma inválida para X: {X.shape}, esperada (N, 3, {num_states}, {num_states})."
        )
    if X.dtype != np.float32:
        raise ValueError(f"dtype inválido para X: {X.dtype}, esperado float32.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"Desalineación entre X ({X.shape[0]}) e y ({y.shape}).")
    if not np.isfinite(X).all():
        raise ValueError("El tensor X contiene valores NaN o Inf.")
    if (X < 0.0).any() or (X > 1.0 + 1e-6).any():
        raise ValueError("El tensor X contiene probabilidades fuera de [0, 1].")

    expected_diff = np.abs(X[:, 0] - X[:, 1])
    if not np.allclose(X[:, 2], expected_diff, atol=1e-6):
        raise ValueError("El canal 2 no coincide con |P_A - P_B|.")

    for ch in (0, 1):
        row_sums = X[:, ch].sum(axis=-1)
        if smoothing_alpha > 0.0:
            valid_rows = np.isclose(row_sums, 1.0, atol=1e-5)
        else:
            valid_rows = np.isclose(row_sums, 1.0, atol=1e-5) | np.isclose(
                row_sums, 0.0, atol=1e-5
            )
        if not valid_rows.all():
            raise ValueError(f"Filas con suma estocástica inválida en canal {ch}.")


def build_split_tensors(
    seqs_a: list[np.ndarray],
    seqs_b: list[np.ndarray],
    y: np.ndarray,
    num_states: int = NUM_STATES,
    pad_token: int = PAD_TOKEN,
    smoothing_alpha: float = 0.0,
    drop_degenerate: bool = False,
    augment_symmetry: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Transforma un split completo a tensores (N, 3, K, K) y etiquetas alineadas."""
    tensors: list[np.ndarray] = []
    labels: list[int] = []
    dropped = 0

    for seq_a, seq_b, label in zip(seqs_a, seqs_b, y):
        if drop_degenerate:
            clean_a = clean_sequence(seq_a, num_states=num_states, pad_token=pad_token)
            clean_b = clean_sequence(seq_b, num_states=num_states, pad_token=pad_token)
            if clean_a.size < 2 or clean_b.size < 2:
                dropped += 1
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
        X = np.empty((0, 3, num_states, num_states), dtype=np.float32)
        y_out = np.empty((0,), dtype=np.uint8)
        return X, y_out, dropped

    X = np.stack(tensors, axis=0).astype(np.float32)
    y_out = np.asarray(labels, dtype=np.uint8)

    if augment_symmetry:
        # Intercambia canal 0 y 1; el canal 2 (|P_A - P_B|) es invariante.
        X_swapped = X[:, [1, 0, 2], :, :]
        X = np.concatenate([X, X_swapped], axis=0)
        y_out = np.concatenate([y_out, y_out], axis=0)

    validate_tensor_batch(X, y_out, num_states=num_states, smoothing_alpha=smoothing_alpha)
    return X, y_out, dropped


def run_pipeline(
    input_path: Path,
    output_path: Path,
    num_states: int = NUM_STATES,
    pad_token: int = PAD_TOKEN,
    smoothing_alpha: float = 0.0,
    drop_degenerate: bool = False,
    augment_symmetry_train: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Lee el dataset preprocesado, genera los tensores por split y guarda el .npz final."""
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

        for split in SPLITS:
            seqs_a, seqs_b, y = _extract_split_pairs(data, split)
            X_split, y_split, dropped = build_split_tensors(
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
            if dropped > 0:
                print(f"[{split}] Pares descartados por longitud < 2: {dropped}")

    payload["num_states"] = np.asarray(num_states, dtype=np.int32)
    payload["smoothing_alpha"] = np.asarray(smoothing_alpha, dtype=np.float32)
    payload["state_names"] = state_names

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extracción: AST -> Tensores de Markov")
    parser.add_argument("--input", type=Path, required=True, help="Ruta del .npz (preprocesamiento)")
    parser.add_argument("--output", type=Path, required=True, help="Ruta del .npz (features)")
    parser.add_argument("--num-states", type=int, default=NUM_STATES)
    parser.add_argument("--alpha", type=float, default=0.0, help="Suavizado de Laplace (0 = sin suavizado)")
    parser.add_argument("--drop-degenerate", action="store_true")
    parser.add_argument("--augment-symmetry-train", action="store_true")
    args = parser.parse_args()

    summary = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        num_states=args.num_states,
        smoothing_alpha=args.alpha,
        drop_degenerate=args.drop_degenerate,
        augment_symmetry_train=args.augment_symmetry_train,
    )
    for split, shape in summary.items():
        print(f"Split '{split}' listo -> X_{split}.shape = {shape}")


if __name__ == "__main__":
    main()
