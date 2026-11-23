"""Paso Markov de la CNN: secuencias AST (dataset/preprocessed) -> matrices (dataset/processed).

Cada programa pasa de su secuencia de estados AST a una matriz de transición K x K (la misma función
que usa la app: `src.features.markov`, sin suavizado). Cada muestra de la CNN tiene 3 canales:
0 = programa A, 1 = programa B, 2 = A - B (con signo).
"""
from pathlib import Path
import numpy as np

from src.features.markov import sequence_to_transition_matrix

SPLITS = ("train", "val", "test")

# smoothing_alpha=0.0 es explícito a propósito: la CNN se entrenó sin suavizado y no debe depender del valor por defecto de src.
def _matrices(tokens: np.ndarray, offsets: np.ndarray, n_states: int) -> np.ndarray:
    """Corta el arreglo plano de tokens con `offsets` y devuelve una matriz por programa, forma (N, K, K)."""
    return np.stack([sequence_to_transition_matrix(tokens[s:e], num_states=n_states, smoothing_alpha=0.0)
                     for s, e in zip(offsets[:-1], offsets[1:])])

def build_split(npz: "np.lib.npyio.NpzFile", split: str) -> dict[str, np.ndarray]:
    """Construye X (N, 3, K, K), y y group de una partición; canales: A, B y A - B."""
    k = int(npz["num_states"])
    a = _matrices(npz[f"{split}_tokens_a"], npz[f"{split}_offsets_a"], k)
    b = _matrices(npz[f"{split}_tokens_b"], npz[f"{split}_offsets_b"], k)
    return {"X": np.stack([a, b, a - b], axis=1),
            "y": npz[f"{split}_y"].astype(np.int64),
            "group": npz[f"{split}_group_ids"]}

def convert(sequences_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Convierte un archivo *_sequences.npz en out_dir/{train,val,test}.npz y devuelve las rutas."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    with np.load(sequences_path) as npz:
        for split in SPLITS:
            if f"{split}_tokens_a" not in npz.files:
                raise KeyError(f"{sequences_path} no tiene la partición '{split}' (tiene {npz.files[:6]}...)")
            paths[split] = out_dir / f"{split}.npz"
            np.savez(paths[split], **build_split(npz, split))
    return paths

def convert_diagnostic(sequences_path: str | Path, out_path: str | Path) -> Path:
    """Convierte el split 'diagnostic' (IR Plag, solo evaluación) en un único .npz con X, y, group y levels."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(sequences_path) as npz:
        data = build_split(npz, "diagnostic")
        data["levels"] = npz["diagnostic_levels"]
    np.savez(out_path, **data)
    return out_path
