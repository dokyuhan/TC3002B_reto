"""Paso Markov de la CNN: secuencias AST (dataset/preprocessed) -> matrices (dataset/processed).

Cada programa (una función Java) pasa de su secuencia de estados AST a UNA matriz de transición K x K
(la misma función que usa la app: `src.features.markov`, sin suavizado). La muestra de la CNN tiene
1 canal: la matriz del programa. La etiqueta es 0 = humano, 1 = IA.
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
    """Construye X (N, 1, K, K), y, group y source de una partición."""
    k = int(npz["num_states"])
    matrices = _matrices(npz[f"{split}_tokens"], npz[f"{split}_offsets"], k)
    return {"X": matrices[:, None],
            "y": npz[f"{split}_y"].astype(np.int64),
            "group": npz[f"{split}_group_ids"],
            "source": npz[f"{split}_sources"]}

def convert(sequences_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Convierte el archivo *_sequences.npz en out_dir/{train,val,test}.npz y devuelve las rutas."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    with np.load(sequences_path) as npz:
        for split in SPLITS:
            if f"{split}_tokens" not in npz.files:
                raise KeyError(f"{sequences_path} no tiene la partición '{split}' (tiene {npz.files[:6]}...)")
            paths[split] = out_dir / f"{split}.npz"
            np.savez(paths[split], **build_split(npz, split))
    return paths

