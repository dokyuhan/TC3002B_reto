from pathlib import Path
import numpy as np

SPLITS = ("train", "val", "test")

def transition_matrix(tokens: np.ndarray, n_states: int) -> np.ndarray:
    """Matriz (K, K) de transiciones consecutivas, normalizada por filas."""
    tokens = np.asarray(tokens, dtype=np.int64)
    if len(tokens) and (tokens.min() < 0 or tokens.max() >= n_states):
        raise ValueError(f"Estados fuera de [0, {n_states}): min={tokens.min()}, max={tokens.max()}")
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    if len(tokens) > 1:
        np.add.at(counts, (tokens[:-1], tokens[1:]), 1.0)
    rows = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0).astype(np.float32)

def _matrices(tokens: np.ndarray, offsets: np.ndarray, n_states: int) -> np.ndarray:
    """Corta el arreglo plano de tokens con `offsets` y devuelve una matriz por programa, forma (N, K, K)."""
    return np.stack([transition_matrix(tokens[s:e], n_states)
                     for s, e in zip(offsets[:-1], offsets[1:])])

def build_split(npz: "np.lib.npyio.NpzFile", split: str) -> dict[str, np.ndarray]:
    """Construye X (N, 3, K, K), y y group de una partición; canales: 0 = programa A, 1 = programa B, 2 = A - B."""
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
