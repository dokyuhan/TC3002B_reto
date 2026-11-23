from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import PlagiarismDataset
from src.data.features import validate_features

def _load_npz(path: Path) -> dict[str, np.ndarray]:
    """Lee un .npz con las claves X, y y (opcional) group; falla si falta X o y."""
    with np.load(path) as f:
        missing = {"X", "y"} - set(f.files)
        if missing:
            raise KeyError(f"{path} no contiene las claves {sorted(missing)} (tiene {f.files})")
        data = {"X": f["X"], "y": f["y"]}
        if "group" in f.files:
            data["group"] = f["group"]
    return data

_LOADERS = {".npz": _load_npz}

def load_split(path: str | Path) -> dict[str, np.ndarray]:
    """Carga y valida una partición; devuelve X (float32), y (int64) y opcionalmente group."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    loader = _LOADERS.get(path.suffix)
    if loader is None:
        raise ValueError(f"Formato no soportado: '{path.suffix}' (soportados: {sorted(_LOADERS)})")
    data = loader(path)

    X, y = data["X"], np.asarray(data["y"])
    validate_features(X)
    if y.shape != (len(X),):
        raise ValueError(f"y debe tener forma ({len(X)},), se recibió {y.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y debe contener solo 0 y 1")
    data["X"] = X.astype(np.float32)
    data["y"] = y.astype(np.int64)
    return data

def get_data_info(X: np.ndarray) -> dict[str, int]:
    """Devuelve el número de canales y de estados (K) leídos de la forma de X."""
    return {"in_channels": int(X.shape[1]), "n_states": int(X.shape[2])}

def make_dataloader(data: dict[str, np.ndarray], batch_size: int = 64, shuffle: bool = False, seed: int | None = None) -> DataLoader:
    """Crea un DataLoader a partir del dict de `load_split`; con `seed` el orden aleatorio es reproducible."""
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(PlagiarismDataset(data["X"], data["y"]), batch_size=batch_size, shuffle=shuffle, generator=generator)
