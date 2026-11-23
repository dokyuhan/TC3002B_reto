import numpy as np
import torch
from torch.utils.data import Dataset

class PlagiarismDataset(Dataset):
    """Dataset de PyTorch con X (N, C, K, K) e y (N,) como tensores float32 (MPS no soporta float64)."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Convierte X e y a tensores float32 y verifica que tengan la misma longitud."""
        if len(X) != len(y):
            raise ValueError(f"X e y tienen distinta longitud: {len(X)} vs {len(y)}")
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        """Número de muestras."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve la pareja (matrices, etiqueta) de la muestra `idx`."""
        return self.X[idx], self.y[idx]
