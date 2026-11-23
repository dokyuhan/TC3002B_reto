from pathlib import Path
import numpy as np
import torch
from src.data.features import validate_features
from src.model.checkpoints import load_checkpoint
from src.utils import get_device

class Predictor:
    """Modelo entrenado listo para clasificar; usa el umbral del checkpoint salvo que se indique otro."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto", threshold: float | None = None) -> None:
        """Carga el checkpoint y fija el umbral (debe estar en (0, 1))."""
        self.device = get_device(device)
        self.model, ckpt = load_checkpoint(checkpoint_path, self.device)
        self.config = ckpt["model_config"]
        self.input_spec = ckpt["extra"].get("input_spec")   # orden de estados y normalización del entrenamiento
        self.threshold = float(ckpt["threshold"] if threshold is None else threshold)
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"El umbral debe estar en (0, 1), se recibió {self.threshold}")

    def _prepare(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Convierte la entrada a float32 (N, C, K, K), la valida y revisa que coincida con el modelo."""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:                      # un solo ejemplo (C, K, K)
            X = X[None]
        validate_features(X)
        expected = (self.config["in_channels"], self.config["n_states"], self.config["n_states"])
        if X.shape[1:] != expected:
            raise ValueError(f"Forma de entrada {X.shape[1:]} incompatible con el modelo {expected}")
        return X

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray | torch.Tensor, batch_size: int = 256) -> np.ndarray:
        """Devuelve la probabilidad de la clase positiva para cada muestra, forma (N,)."""
        X = self._prepare(X)
        out = []
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(self.device)
            out.append(torch.sigmoid(self.model(batch)).cpu().numpy())
        return np.concatenate(out)

    def predict(self, X: np.ndarray | torch.Tensor, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (probabilidades, etiquetas 0/1 según el umbral)."""
        probs = self.predict_proba(X, batch_size)
        return probs, (probs >= self.threshold).astype(int)

def predict(X: np.ndarray | torch.Tensor, checkpoint_path: str | Path, device: str = "auto", threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Atajo de un solo paso: carga el modelo (en cada llamada) y predice."""
    return Predictor(checkpoint_path, device, threshold).predict(X)
