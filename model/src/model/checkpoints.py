import json
from pathlib import Path
from typing import Any
import torch

from src.model.cnn import PlagiarismCNN

CHECKPOINT_VERSION = 1

def _plain(obj: Any) -> Any:
    """Fuerza tipos simples de Python (pasa por JSON); falla si algo no es serializable, p. ej. np.float32."""
    return json.loads(json.dumps(obj))

def save_checkpoint(path: str | Path, model: PlagiarismCNN, threshold: float = 0.5, extra: dict | None = None) -> None:
    """Guarda pesos, configuración del modelo, umbral y datos extra solo con tipos simples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": CHECKPOINT_VERSION,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": _plain(model.config),
        "threshold": float(threshold),
        "extra": _plain(extra or {}),
    }, path)

def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[PlagiarismCNN, dict]:
    """Reconstruye el modelo en modo eval y devuelve (modelo, diccionario del checkpoint)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PlagiarismCNN.from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
