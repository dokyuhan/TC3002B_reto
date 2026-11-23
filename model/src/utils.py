import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Fija la semilla de Python, NumPy y PyTorch para que los resultados sean reproducibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(name: str = "auto") -> torch.device:
    """Elige el dispositivo de cómputo: 'auto' usa CUDA, luego MPS y por último CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
