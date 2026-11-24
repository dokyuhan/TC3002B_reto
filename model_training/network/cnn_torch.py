import torch
from torch import nn

class AdaptiveAvgPool2dMPS(nn.AdaptiveAvgPool2d):
    """AdaptiveAvgPool2d que, en MPS con tamaños no divisibles, hace el pooling en CPU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica el pooling adaptativo; pasa por CPU solo si MPS no soporta ese tamaño."""
        out = self.output_size
        out = (out, out) if isinstance(out, int) else out
        if x.device.type == "mps" and (x.shape[-2] % out[0] or x.shape[-1] % out[1]):
            return super().forward(x.cpu()).to(x.device)
        return super().forward(x)

class AiCodeCNN(nn.Module):
    """CNN pequeña (2 convoluciones + 2 capas lineales) que devuelve un logit por muestra."""

    def __init__(self, in_channels: int = 1, n_states: int = 12, conv_channels: tuple[int, int] = (32, 64),
                 hidden_dim: int = 64, dropout: float = 0.3, pool_size: int = 4) -> None:
        """Construye las capas y guarda la configuración en `self.config` para poder reconstruir el modelo."""
        super().__init__()
        c1, c2 = conv_channels
        self.config = {
            "in_channels": int(in_channels), "n_states": int(n_states),
            "conv_channels": [int(c1), int(c2)], "hidden_dim": int(hidden_dim),
            "dropout": float(dropout), "pool_size": int(pool_size),
        }
        use_maxpool = n_states >= 4  # evita reducir matrices diminutas a 1x1
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) if use_maxpool else nn.Identity(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            AdaptiveAvgPool2dMPS(pool_size),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2 * pool_size * pool_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recibe (B, C, K, K) y devuelve logits (B,); aplicar sigmoid solo para obtener probabilidades."""
        return self.classifier(self.features(x)).squeeze(-1)

    @classmethod
    def from_config(cls, config: dict) -> "AiCodeCNN":
        """Crea un modelo nuevo a partir de un diccionario de configuración."""
        return cls(**config)
