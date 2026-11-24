"""PyTorch CNN definition for pairwise Markov transition matrices."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveAvgPool2dMPS(nn.AdaptiveAvgPool2d):
    """AdaptiveAvgPool2d fallback to CPU when MPS receives non divisible input sizes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive pooling, routing through CPU only on unsupported MPS shapes."""
        out = self.output_size
        out = (out, out) if isinstance(out, int) else out
        if x.device.type == "mps" and (x.shape[-2] % out[0] or x.shape[-1] % out[1]):
            return super().forward(x.cpu()).to(x.device)
        return super().forward(x)


class PlagiarismCNN(nn.Module):
    """Compact 2 conv + 2 linear CNN returning one plagiarism logit per sample."""

    def __init__(
        self,
        in_channels: int = 1,
        n_states: int = 12,
        conv_channels: tuple[int, int] = (32, 64),
        hidden_dim: int = 64,
        dropout: float = 0.3,
        pool_size: int = 4,
    ) -> None:
        """Build named layers and record configuration in `self.config`."""
        super().__init__()
        c1, c2 = conv_channels
        self.config = {
            "in_channels": int(in_channels),
            "n_states": int(n_states),
            "conv_channels": [int(c1), int(c2)],
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "pool_size": int(pool_size),
        }
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2) if n_states >= 4 else nn.Identity()
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.adaptive_pool = AdaptiveAvgPool2dMPS(pool_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(c2 * pool_size * pool_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map `(B, C, K, K)` tensors to logits `(B,)`; apply sigmoid for probabilities."""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)

    @classmethod
    def from_config(cls, config: dict) -> PlagiarismCNN:
        """Instantiate a model from a serialized configuration dictionary."""
        return cls(**config)
