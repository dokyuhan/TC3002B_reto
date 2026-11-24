"""PyTorch training loop, early stopping, and evaluation helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model_training.network.checkpoints import save_checkpoint
from model_training.training.metrics import compute_metrics


@dataclass
class TrainConfig:
    """Training hyperparameters (`patience=None` disables early stopping)."""

    batch_size: int
    lr: float
    epochs: int
    patience: int | None
    min_delta: float
    weight_decay: float
    seed: int


class EarlyStopping:
    """Stop training when validation loss fails to improve for `patience` epochs."""

    def __init__(self, patience: int | None = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        """Record validation loss for an epoch and return True on improvement."""
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Return True once `patience` consecutive epochs pass without improvement."""
        return self.patience is not None and self.counter >= self.patience


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    symmetric: bool = False,
) -> tuple[float | None, np.ndarray, np.ndarray]:
    """Run eval mode forward pass and return `(mean_loss_or_None, probabilities, labels)`."""
    model.eval()
    probs, ys, total = [], [], 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        if criterion is not None:
            total += criterion(logits, y).item() * len(y)
        p_ab = torch.sigmoid(logits)
        if symmetric and X.ndim == 4 and X.shape[1] == 3:
            swapped = torch.stack([X[:, 1], X[:, 0], -X[:, 2]], dim=1)
            p_ba = torch.sigmoid(model(swapped))
            batch_probs = 0.5 * (p_ab + p_ba)
        else:
            batch_probs = p_ab
        probs.append(batch_probs.cpu().numpy())
        ys.append(y.cpu().numpy())
    probs_arr = np.concatenate(probs)
    ys_arr = np.concatenate(ys)
    return (total / len(ys_arr) if criterion is not None else None), probs_arr, ys_arr


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
) -> tuple[list[dict], int]:
    """Train with early stopping, restore best weights, and return `(history, best_epoch)`."""
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopping(cfg.patience, cfg.min_delta)
    history: list[dict] = []
    best_state = None
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        tr_loss, tr_p, tr_y = run_inference(model, train_loader, device, criterion)
        va_loss, va_p, va_y = run_inference(model, val_loader, device, criterion)
        tr = compute_metrics(tr_y, tr_p)
        va = compute_metrics(va_y, va_p)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(tr_loss),
                "val_loss": float(va_loss),
                "train_acc": tr["accuracy"],
                "val_acc": va["accuracy"],
                "train_f1": tr["f1"],
                "val_f1": va["f1"],
                "val_pr_auc": va["pr_auc"],
            }
        )
        improved = stopper.step(va_loss)
        if improved:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    threshold=0.5,
                    extra={"epoch": epoch, "val_loss": float(va_loss)},
                )
        h = history[-1]
        print(
            f"Epoch {epoch:3d} | train_loss {h['train_loss']:.4f} val_loss {h['val_loss']:.4f} | "
            f"train_acc {h['train_acc']:.3f} val_acc {h['val_acc']:.3f} val_f1 {h['val_f1']:.3f}"
            f"{'  *' if improved else ''}"
        )
        if stopper.should_stop:
            print(f"Early stopping after {cfg.patience} epochs without improvement (best: {best_epoch}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch
