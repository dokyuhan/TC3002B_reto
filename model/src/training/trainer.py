import copy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import compute_metrics
from src.model.checkpoints import save_checkpoint

@dataclass
class TrainConfig:
    """Hiperparámetros del entrenamiento (`patience=None` desactiva el early stopping); los valores viven en config.py."""
    batch_size: int
    lr: float
    epochs: int
    patience: int | None
    min_delta: float
    weight_decay: float
    seed: int

class EarlyStopping:
    """Detiene el entrenamiento cuando val_loss no mejora en `patience` épocas seguidas."""

    def __init__(self, patience: int | None = 5, min_delta: float = 0.0) -> None:
        """Guarda la paciencia y la mejora mínima, y reinicia el mejor valor y el contador."""
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter = float("inf"), 0

    def step(self, value: float) -> bool:
        """Registra el val_loss de una época; devuelve True si fue una mejora."""
        if value < self.best - self.min_delta:
            self.best, self.counter = value, 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        """True cuando ya pasaron `patience` épocas sin mejora."""
        return self.patience is not None and self.counter >= self.patience

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """Entrena una época (forward, backpropagation y actualización de pesos); devuelve la pérdida promedio."""
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X), y)          # logits (B,) vs y (B,)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n

@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module | None = None) -> tuple[float | None, np.ndarray, np.ndarray]:
    """Evalúa en modo eval y devuelve (pérdida o None, probabilidades, etiquetas)."""
    model.eval()
    probs, ys, total = [], [], 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        if criterion is not None:
            total += criterion(logits, y).item() * len(y)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.cpu().numpy())
    probs, ys = np.concatenate(probs), np.concatenate(ys)
    return (total / len(ys) if criterion is not None else None), probs, ys

def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig, device: torch.device, checkpoint_path: str | Path | None = None) -> tuple[list[dict], int]:
    """Entrena con early stopping, restaura los mejores pesos y devuelve (historial por época, mejor época)."""
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopping(cfg.patience, cfg.min_delta)
    history, best_state, best_epoch = [], None, 0

    for epoch in range(1, cfg.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        # Métricas de train en modo eval (sin dropout) para compararlas con validación.
        tr_loss, tr_p, tr_y = run_inference(model, train_loader, device, criterion)
        va_loss, va_p, va_y = run_inference(model, val_loader, device, criterion)
        tr, va = compute_metrics(tr_y, tr_p), compute_metrics(va_y, va_p)
        history.append({
            "epoch": epoch,
            "train_loss": float(tr_loss), "val_loss": float(va_loss),
            "train_acc": tr["accuracy"], "val_acc": va["accuracy"],
            "train_f1": tr["f1"], "val_f1": va["f1"],
            "val_pr_auc": va["pr_auc"],
        })
        improved = stopper.step(va_loss)
        if improved:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model, threshold=0.5,
                                extra={"epoch": epoch, "val_loss": float(va_loss)})
        h = history[-1]
        print(f"Época {epoch:3d} | train_loss {h['train_loss']:.4f} val_loss {h['val_loss']:.4f} | "
              f"train_acc {h['train_acc']:.3f} val_acc {h['val_acc']:.3f} val_f1 {h['val_f1']:.3f}"
              f"{'  *' if improved else ''}")
        if stopper.should_stop:
            print(f"Early stopping: sin mejora en {cfg.patience} épocas (mejor época: {best_epoch}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch
