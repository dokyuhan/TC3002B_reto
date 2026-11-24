"""Binary classification metrics and formatting for pairwise plagiarism CNN evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLASS_NAMES = {0: "clean", 1: "plagiarized"}


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute accuracy, precision, recall, F1, FPR, PR AUC, and confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    y_pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = y_true.min() != y_true.max()
    return {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "pr_auc": float(average_precision_score(y_true, probs)) if has_both else None,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Return the threshold maximizing F1 on validation (ties broken toward 0.5)."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid)
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    scores = np.array([f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in grid])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])


def format_summary(m: dict, title: str = "Results") -> str:
    """Format a metrics dictionary into a readable terminal summary."""
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    pr_auc = "n/a" if m["pr_auc"] is None else f"{m['pr_auc']:.4f}"
    return "\n".join(
        [
            f"=== {title} (n={m['n_samples']}, threshold={m['threshold']:.2f}) ===",
            f"Accuracy : {m['accuracy']:.4f}",
            f"Precision: {m['precision']:.4f}",
            f"Recall   : {m['recall']:.4f}",
            f"F1 score : {m['f1']:.4f}",
            f"FPR      : {m['false_positive_rate']:.4f}  (clean pairs flagged as plagiarized)",
            f"PR AUC   : {pr_auc}",
            "Confusion matrix (rows = true, cols = predicted):",
            "                  pred clean  pred plagiarized",
            f"  true clean       {tn:9d}  {fp:16d}",
            f"  true plagiarized {fn:9d}  {tp:16d}",
            "(Precision, Recall, and F1 are computed for the plagiarized class)",
        ]
    )


def save_metrics(path: str | Path, metrics: dict) -> None:
    """Write metrics JSON to disk, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
