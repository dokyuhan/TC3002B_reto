import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score)

# Clase positiva (1) = código refactorizado con IA.
CLASS_NAMES = {0: "humano", 1: "IA"}

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Calcula accuracy, precision, recall, F1, FPR, PR-AUC y matriz de confusión [[TN, FP], [FN, TP]]."""
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

def select_threshold(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Devuelve el umbral que maximiza F1 (en empate, el más cercano a 0.5); usar solo con validación."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid)
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    scores = np.array([f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in grid])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])

def format_summary(m: dict, title: str = "Resultados") -> str:
    """Convierte el dict de `compute_metrics` en un texto legible para la terminal."""
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    pr_auc = "n/a" if m["pr_auc"] is None else f"{m['pr_auc']:.4f}"
    return "\n".join([
        f"=== {title} (n={m['n_samples']}, umbral={m['threshold']:.2f}) ===",
        f"Accuracy : {m['accuracy']:.4f}",
        f"Precision: {m['precision']:.4f}",
        f"Recall   : {m['recall']:.4f}",
        f"F1-score : {m['f1']:.4f}",
        f"FPR      : {m['false_positive_rate']:.4f}  (código humano marcado como IA)",
        f"PR-AUC   : {pr_auc}",
        "Matriz de confusión (filas = real, columnas = predicho):",
        "             pred humano  pred IA",
        f"  real humano {tn:11d} {fp:8d}",
        f"  real IA     {fn:11d} {tp:8d}",
        "(Precision/Recall/F1 se calculan para la clase IA)",
    ])

def save_metrics(path: str | Path, metrics: dict) -> None:
    """Guarda las métricas como JSON, creando la carpeta si no existe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
