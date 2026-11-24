import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

# Clase positiva (1) = código escrito por una IA.
CLASS_NAMES = {0: "humano", 1: "IA"}

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Calcula exactitud (normal y balanceada), precision, recall, F1, FPR, AUC, PR-AUC y matriz de confusión [[TN, FP], [FN, TP]]."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    y_pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = y_true.min() != y_true.max()
    return {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probs)) if has_both else None,
        "pr_auc": float(average_precision_score(y_true, probs)) if has_both else None,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

def select_threshold(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Devuelve el umbral que maximiza la exactitud balanceada (en empate, el más cercano a 0.5); usar solo con validación."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid)
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    scores = np.array([balanced_accuracy_score(y_true, (probs >= t).astype(int)) for t in grid])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])

def threshold_for_fpr(y_true: np.ndarray, probs: np.ndarray, max_fpr: float = 0.05) -> float:
    """Menor umbral con a lo sumo `max_fpr` de código humano marcado como IA (modo de pocas falsas alarmas); usar solo con validación."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    human = probs[y_true == 0]
    for t in np.linspace(0.01, 0.99, 99):
        if (human >= t).mean() <= max_fpr:
            return float(round(t, 2))
    return 0.99

def per_source(sources: np.ndarray, y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Desglose por autor: de cada IA, qué fracción se detecta; del código humano, qué fracción se acepta como humano."""
    sources, y_true = np.asarray(sources), np.asarray(y_true).astype(int)
    pred_ai = np.asarray(probs, dtype=float) >= threshold
    out = {}
    for s in sorted(set(sources.tolist())):
        m = sources == s
        if y_true[m].max() == 0:
            out[s] = {"n": int(m.sum()), "label": "humano", "correct": float((~pred_ai[m]).mean())}
        else:
            out[s] = {"n": int(m.sum()), "label": "IA", "correct": float(pred_ai[m].mean())}
    return out

def format_summary(m: dict, title: str = "Resultados") -> str:
    """Convierte el dict de `compute_metrics` en un texto legible para la terminal."""
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    roc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
    pr_auc = "n/a" if m["pr_auc"] is None else f"{m['pr_auc']:.4f}"
    return "\n".join([
        f"=== {title} (n={m['n_samples']}, umbral={m['threshold']:.2f}) ===",
        f"Accuracy          : {m['accuracy']:.4f}",
        f"Balanced accuracy : {m['balanced_accuracy']:.4f}",
        f"Precision         : {m['precision']:.4f}",
        f"Recall            : {m['recall']:.4f}",
        f"F1-score          : {m['f1']:.4f}",
        f"FPR               : {m['false_positive_rate']:.4f}  (código humano marcado como IA)",
        f"ROC-AUC           : {roc}",
        f"PR-AUC            : {pr_auc}",
        "Matriz de confusión (filas = real, columnas = predicho):",
        "             pred humano  pred IA",
        f"  real humano {tn:11d} {fp:8d}",
        f"  real IA     {fn:11d} {tp:8d}",
        "(Precision/Recall/F1 se calculan para la clase IA)",
    ])

def format_per_source(report: dict) -> str:
    """Texto legible del desglose por autor de `per_source`."""
    lines = ["Por autor (humano: % aceptado como humano | IA: % detectado como IA):"]
    lines += [f"  {s:9s} n={r['n']:5d} {r['label']:7s} acierto={r['correct']:.3f}" for s, r in report.items()]
    return "\n".join(lines)

def save_metrics(path: str | Path, metrics: dict) -> None:
    """Guarda las métricas como JSON, creando la carpeta si no existe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
