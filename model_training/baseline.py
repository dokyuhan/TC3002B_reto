"""Línea base: regresión logística sobre la matriz aplanada, para comparar contra la CNN.

Usa los mismos datos, el mismo criterio de umbral (máx. exactitud balanceada en validación) y las mismas métricas que `main.py`.
Uso (desde la raíz): python -m model_training.baseline   ->  dataset/results/baseline_metrics.json
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from model_training import config
from model_training.data.data_loader import load_split
from model_training.training.metrics import (compute_metrics, format_per_source, format_summary, per_source,
                                             select_threshold)

def flat(split: dict) -> np.ndarray:
    """Aplana cada muestra (1, K, K) a un vector de K*K valores."""
    return split["X"].reshape(len(split["y"]), -1)

def main() -> None:
    """Entrena la regresión logística con train, elige el umbral con val y evalúa en val y test."""
    train, val = load_split(config.TRAIN_PATH), load_split(config.VAL_PATH)
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=config.SEED)
    model.fit(flat(train), train["y"])
    threshold = select_threshold(val["y"], model.predict_proba(flat(val))[:, 1])
    results = {}
    for name, path in (("val", config.VAL_PATH), ("test", config.TEST_PATH)):
        d = load_split(path)
        probs = model.predict_proba(flat(d))[:, 1]
        results[name] = compute_metrics(d["y"], probs, threshold)
        print(format_summary(results[name], f"Baseline logística: {name}"))
        if "source" in d:
            results[name]["per_source"] = per_source(d["source"], d["y"], probs, threshold)
            print(format_per_source(results[name]["per_source"]), "\n")
    with open(config.BASELINE_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
