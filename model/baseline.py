"""Línea base: regresión logística sobre las matrices aplanadas, para comparar contra la CNN.

Usa los mismos datos, el mismo umbral (máx. F1 en validación) y las mismas métricas que `main.py`.
Uso: python baseline.py   ->  results/baseline_metrics.json
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
import config
from src.data.data_loader import load_split
from src.training.metrics import compute_metrics, format_summary, select_threshold

def flat(split: dict) -> np.ndarray:
    """Aplana cada muestra (2, K, K) a un vector de 2*K*K valores."""
    return split["X"].reshape(len(split["y"]), -1)

def main() -> None:
    """Entrena la regresión logística con train, elige el umbral con val y evalúa en test e IR Plag."""
    train, val = load_split(config.TRAIN_PATH), load_split(config.VAL_PATH)
    model = LogisticRegression(max_iter=2000, random_state=config.SEED)
    model.fit(flat(train), train["y"])
    threshold = select_threshold(val["y"], model.predict_proba(flat(val))[:, 1])
    results = {}
    for name, path in (("val", config.VAL_PATH), ("test", config.TEST_PATH), ("ir_plag", config.IR_PLAG_PATH)):
        d = load_split(path)
        results[name] = compute_metrics(d["y"], model.predict_proba(flat(d))[:, 1], threshold)
        print(format_summary(results[name], f"Baseline logística: {name}"), "\n")
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
