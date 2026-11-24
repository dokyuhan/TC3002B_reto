"""Logistic regression baseline on flattened 3 channel `(3, K, K)` CNN tensors.

Usage (from repository root):
    uv run --group train python -m model_training.baseline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from model_training import config
from model_training.data.data_loader import load_split
from model_training.training.metrics import compute_metrics, format_summary, select_threshold


def flat(split: dict) -> np.ndarray:
    """Flatten each `(C, K, K)` tensor sample into a `C * K * K` feature vector."""
    return split["X"].reshape(len(split["y"]), -1)


def main() -> None:
    """Fit logistic regression on train, select threshold on val, and evaluate test and IR Plag."""
    train, val = load_split(config.TRAIN_PATH), load_split(config.VAL_PATH)
    model = LogisticRegression(max_iter=2000, random_state=config.SEED)
    model.fit(flat(train), train["y"])
    threshold = select_threshold(val["y"], model.predict_proba(flat(val))[:, 1])
    results = {}
    for name, path in (
        ("val", config.VAL_PATH),
        ("test", config.TEST_PATH),
        ("ir_plag", config.IR_PLAG_PATH),
    ):
        d = load_split(path)
        results[name] = compute_metrics(d["y"], model.predict_proba(flat(d))[:, 1], threshold)
        print(format_summary(results[name], f"Logistic baseline: {name}"), "\n")
    out_path = Path(config.BASELINE_METRICS_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
