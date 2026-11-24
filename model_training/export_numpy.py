"""Export trained PyTorch CNN weights to NumPy `.npz` format and evaluate parity.

Usage (from repository root):
    uv run --group train python -m model_training.export_numpy
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from model_training import config
from model_training.data.data_loader import load_split
from model_training.network.checkpoints import load_checkpoint, normalize_state_dict
from model_training.training.metrics import compute_metrics
from src.models.cnn_numpy import NumpyCnnClassifier

MODELS_DIR = config.ROOT / "models"
# Fixed 0.50 operating point documented for uncalibrated CNN probabilities.
BALANCED_THRESHOLD = 0.5
LOW_FPR_TARGET = 0.05


def export_weights(state: dict, model_config: dict, path: Path) -> None:
    """Export named `PlagiarismCNN` weights to a NumPy `.npz` archive."""
    normalized = normalize_state_dict(state)

    def _arr(key: str) -> np.ndarray:
        return normalized[key].detach().cpu().numpy().astype(np.float32)

    np.savez(
        path,
        conv1_w=_arr("conv1.weight"),
        conv1_b=_arr("conv1.bias"),
        conv2_w=_arr("conv2.weight"),
        conv2_b=_arr("conv2.bias"),
        fc1_w=_arr("fc1.weight"),
        fc1_b=_arr("fc1.bias"),
        fc2_w=_arr("fc2.weight"),
        fc2_b=_arr("fc2.bias"),
        n_states=np.int64(model_config["n_states"]),
        pool_size=np.int64(model_config["pool_size"]),
    )


def low_fpr_threshold(y: np.ndarray, probs: np.ndarray) -> float:
    """Return the smallest threshold >= BALANCED_THRESHOLD with validation FPR <= LOW_FPR_TARGET."""
    for threshold in np.arange(BALANCED_THRESHOLD, 1.0, 0.01):
        if compute_metrics(y, probs, float(threshold))["false_positive_rate"] <= LOW_FPR_TARGET:
            return float(round(threshold, 2))
    return 0.99


def main() -> None:
    """Export weights, verify PyTorch vs NumPy parity, compute symmetric metrics, and save metadata."""
    model, ckpt = load_checkpoint(config.CHECKPOINT_PATH)
    MODELS_DIR.mkdir(exist_ok=True)
    weights_path = MODELS_DIR / "cnn_weights.npz"
    export_weights(ckpt["state_dict"], ckpt["model_config"], weights_path)
    clf = NumpyCnnClassifier.load(weights_path)

    splits = {
        "val": load_split(config.VAL_PATH),
        "test": load_split(config.TEST_PATH),
        "ir_plag": load_split(config.IR_PLAG_PATH),
    }

    X = splits["test"]["X"]
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(X)).numpy()
    diff = float(np.abs(torch_logits - clf.predict_logits(X)).max())
    print(f"Max torch vs numpy logit diff (test): {diff:.2e}")
    assert diff < 1e-4, "NumPy and PyTorch forward passes diverged"

    probs = {k: clf.predict_pairs(d["X"][:, 0], d["X"][:, 1]) for k, d in splits.items()}
    thr_low = low_fpr_threshold(splits["val"]["y"], probs["val"])
    print(
        f"Thresholds: balanced={BALANCED_THRESHOLD} | "
        f"low_fpr={thr_low} (val FPR <= {LOW_FPR_TARGET:.0%})"
    )

    cnn_metrics = {}
    for mode, thr in (("balanced", BALANCED_THRESHOLD), ("low_fpr", thr_low)):
        for name in ("val", "test", "ir_plag"):
            m = compute_metrics(splits[name]["y"], probs[name], thr)
            cnn_metrics[f"{mode}_{name}"] = {
                k: m[k]
                for k in (
                    "threshold",
                    "n_samples",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "false_positive_rate",
                    "pr_auc",
                )
            }
            print(
                f"  {mode:8s} {name:8s} acc={m['accuracy']:.3f} recall={m['recall']:.3f} "
                f"FPR={m['false_positive_rate']:.3f} F1={m['f1']:.3f} PR_AUC={m['pr_auc']:.3f}"
            )

    meta_path = MODELS_DIR / "model_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    keep = ("smoothing_alpha", "num_states", "strip_boilerplate", "supported_languages")
    meta = {k: v for k, v in meta.items() if k in keep}
    meta.update(
        {
            "model_version": "0.2.0-cnn",
            "classifier": "cnn",
            "threshold": BALANCED_THRESHOLD,
            "threshold_low_fpr": thr_low,
            "cnn_metrics": cnn_metrics,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {weights_path} and {meta_path}")


if __name__ == "__main__":
    main()
