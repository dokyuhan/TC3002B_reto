"""Exporta la CNN entrenada a numpy para la app (sin torch) y deja su metadata lista.

Uso (desde la raíz):  python -m model_training.export_numpy
Escribe en ../models/: cnn_weights.npz y model_metadata.json (se actualizan threshold y métricas de la CNN).

Pasos: 1) copia los pesos de checkpoints/best.pt, 2) comprueba que numpy y torch dan lo mismo,
3) evalúa la versión simétrica (promedio de (A,B) y (B,A)) que usa la app, 4) fija los umbrales.
"""
import json
from pathlib import Path
import numpy as np
import torch

from model_training import config
from src.models.cnn_numpy import NumpyCnnClassifier
from model_training.data.data_loader import load_split
from model_training.network.checkpoints import load_checkpoint
from model_training.training.metrics import compute_metrics

MODELS_DIR = config.ROOT / "models"
BALANCED_THRESHOLD = 0.5     # el mismo umbral fijo que config.THRESHOLD
LOW_FPR_TARGET = 0.05        # el modo "low_fpr" busca FPR <= 5% en validación

def export_weights(state: dict, model_config: dict, path: Path) -> None:
    """Guarda los pesos con nombres simples (índices de capas de PlagiarismCNN)."""
    def t(key: str) -> np.ndarray:
        return state[key].numpy().astype(np.float32)

    np.savez(path,
             conv1_w=t("features.0.weight"), conv1_b=t("features.0.bias"),
             conv2_w=t("features.3.weight"), conv2_b=t("features.3.bias"),
             fc1_w=t("classifier.2.weight"), fc1_b=t("classifier.2.bias"),
             fc2_w=t("classifier.5.weight"), fc2_b=t("classifier.5.bias"),
             n_states=np.int64(model_config["n_states"]), pool_size=np.int64(model_config["pool_size"]))

def low_fpr_threshold(y: np.ndarray, probs: np.ndarray) -> float:
    """Menor umbral >= BALANCED_THRESHOLD cuyo FPR en validación sea <= LOW_FPR_TARGET."""
    for t in np.arange(BALANCED_THRESHOLD, 1.0, 0.01):
        if compute_metrics(y, probs, t)["false_positive_rate"] <= LOW_FPR_TARGET:
            return float(round(t, 2))
    return 0.99

def main() -> None:
    """Exporta, verifica contra torch, calcula umbrales y métricas, y actualiza la metadata."""
    model, ckpt = load_checkpoint(config.CHECKPOINT_PATH)
    MODELS_DIR.mkdir(exist_ok=True)
    weights_path = MODELS_DIR / "cnn_weights.npz"
    export_weights(ckpt["state_dict"], ckpt["model_config"], weights_path)
    clf = NumpyCnnClassifier.load(weights_path)

    splits = {"val": load_split(config.VAL_PATH), "test": load_split(config.TEST_PATH),
              "ir_plag": load_split(config.IR_PLAG_PATH)}

    # 1) numpy == torch (mismo orden A,B, sin simetrizar)
    X = splits["test"]["X"]
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(X)).numpy()
    diff = float(np.abs(torch_logits - clf.predict_logits(X)).max())
    print(f"Diferencia máxima torch vs numpy (logits, test): {diff:.2e}")
    assert diff < 1e-4, "numpy y torch no coinciden"

    # 2) probabilidades simétricas que usa la app
    probs = {k: clf.predict_pairs(d["X"][:, 0], d["X"][:, 1]) for k, d in splits.items()}
    thr_low = low_fpr_threshold(splits["val"]["y"], probs["val"])
    print(f"Umbral balanced={BALANCED_THRESHOLD} | low_fpr={thr_low} (FPR<={LOW_FPR_TARGET:.0%} en validación)")

    cnn_metrics = {}
    for mode, thr in (("balanced", BALANCED_THRESHOLD), ("low_fpr", thr_low)):
        for name in ("val", "test", "ir_plag"):
            m = compute_metrics(splits[name]["y"], probs[name], thr)
            cnn_metrics[f"{mode}_{name}"] = {k: m[k] for k in
                ("threshold", "n_samples", "accuracy", "precision", "recall", "f1", "false_positive_rate", "pr_auc")}
            print(f"  {mode:8s} {name:8s} acc={m['accuracy']:.3f} recall={m['recall']:.3f} "
                  f"FPR={m['false_positive_rate']:.3f} F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.3f}")

    meta_path = MODELS_DIR / "model_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    keep = ("smoothing_alpha", "num_states", "strip_boilerplate", "supported_languages")
    meta = {k: v for k, v in meta.items() if k in keep}
    meta.update({"model_version": "0.2.0-cnn", "classifier": "cnn",
                 "threshold": BALANCED_THRESHOLD, "threshold_low_fpr": thr_low,
                 "cnn_metrics": cnn_metrics})
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nGuardado: {weights_path} y {meta_path}")

if __name__ == "__main__":
    main()
