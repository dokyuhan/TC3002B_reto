"""Exporta la CNN entrenada a numpy para la app (sin torch) y deja su metadata lista.

Uso (desde la raíz):  python -m model_training.export_numpy
Escribe en models/: ai_detector_weights.npz y ai_detector_metadata.json (no toca los archivos del modelo de pares).

Pasos: 1) copia los pesos de checkpoints/best.pt, 2) comprueba que numpy y torch dan lo mismo,
3) evalúa con los umbrales del modo "balanced" y del modo "low_fpr", 4) guarda umbrales y métricas.
"""
import json
from pathlib import Path
import numpy as np
import torch

from model_training import config
from src.models.cnn_numpy import NumpyCnnClassifier
from model_training.data.data_loader import load_split
from model_training.network.checkpoints import load_checkpoint
from model_training.training.metrics import compute_metrics, per_source, threshold_for_fpr

MODELS_DIR = config.ROOT / "models"
WEIGHTS_NAME = "ai_detector_weights.npz"
METADATA_NAME = "ai_detector_metadata.json"
KEYS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "false_positive_rate", "roc_auc", "pr_auc", "n_samples", "threshold")

def export_weights(state: dict, model_config: dict, path: Path) -> None:
    """Guarda los pesos con nombres simples (índices de capas de AiCodeCNN)."""
    def t(key: str) -> np.ndarray:
        return state[key].numpy().astype(np.float32)

    np.savez(path,
             conv1_w=t("features.0.weight"), conv1_b=t("features.0.bias"),
             conv2_w=t("features.3.weight"), conv2_b=t("features.3.bias"),
             fc1_w=t("classifier.2.weight"), fc1_b=t("classifier.2.bias"),
             fc2_w=t("classifier.5.weight"), fc2_b=t("classifier.5.bias"),
             n_states=np.int64(model_config["n_states"]), pool_size=np.int64(model_config["pool_size"]))

def main() -> None:
    """Exporta, verifica contra torch, calcula umbrales y métricas, y escribe la metadata."""
    model, ckpt = load_checkpoint(config.CHECKPOINT_PATH)
    MODELS_DIR.mkdir(exist_ok=True)
    weights_path = MODELS_DIR / WEIGHTS_NAME
    export_weights(ckpt["state_dict"], ckpt["model_config"], weights_path)
    clf = NumpyCnnClassifier.load(weights_path)

    splits = {"val": load_split(config.VAL_PATH), "test": load_split(config.TEST_PATH)}

    # 1) numpy == torch
    X = splits["test"]["X"]
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(X)).numpy()
    diff = float(np.abs(torch_logits - clf.predict_logits(X)).max())
    print(f"Diferencia máxima torch vs numpy (logits, test): {diff:.2e}")
    assert diff < 1e-4, "numpy y torch no coinciden"

    # 2) umbrales: "balanced" = el de validación guardado en el checkpoint; "low_fpr" = pocas falsas alarmas
    probs = {k: clf.predict_proba(d["X"][:, 0]) for k, d in splits.items()}
    thr_balanced = float(config.THRESHOLD if config.THRESHOLD is not None else ckpt["threshold"])
    thr_low = threshold_for_fpr(splits["val"]["y"], probs["val"], config.LOW_FPR_TARGET)
    print(f"Umbral balanced={thr_balanced:.2f} | low_fpr={thr_low:.2f} (<= {config.LOW_FPR_TARGET:.0%} de humano marcado como IA en validación)")

    metrics = {}
    for mode, thr in (("balanced", thr_balanced), ("low_fpr", thr_low)):
        for name in ("val", "test"):
            m = compute_metrics(splits[name]["y"], probs[name], thr)
            metrics[f"{mode}_{name}"] = {k: m[k] for k in KEYS}
            print(f"  {mode:8s} {name:5s} bal-acc={m['balanced_accuracy']:.3f} recall(IA)={m['recall']:.3f} "
                  f"FPR={m['false_positive_rate']:.3f} AUC={m['roc_auc']:.3f}")
        metrics[f"{mode}_test_per_source"] = per_source(splits["test"]["source"], splits["test"]["y"], probs["test"], thr)

    extra = ckpt.get("extra", {})
    meta = {"model_version": "1.0.0-ai-detector", "task": "human_vs_ai", "unit": "single function", "language": "java",
            "num_states": int(ckpt["model_config"]["n_states"]), "input_channels": int(ckpt["model_config"]["in_channels"]),
            "threshold": thr_balanced, "threshold_low_fpr": thr_low,
            "training": {"best_epoch": extra.get("best_epoch"), "pos_weight": extra.get("pos_weight"),
                         "excluded_sources": extra.get("excluded_sources", [])},
            "dataset": "OSS-forge/HumanVsAICode (Java), CC-BY-SA-4.0",
            "metrics": metrics}
    meta_path = MODELS_DIR / METADATA_NAME
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nGuardado: {weights_path} y {meta_path}")

if __name__ == "__main__":
    main()
