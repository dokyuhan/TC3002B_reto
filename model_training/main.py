"""CLI entrypoint for building CNN matrices, training, evaluating, and predicting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model_training import config
from model_training.data.build_matrices import convert, convert_diagnostic
from model_training.data.data_loader import get_data_info, load_split, make_dataloader
from model_training.data.features import INPUT_SPEC
from model_training.network.checkpoints import save_checkpoint
from model_training.network.cnn_torch import PlagiarismCNN
from model_training.network.predictor import Predictor
from model_training.training.metrics import (
    compute_metrics,
    format_summary,
    save_metrics,
    select_threshold,
)
from model_training.training.trainer import TrainConfig, fit, run_inference
from model_training.utils import get_device, set_seed


def build_data() -> None:
    """Convert AST state sequences under dataset/preprocessed into CNN matrices."""
    for name, path in convert(config.SEQUENCES_PATH, config.PROCESSED_DIR).items():
        print(f"{name}: {path}")
    print(f"ir_plag: {convert_diagnostic(config.IR_PLAG_SEQUENCES_PATH, config.IR_PLAG_PATH)}")


def train() -> None:
    """Train the CNN on train/val, select a validation threshold, and save the best checkpoint."""
    set_seed(config.SEED)
    device = get_device(config.DEVICE)
    train_data, val_data = load_split(config.TRAIN_PATH), load_split(config.VAL_PATH)
    info = get_data_info(train_data["X"])
    if info != get_data_info(val_data["X"]):
        raise ValueError("train and val splits have mismatched dimensions")
    print(f"Device: {device} | train={len(train_data['y'])} val={len(val_data['y'])} | {info}")

    model = PlagiarismCNN(
        **info,
        conv_channels=config.CONV_CHANNELS,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        pool_size=config.POOL_SIZE,
    )
    cfg = TrainConfig(
        batch_size=config.BATCH_SIZE,
        lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
        weight_decay=config.WEIGHT_DECAY,
        seed=config.SEED,
    )
    train_loader = make_dataloader(train_data, cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_dataloader(val_data, cfg.batch_size)

    history, best_epoch = fit(model, train_loader, val_loader, cfg, device, config.CHECKPOINT_PATH)

    _, val_probs, val_y = run_inference(model, val_loader, device)
    threshold = select_threshold(val_y, val_probs)
    save_checkpoint(
        config.CHECKPOINT_PATH,
        model,
        threshold=threshold,
        extra={
            "best_epoch": best_epoch,
            "val_loss": history[best_epoch - 1]["val_loss"],
            "input_spec": INPUT_SPEC,
        },
    )
    print(f"\nBest epoch: {best_epoch} | validation max F1 threshold: {threshold:.2f}")
    print(format_summary(compute_metrics(val_y, val_probs, threshold), "Validation"))

    Path(config.HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.HISTORY_PATH).write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"\nCheckpoint: {config.CHECKPOINT_PATH} | history: {config.HISTORY_PATH}")


def evaluate(data_path: str = config.TEST_PATH, out_path: str = config.TEST_METRICS_PATH) -> None:
    """Evaluate the saved checkpoint with symmetrized inference and write metrics JSON."""
    predictor = Predictor(config.CHECKPOINT_PATH, config.DEVICE, config.THRESHOLD)
    test = load_split(data_path)
    probs = predictor.predict_proba(test["X"], symmetric=True)
    metrics = compute_metrics(test["y"], probs, predictor.threshold)
    print(format_summary(metrics, f"Evaluation: {data_path}"))

    with np.load(data_path) as f:
        levels = f["levels"] if "levels" in f.files else None
    if levels is not None:
        pred = probs >= predictor.threshold
        metrics["per_level"] = {}
        print("\nPer level accuracy:")
        for lv in sorted(set(levels.tolist())):
            m = levels == lv
            acc = float((pred[m] == (test["y"][m] == 1)).mean())
            metrics["per_level"][lv] = {"n": int(m.sum()), "accuracy": acc}
            print(f"  {lv:9s} n={int(m.sum()):4d} accuracy={acc:.3f}")
    save_metrics(out_path, metrics)
    print(f"\nMetrics saved to {out_path}")


def predict(index: int | None) -> None:
    """Predict plagiarism probabilities for samples in `PREDICT_INPUT_PATH`."""
    predictor = Predictor(config.CHECKPOINT_PATH, config.DEVICE, config.THRESHOLD)
    with np.load(config.PREDICT_INPUT_PATH) as f:
        if "X" not in f.files:
            raise KeyError(f"{config.PREDICT_INPUT_PATH} must contain key 'X' (found {f.files})")
        X = f["X"][[index]] if index is not None else f["X"]
    probs, labels = predictor.predict(X, symmetric=True)

    print(f"Threshold: {predictor.threshold:.2f}")
    for i, (pr, lb) in enumerate(
        zip(probs[: config.PREDICT_MAX_SHOW], labels[: config.PREDICT_MAX_SHOW])
    ):
        idx = index if index is not None else i
        label_str = "PLAGIARIZED" if lb else "CLEAN"
        print(f"sample {idx}: P(plagiarism)={pr:.4f} -> {label_str}")


def main() -> None:
    """CLI entrypoint selecting build-data, train, evaluate, or predict."""
    parser = argparse.ArgumentParser(
        description="Build data, train, evaluate, or predict with the plagiarism CNN."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-data", help="generate transition matrices under dataset/processed")
    sub.add_parser("train", help="train CNN and save the best checkpoint")
    ev = sub.add_parser("evaluate", help="evaluate checkpoint on test or custom .npz split")
    ev.add_argument("--data", default=None, help="input .npz path (default: TEST_PATH)")
    ev.add_argument("--out", default=None, help="output metrics JSON path (default: TEST_METRICS_PATH)")
    ev.add_argument(
        "--ir-plag",
        action="store_true",
        help="evaluate IR_PLAG_PATH and save to IR_PLAG_METRICS_PATH",
    )
    pr = sub.add_parser("predict", help="classify samples from PREDICT_INPUT_PATH")
    pr.add_argument("--index", type=int, default=None, help="predict only this sample index")
    args = parser.parse_args()

    if args.command == "build-data":
        build_data()
    elif args.command == "train":
        train()
    elif args.command == "evaluate":
        if args.ir_plag:
            evaluate(config.IR_PLAG_PATH, config.IR_PLAG_METRICS_PATH)
        else:
            evaluate(args.data or config.TEST_PATH, args.out or config.TEST_METRICS_PATH)
    else:
        predict(args.index)


if __name__ == "__main__":
    main()
