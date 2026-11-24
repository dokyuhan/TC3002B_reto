import argparse
import json
from pathlib import Path
import numpy as np
from model_training import config
from model_training.data.build_matrices import convert
from model_training.data.data_loader import exclude_sources, get_data_info, load_split, make_dataloader
from model_training.data.features import INPUT_SPEC
from model_training.network.checkpoints import save_checkpoint
from model_training.network.cnn_torch import AiCodeCNN
from model_training.network.predictor import Predictor
from model_training.training.metrics import (compute_metrics, format_per_source, format_summary, per_source,
                                             save_metrics, select_threshold)
from model_training.training.trainer import TrainConfig, fit, run_inference
from model_training.utils import get_device, set_seed

def build_data() -> None:
    """Paso Markov: convierte las secuencias AST de dataset/preprocessed en las matrices de dataset/processed."""
    for name, path in convert(config.SEQUENCES_PATH, config.PROCESSED_DIR).items():
        print(f"{name}: {path}")

def resolve_pos_weight(y: np.ndarray) -> float | None:
    """Peso de la clase IA en la pérdida: "auto" equilibra las dos clases (n_humano / n_IA)."""
    if config.POS_WEIGHT == "auto":
        return float((y == 0).sum() / max((y == 1).sum(), 1))
    return None if config.POS_WEIGHT is None else float(config.POS_WEIGHT)

def train() -> None:
    """Entrena la CNN con train/val, elige el umbral en validación y guarda el mejor checkpoint."""
    set_seed(config.SEED)
    device = get_device(config.DEVICE)
    train_data = exclude_sources(load_split(config.TRAIN_PATH), tuple(config.EXCLUDE_TRAIN_SOURCES))
    val_data = exclude_sources(load_split(config.VAL_PATH), tuple(config.EXCLUDE_TRAIN_SOURCES))
    info = get_data_info(train_data["X"])
    if info != get_data_info(val_data["X"]):
        raise ValueError("train y val tienen dimensiones distintas")
    pos_weight = resolve_pos_weight(train_data["y"])
    print(f"Dispositivo: {device} | train={len(train_data['y'])} val={len(val_data['y'])} | {info} | "
          f"IA en train={train_data['y'].mean():.2f} | pos_weight={pos_weight} | sin entrenar con: {list(config.EXCLUDE_TRAIN_SOURCES)}")

    model = AiCodeCNN(**info, conv_channels=config.CONV_CHANNELS, hidden_dim=config.HIDDEN_DIM,
                      dropout=config.DROPOUT, pool_size=config.POOL_SIZE)
    cfg = TrainConfig(batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE, epochs=config.EPOCHS,
                      patience=config.PATIENCE, min_delta=config.MIN_DELTA,
                      weight_decay=config.WEIGHT_DECAY, seed=config.SEED, pos_weight=pos_weight)
    train_loader = make_dataloader(train_data, cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_dataloader(val_data, cfg.batch_size)

    history, best_epoch = fit(model, train_loader, val_loader, cfg, device, config.CHECKPOINT_PATH)

    # Umbral elegido con validación (con los mejores pesos); nunca con test.
    _, val_probs, val_y = run_inference(model, val_loader, device)
    threshold = select_threshold(val_y, val_probs)
    save_checkpoint(config.CHECKPOINT_PATH, model, threshold=threshold,
                    extra={"best_epoch": best_epoch, "val_loss": history[best_epoch - 1]["val_loss"],
                           "input_spec": INPUT_SPEC, "pos_weight": pos_weight,
                           "excluded_sources": list(config.EXCLUDE_TRAIN_SOURCES)})
    print(f"\nMejor época: {best_epoch} | umbral (máx. exactitud balanceada en validación): {threshold:.2f}")
    print(format_summary(compute_metrics(val_y, val_probs, threshold), "Validación"))

    Path(config.HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.HISTORY_PATH).write_text(json.dumps(history, indent=2))
    print(f"\nCheckpoint: {config.CHECKPOINT_PATH} | historial: {config.HISTORY_PATH}")

def evaluate(data_path: str = config.TEST_PATH, out_path: str = config.TEST_METRICS_PATH) -> None:
    """Evalúa el checkpoint en `data_path` y escribe las métricas en JSON, con el desglose por autor si hay `source`."""
    predictor = Predictor(config.CHECKPOINT_PATH, config.DEVICE, config.THRESHOLD)
    test = load_split(data_path)
    probs = predictor.predict_proba(test["X"])
    metrics = compute_metrics(test["y"], probs, predictor.threshold)
    print(format_summary(metrics, f"Evaluación: {data_path}"))
    if "source" in test:
        metrics["per_source"] = per_source(test["source"], test["y"], probs, predictor.threshold)
        print("\n" + format_per_source(metrics["per_source"]))
    save_metrics(out_path, metrics)
    print(f"\nMétricas guardadas en {out_path}")

def predict(index: int | None) -> None:
    """Predice sobre los ejemplos de `PREDICT_INPUT_PATH` (todos, o solo el de `index`)."""
    predictor = Predictor(config.CHECKPOINT_PATH, config.DEVICE, config.THRESHOLD)
    with np.load(config.PREDICT_INPUT_PATH) as f:
        if "X" not in f.files:
            raise KeyError(f"{config.PREDICT_INPUT_PATH} debe contener la clave 'X' (tiene {f.files})")
        X = f["X"][[index]] if index is not None else f["X"]
    probs, labels = predictor.predict(X)

    print(f"Umbral: {predictor.threshold:.2f}")
    for i, (pr, lb) in enumerate(zip(probs[:config.PREDICT_MAX_SHOW], labels[:config.PREDICT_MAX_SHOW])):
        idx = index if index is not None else i
        print(f"ejemplo {idx}: P(IA)={pr:.4f} -> {'CÓDIGO DE IA' if lb else 'HUMANO'}")

def main() -> None:
    """Punto de entrada: elige entre build-data, train, evaluate y predict; todos los valores salen de config.py."""
    p = argparse.ArgumentParser(description="Entrena, evalúa o usa la CNN (la configuración está en config.py).")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build-data", help="genera las matrices de transición en dataset/processed")
    sub.add_parser("train", help="entrena y guarda el mejor checkpoint")
    ev = sub.add_parser("evaluate", help="evalúa el checkpoint sobre test (o sobre otro .npz con --data)")
    ev.add_argument("--data", default=None, help="ruta del .npz a evaluar (por defecto TEST_PATH)")
    ev.add_argument("--out", default=None, help="ruta del JSON de métricas (por defecto TEST_METRICS_PATH)")
    pr = sub.add_parser("predict", help="clasifica ejemplos del archivo PREDICT_INPUT_PATH")
    pr.add_argument("--index", type=int, default=None, help="predecir solo este ejemplo")
    a = p.parse_args()
    if a.command == "build-data":
        build_data()
    elif a.command == "train":
        train()
    elif a.command == "evaluate":
        evaluate(a.data or config.TEST_PATH, a.out or config.TEST_METRICS_PATH)
    else:
        predict(a.index)

if __name__ == "__main__":
    main()
