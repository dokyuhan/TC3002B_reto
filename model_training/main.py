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
from model_training.training.metrics import compute_metrics, format_summary, save_metrics, select_threshold
from model_training.training.trainer import TrainConfig, fit, run_inference
from model_training.utils import get_device, set_seed

def build_data() -> None:
    """Paso Markov: convierte las secuencias AST de dataset/preprocessed en las matrices de dataset/processed."""
    for name, path in convert(config.SEQUENCES_PATH, config.PROCESSED_DIR).items():
        print(f"{name}: {path}")
    print(f"ir_plag: {convert_diagnostic(config.IR_PLAG_SEQUENCES_PATH, config.IR_PLAG_PATH)}")

def train() -> None:
    """Entrena la CNN con train/val, elige el umbral en validación y guarda el mejor checkpoint."""
    set_seed(config.SEED)
    device = get_device(config.DEVICE)
    train_data, val_data = load_split(config.TRAIN_PATH), load_split(config.VAL_PATH)
    info = get_data_info(train_data["X"])
    if info != get_data_info(val_data["X"]):
        raise ValueError("train y val tienen dimensiones distintas")
    print(f"Dispositivo: {device} | train={len(train_data['y'])} val={len(val_data['y'])} | {info}")

    model = PlagiarismCNN(**info, conv_channels=config.CONV_CHANNELS, hidden_dim=config.HIDDEN_DIM,
                          dropout=config.DROPOUT, pool_size=config.POOL_SIZE)
    cfg = TrainConfig(batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE, epochs=config.EPOCHS,
                      patience=config.PATIENCE, min_delta=config.MIN_DELTA,
                      weight_decay=config.WEIGHT_DECAY, seed=config.SEED)
    train_loader = make_dataloader(train_data, cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_dataloader(val_data, cfg.batch_size)

    history, best_epoch = fit(model, train_loader, val_loader, cfg, device, config.CHECKPOINT_PATH)

    # Umbral elegido con validación (con los mejores pesos); nunca con test.
    _, val_probs, val_y = run_inference(model, val_loader, device)
    threshold = select_threshold(val_y, val_probs)
    save_checkpoint(config.CHECKPOINT_PATH, model, threshold=threshold,
                    extra={"best_epoch": best_epoch, "val_loss": history[best_epoch - 1]["val_loss"],
                           "input_spec": INPUT_SPEC})
    print(f"\nMejor época: {best_epoch} | umbral (máx. F1 en validación): {threshold:.2f}")
    print(format_summary(compute_metrics(val_y, val_probs, threshold), "Validación"))

    Path(config.HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.HISTORY_PATH).write_text(json.dumps(history, indent=2))
    print(f"\nCheckpoint: {config.CHECKPOINT_PATH} | historial: {config.HISTORY_PATH}")

def evaluate(data_path: str = config.TEST_PATH, out_path: str = config.TEST_METRICS_PATH) -> None:
    """Evalúa el checkpoint en `data_path` con el umbral guardado y escribe las métricas en JSON.

    Si el archivo trae `levels` (IR Plag), añade un desglose por nivel: recall en los niveles plagiados
    y tasa de acierto en NON_PLAG.
    """
    predictor = Predictor(config.CHECKPOINT_PATH, config.DEVICE, config.THRESHOLD)
    test = load_split(data_path)
    probs = predictor.predict_proba(test["X"])
    metrics = compute_metrics(test["y"], probs, predictor.threshold)
    print(format_summary(metrics, f"Evaluación: {data_path}"))
    with np.load(data_path) as f:
        levels = f["levels"] if "levels" in f.files else None
    if levels is not None:
        pred = probs >= predictor.threshold
        metrics["per_level"] = {}
        print("\nPor nivel (acierto = clasificado según su etiqueta real):")
        for lv in sorted(set(levels.tolist())):
            m = levels == lv
            acc = float((pred[m] == (test["y"][m] == 1)).mean())
            metrics["per_level"][lv] = {"n": int(m.sum()), "accuracy": acc}
            print(f"  {lv:9s} n={int(m.sum()):4d} acierto={acc:.3f}")
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
        print(f"ejemplo {idx}: P(IA)={pr:.4f} -> {'REFACTORIZADO CON IA' if lb else 'HUMANO'}")

def main() -> None:
    """Punto de entrada: elige entre train, evaluate y predict; todos los valores salen de config.py."""
    p = argparse.ArgumentParser(description="Entrena, evalúa o usa la CNN (la configuración está en config.py).")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build-data", help="genera las matrices de transición en dataset/processed")
    sub.add_parser("train", help="entrena y guarda el mejor checkpoint")
    ev = sub.add_parser("evaluate", help="evalúa el checkpoint sobre test (o sobre otro .npz con --data)")
    ev.add_argument("--data", default=None, help="ruta del .npz a evaluar (por defecto TEST_PATH)")
    ev.add_argument("--out", default=None, help="ruta del JSON de métricas (por defecto TEST_METRICS_PATH)")
    ev.add_argument("--ir-plag", action="store_true", help="atajo: evalúa IR_PLAG_PATH y guarda IR_PLAG_METRICS_PATH")
    pr = sub.add_parser("predict", help="clasifica ejemplos del archivo PREDICT_INPUT_PATH")
    pr.add_argument("--index", type=int, default=None, help="predecir solo este ejemplo")
    a = p.parse_args()
    if a.command == "build-data":
        build_data()
    elif a.command == "train":
        train()
    elif a.command == "evaluate":
        if a.ir_plag:
            evaluate(config.IR_PLAG_PATH, config.IR_PLAG_METRICS_PATH)
        else:
            evaluate(a.data or config.TEST_PATH, a.out or config.TEST_METRICS_PATH)
    else:
        predict(a.index)

if __name__ == "__main__":
    main()
