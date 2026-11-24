# Configuración del proyecto: aquí se cambian rutas e hiperparámetros; main.py y los demás módulos no se tocan.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # raíz del repositorio: las rutas funcionan desde cualquier carpeta

# --- Entrada del paso Markov (secuencias AST de dataset/preprocessed, generadas con scripts/build_human_vs_ai.py) ---
SEQUENCES_PATH = str(ROOT / "dataset/preprocessed/human_vs_ai_java_sequences.npz")

# --- Rutas de los datos (un .npz por partición con las claves X, y, group y source) ---
PROCESSED_DIR = str(ROOT / "dataset/processed")
TRAIN_PATH = f"{PROCESSED_DIR}/train.npz"
VAL_PATH = f"{PROCESSED_DIR}/val.npz"
TEST_PATH = f"{PROCESSED_DIR}/test.npz"

# --- Rutas de salida ---
CHECKPOINT_PATH = str(ROOT / "model_training/checkpoints/best.pt")
RESULTS_DIR = ROOT / "dataset/results"
HISTORY_PATH = str(RESULTS_DIR / "history.json")
TEST_METRICS_PATH = str(RESULTS_DIR / "test_metrics.json")
BASELINE_METRICS_PATH = str(RESULTS_DIR / "baseline_metrics.json")

# --- Entrenamiento ---
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10          # épocas sin mejora de val_loss antes de parar; None desactiva el early stopping
MIN_DELTA = 1e-4      # mejora mínima de val_loss para contar como mejora
SEED = 42
DEVICE = "auto"       # auto | cpu | cuda | mps
POS_WEIGHT = "auto"   # peso de la clase IA en la pérdida: "auto" = n_humano / n_IA, un número, o None
EXCLUDE_TRAIN_SOURCES = ()   # fuentes que NO se usan para entrenar, p. ej. ("qwen",) para probar con una IA nunca vista

# --- Arquitectura de la CNN (canales de entrada y K se leen de los datos) ---
CONV_CHANNELS = (16, 32)
HIDDEN_DIM = 64
DROPOUT = 0.4
POOL_SIZE = 4

# --- Evaluación y predicción ---
THRESHOLD = None      # None usa el umbral elegido en validación (máx. exactitud balanceada), guardado en el checkpoint
LOW_FPR_TARGET = 0.05 # modo "pocas falsas alarmas": umbral con a lo sumo este % de código humano marcado como IA
PREDICT_INPUT_PATH = TEST_PATH
PREDICT_MAX_SHOW = 10
