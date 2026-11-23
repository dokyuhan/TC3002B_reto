# Configuración del proyecto: aquí se cambian rutas e hiperparámetros; main.py y src/ no se tocan.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # raíz del repositorio: las rutas funcionan desde cualquier carpeta

# --- Entrada del paso Markov (secuencias AST de dataset/preprocessed) ---
SEQUENCES_PATH = str(ROOT / "dataset/preprocessed/conplag_v2_sequences.npz")   # conplag_v1 o conplag_v2
IR_PLAG_SEQUENCES_PATH = str(ROOT / "dataset/preprocessed/ir_plag_sequences.npz")   # solo split 'diagnostic'

# --- Rutas de los datos (un .npz por partición con las claves X, y y group) ---
PROCESSED_DIR = str(ROOT / "dataset/processed")
TRAIN_PATH = f"{PROCESSED_DIR}/train.npz"
VAL_PATH = f"{PROCESSED_DIR}/val.npz"
TEST_PATH = f"{PROCESSED_DIR}/test.npz"
IR_PLAG_PATH = f"{PROCESSED_DIR}/ir_plag.npz"   # dataset externo, solo para evaluar (nunca entrena)

# --- Rutas de salida ---
CHECKPOINT_PATH = str(ROOT / "model_training/checkpoints/best.pt")
RESULTS_DIR = ROOT / "dataset/results"
HISTORY_PATH = str(RESULTS_DIR / "history.json")
TEST_METRICS_PATH = str(RESULTS_DIR / "test_metrics.json")
IR_PLAG_METRICS_PATH = str(RESULTS_DIR / "ir_plag_metrics.json")
BASELINE_METRICS_PATH = str(RESULTS_DIR / "baseline_metrics.json")

# --- Entrenamiento ---
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15          # épocas sin mejora de val_loss antes de parar; None desactiva el early stopping
MIN_DELTA = 1e-3      # mejora mínima de val_loss para contar como mejora
SEED = 42
DEVICE = "auto"       # auto | cpu | cuda | mps

# --- Arquitectura de la CNN (canales de entrada y K se leen de los datos) ---
CONV_CHANNELS = (16, 32)
HIDDEN_DIM = 64
DROPOUT = 0.4
POOL_SIZE = 4

# --- Evaluación y predicción ---
THRESHOLD = 0.5       # umbral fijo (None usaría el elegido con validación, guardado en el checkpoint)
PREDICT_INPUT_PATH = TEST_PATH
PREDICT_MAX_SHOW = 10
