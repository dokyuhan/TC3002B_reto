"""Paths and hyperparameters for PyTorch CNN training and evaluation."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Input AST state sequences under dataset/preprocessed
SEQUENCES_PATH = str(ROOT / "dataset/preprocessed/conplag_v2_sequences.npz")
IR_PLAG_SEQUENCES_PATH = str(ROOT / "dataset/preprocessed/ir_plag_sequences.npz")

# Derived CNN matrix splits (regenerated via `build-data`, gitignored)
PROCESSED_DIR = str(ROOT / "dataset/processed")
TRAIN_PATH = f"{PROCESSED_DIR}/train.npz"
VAL_PATH = f"{PROCESSED_DIR}/val.npz"
TEST_PATH = f"{PROCESSED_DIR}/test.npz"
IR_PLAG_PATH = f"{PROCESSED_DIR}/ir_plag.npz"

# Training outputs (gitignored)
CHECKPOINT_PATH = str(ROOT / "model_training/checkpoints/best.pt")
RESULTS_DIR = ROOT / "dataset/results"
HISTORY_PATH = str(RESULTS_DIR / "history.json")
TEST_METRICS_PATH = str(RESULTS_DIR / "test_metrics.json")
IR_PLAG_METRICS_PATH = str(RESULTS_DIR / "ir_plag_metrics.json")
BASELINE_METRICS_PATH = str(RESULTS_DIR / "baseline_metrics.json")

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
MIN_DELTA = 1e-3
SEED = 42
DEVICE = "auto"

# CNN architecture (in_channels and n_states are read from split arrays)
CONV_CHANNELS = (16, 32)
HIDDEN_DIM = 64
DROPOUT = 0.4
POOL_SIZE = 4

# Evaluation and prediction settings (fixed 0.50 operating point for uncalibrated CNN outputs)
THRESHOLD = 0.5
PREDICT_INPUT_PATH = TEST_PATH
PREDICT_MAX_SHOW = 10
