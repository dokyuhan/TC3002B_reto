# model_training: how the AI-code detector is built, trained, tested and exported

This folder holds everything needed to **train and test** the CNN that decides whether **one Java function** was written by a human or by an AI. The app does not use this code at runtime: it only uses the weights that `export_numpy.py` copies into `../models/`.

Run every command from the **repository root**, with `uv run --group train` (this installs PyTorch the first time).

```bash
# 0. Download the raw dataset once (see section 2), then:
uv run python -m scripts.build_human_vs_ai                            # 1. raw functions -> AST state sequences
uv run --group train python -m model_training.main build-data         # 2. sequences -> Markov matrices
uv run --group train python -m model_training.main train              # 3. train, keep the best epoch
uv run --group train python -m model_training.main evaluate           # 4. test set + per-author report
uv run --group train python -m model_training.export_numpy            # 5. copy the weights for the app
uv run --group train python -m model_training.baseline                # (optional) logistic baseline
uv run --group train python -m pytest model_training/tests            # (optional) torch vs numpy check
```

## 1. The idea

The model learns **structural habits** that differ between human and AI code.

* A function becomes a sequence of 12 AST states (`IMPORT`, `FUNCTION`, `CLASS`, `ASSIGN`, `IF`, `FOR`, `WHILE`, `CALL`, `RETURN`, `EXCEPTION`, `OP`, `OTHER`), using the parser in `src/preprocessing`.
* The sequence becomes a **12 x 12 Markov transition matrix**: cell `(i, j)` is the probability that state `j` follows state `i`. Each row sums to 1, or is all zeros if the state never appears.
* The CNN reads that matrix (1 channel, shape `(1, 12, 12)`) and outputs one number: the probability that the function was written by an AI.

**Labels.** `y = 1` means AI-written, `y = 0` means human-written. A second array, `source`, records who wrote each sample (`human`, `chatgpt`, `deepseek`, `qwen`). It is never an input to the model; it only lets us report results per author.

**Why a CNN on a transition matrix?** It is a hypothesis, not a fact. Matrix positions are AST categories, not neighbouring pixels, so convolution is not a natural fit. `baseline.py` (logistic regression) exists to show what the CNN adds.

## 2. The dataset

[HumanVsAICode](https://huggingface.co/datasets/OSS-forge/HumanVsAICode) (CC-BY-SA-4.0, Cotroneo, Improta and Liguori, ISSRE 2025). Each row is one Java function with one human implementation (from GitHub, via CodeSearchNet) and up to three AI implementations: ChatGPT (gpt-3.5-turbo), DeepSeek-Coder-Instruct 33B and Qwen2.5-Coder-Instruct 32B. The Java file has 221,796 functions (505 MB). Download it once; it is gitignored:

```bash
mkdir -p dataset/raw/human_vs_ai_code
curl -L -o dataset/raw/human_vs_ai_code/java_dataset.jsonl \
  https://huggingface.co/datasets/OSS-forge/HumanVsAICode/resolve/main/java_dataset.jsonl
```

**Preprocessing (`scripts/build_human_vs_ai.py`)** samples about 20,000 functions and turns every version into one labeled sample:

| Split | Programs | Human | ChatGPT | DeepSeek | Qwen |
|---|---|---|---|---|---|
| train | 48,367 | 14,000 | 13,599 | 11,071 | 9,697 |
| val | 10,387 | 3,000 | 2,914 | 2,362 | 2,111 |
| test | 10,324 | 3,000 | 2,903 | 2,380 | 2,041 |

About 71% of the samples are AI, which is why the training uses class weights and the results use balanced accuracy.

**Leakage control.** The raw file has no repository column, so functions are kept in blocks of 50 consecutive rows, and every block goes entirely into one split.

**Shortcut control.** The raw data has artifacts that would let a model "cheat". These rules apply to every source equally:

* Comments and whitespace never reach the model. Human code has comments in about 1% of samples against 36-62% for AI code, and the AST skips comment nodes.
* Samples that declare their own top-level class are dropped (this was mostly AI output; human methods never do it).
* Methods with an empty body (for example `// Implementation goes here`) are dropped. About 35% of Qwen's outputs were empty stubs.
* Samples with fewer than 5 AST states are dropped.

Raw rows are bare methods, so each one is wrapped in `class W { ... }` before parsing.

## 3. The process, step by step

```
dataset/raw/human_vs_ai_code/java_dataset.jsonl           raw functions (human + 3 AIs)
        |  scripts/build_human_vs_ai.py
        v
dataset/preprocessed/human_vs_ai_java_sequences.npz       AST state sequences + labels + authors
        |  main.py build-data   (data/build_matrices.py)
        v
dataset/processed/{train,val,test}.npz                    X (N,1,12,12), y, group, source
        |  main.py train        (training/trainer.py)
        v
model_training/checkpoints/best.pt                        best weights + saved threshold
dataset/results/history.json                              loss and metrics per epoch
        |  main.py evaluate     (training/metrics.py)
        v
dataset/results/test_metrics.json                         metrics + per-author report
        |  export_numpy.py
        v
models/ai_detector_weights.npz + models/ai_detector_metadata.json
```

**Training (`main.py train`).**

1. Fix the random seed and choose the device (`cuda`, then `mps`, then `cpu`).
2. Load `train.npz` and `val.npz` (optionally without some authors, see `EXCLUDE_TRAIN_SOURCES`).
3. Compute the class weight so human and AI count equally in the loss (`POS_WEIGHT = "auto"` gives n_human / n_AI).
4. Train the CNN with Adam and a weighted binary cross-entropy loss.
5. After each epoch, compute loss, balanced accuracy and AUC on train and validation. Whenever the validation loss improves, save a checkpoint.
6. Stop when the validation loss has not improved for `PATIENCE` epochs, and restore the best weights.
7. Pick the threshold on the validation set that maximizes balanced accuracy, and save it in the checkpoint.
8. Write the per-epoch history to `dataset/results/history.json`.

**Evaluation (`main.py evaluate`).** Predicts on `test.npz`, computes the metrics at the saved threshold, and prints a per-author report: what percent of each AI's code is detected, and what percent of human code is accepted as human.

**Export (`export_numpy.py`).** The app runs the network with numpy, so it does not need PyTorch. This script copies the weights, checks that numpy and PyTorch give the same output (stops if they differ by more than 1e-4), and computes two thresholds: `balanced` (the saved one) and `low_fpr` (at most 5% of human code flagged as AI on validation). It writes both thresholds and all metrics to `models/ai_detector_metadata.json`.

## 4. The network

Defined in `network/cnn_torch.py`. With `CONV_CHANNELS = (16, 32)`, the shapes for one function are:

```
input                    (1, 12, 12)
Conv 3x3, pad 1 + ReLU   (16, 12, 12)
MaxPool 2x2              (16, 6, 6)
Conv 3x3, pad 1 + ReLU   (32, 6, 6)
AdaptiveAvgPool 4x4      (32, 4, 4)
Flatten                  512
Dropout + Linear + ReLU  64
Dropout + Linear         1      <- logit; sigmoid gives P(AI)
```

About 37,700 trainable parameters.

## 5. Configuration (`config.py`)

Paths are computed from the repository root, so the commands work from any folder.

| Setting | Value | Meaning |
|---|---|---|
| `EPOCHS` | 100 | Maximum number of epochs |
| `BATCH_SIZE` | 64 | Samples per update |
| `LEARNING_RATE` | 1e-3 | Adam step size |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `PATIENCE` | 10 | Epochs without improvement before stopping (`None` disables early stopping) |
| `MIN_DELTA` | 1e-4 | Minimum validation-loss drop that counts as an improvement |
| `SEED` | 42 | Seed for reproducible runs |
| `DEVICE` | `auto` | `auto`, `cpu`, `cuda` or `mps` |
| `POS_WEIGHT` | `"auto"` | Weight of the AI class in the loss (`"auto"`, a number, or `None`) |
| `EXCLUDE_TRAIN_SOURCES` | `()` | Authors left out of training, for example `("qwen",)` to test on an AI the model never saw |
| `CONV_CHANNELS` | (16, 32) | Filters in the two convolution layers |
| `HIDDEN_DIM` | 64 | Units in the hidden linear layer |
| `DROPOUT` | 0.4 | Dropout probability |
| `POOL_SIZE` | 4 | Output size of the adaptive pooling |
| `THRESHOLD` | `None` | Decision threshold at evaluation (`None` uses the one chosen on validation) |
| `LOW_FPR_TARGET` | 0.05 | Maximum share of human code flagged as AI in the `low_fpr` mode |

If you change the architecture settings, retrain and export again. A saved `best.pt` only works with the architecture it was trained with.

## 6. Function reference

### `main.py`: command line entry point

| Function | What it does |
|---|---|
| `build_data()` | Converts the AST sequences in `dataset/preprocessed` into the matrices in `dataset/processed`. |
| `resolve_pos_weight(y)` | Returns the AI-class weight for the loss (`"auto"` = number of human samples divided by number of AI samples). |
| `train()` | Runs the training described above and saves `best.pt` and `history.json`. |
| `evaluate(data_path, out_path)` | Evaluates the checkpoint on a `.npz`, prints the metrics and the per-author report, and saves a JSON. |
| `predict(index)` | Prints the probability and label for examples of `PREDICT_INPUT_PATH`, or for one example with `--index`. |
| `main()` | Parses the command (`build-data`, `train`, `evaluate`, `predict`) and calls the right function. |

### `data/build_matrices.py`: sequences to matrices

| Function | What it does |
|---|---|
| `_matrices(tokens, offsets, n_states)` | The sequences are stored as one long array plus offsets. This cuts it into programs and builds one transition matrix per program with `src.features.markov` and no smoothing. Returns `(N, K, K)`. |
| `build_split(npz, split)` | Builds `X` with shape `(N, 1, K, K)` plus `y`, `group` and `source` for one split. |
| `convert(sequences_path, out_dir)` | Writes `train.npz`, `val.npz` and `test.npz`. |

### `data/features.py`: input checks

| Name | What it does |
|---|---|
| `INPUT_SPEC` | Records how the inputs were built (rows normalized). Saved inside the checkpoint. |
| `validate_transition_matrices(P)` | Checks values are finite, in [0, 1], and each row sums to 1 (or 0 if empty). |
| `validate_features(X)` | Checks `X` is `(N, C, K, K)` with square matrices, and every channel is a valid transition matrix. |

### `data/data_loader.py`: loading

| Function | What it does |
|---|---|
| `_load_npz(path)` | Reads a `.npz` and requires `X` and `y` (`group` and `source` are optional). |
| `load_split(path)` | Loads and validates one split: checks the file, the shape of `X`, that `y` has only 0 and 1, and converts to `float32` and `int64`. |
| `exclude_sources(data, names)` | Removes the samples written by the given authors, to test on an AI the model has not seen. |
| `get_data_info(X)` | Returns the number of channels and of states, read from the shape of `X`. |
| `make_dataloader(data, batch_size, shuffle, seed)` | Wraps the data in a PyTorch `DataLoader`. With a seed, the shuffling is reproducible. |

### `data/dataset.py`

| Name | What it does |
|---|---|
| `CodeDataset` | A PyTorch `Dataset` holding `X` and `y` as `float32` tensors. `__len__` returns the number of samples and `__getitem__` returns one `(matrix, label)` pair. |

### `network/cnn_torch.py`: the model

| Name | What it does |
|---|---|
| `AdaptiveAvgPool2dMPS` | An adaptive average pooling that falls back to the CPU on Apple GPUs when the size is not divisible, a case MPS does not support. |
| `AiCodeCNN` | The network of section 4. `forward(x)` takes `(B, C, K, K)` and returns one logit per sample. `from_config(config)` rebuilds a model from the saved configuration. |

### `network/checkpoints.py`: saving and loading

| Function | What it does |
|---|---|
| `_plain(obj)` | Forces plain Python types (through JSON), so checkpoints never contain numpy objects. |
| `save_checkpoint(path, model, threshold, extra)` | Saves the weights, the model configuration, the threshold and extra data (best epoch, class weight, excluded authors). |
| `load_checkpoint(path, device)` | Rebuilds the model in evaluation mode and returns it with the checkpoint dictionary. |

### `network/predictor.py`: using a trained model

| Name | What it does |
|---|---|
| `Predictor(checkpoint_path, device, threshold)` | Loads a checkpoint. Uses the saved threshold unless you pass one (must be between 0 and 1). |
| `Predictor._prepare(X)` | Accepts one example `(C, K, K)` or a batch, validates it, and checks it matches the model. |
| `Predictor.predict_proba(X)` | Returns the probability of AI for each sample. |
| `Predictor.predict(X)` | Returns the probabilities and the 0/1 labels at the threshold. |
| `predict(X, checkpoint_path, ...)` | A one-line shortcut that loads the model and predicts. |

### `training/trainer.py`: the training loop

| Name | What it does |
|---|---|
| `TrainConfig` | The training hyperparameters in one object, including the AI-class weight. |
| `EarlyStopping` | `step(value)` records the validation loss and returns `True` if it improved. `should_stop` becomes `True` after `patience` epochs without improvement. |
| `train_one_epoch(...)` | One pass over the training data: forward, loss, backpropagation, weight update. Returns the average loss. |
| `run_inference(model, loader, device, criterion)` | Evaluates without updating weights. Returns the loss (if a criterion is given), the probabilities and the labels. |
| `fit(...)` | The full loop: trains epoch by epoch with the weighted loss, records the history, saves a checkpoint on every improvement, stops early, and restores the best weights. Returns the history and the best epoch. |

### `training/metrics.py`: measuring results

| Function | What it does |
|---|---|
| `compute_metrics(y_true, probs, threshold)` | Accuracy, balanced accuracy, precision, recall, F1, false positive rate, ROC-AUC, PR-AUC and the confusion matrix `[[TN, FP], [FN, TP]]`. |
| `select_threshold(y_true, probs)` | Finds the threshold with the best balanced accuracy (ties go to the one closest to 0.5). Use it on validation only. |
| `threshold_for_fpr(y_true, probs, max_fpr)` | The lowest threshold that flags at most `max_fpr` of human code as AI. Use it on validation only. |
| `per_source(sources, y_true, probs, threshold)` | For each author: the share of AI code detected, or the share of human code accepted as human. |
| `format_summary(m, title)` | Turns the metrics dictionary into readable text. |
| `format_per_source(report)` | Turns the per-author report into readable text. |
| `save_metrics(path, metrics)` | Writes the metrics as JSON, creating the folder if needed. |

### `utils.py`

| Function | What it does |
|---|---|
| `set_seed(seed)` | Fixes the random seed of Python, numpy and PyTorch, so runs are reproducible. |
| `get_device(name)` | Chooses the device. `auto` prefers CUDA, then MPS, then CPU. |

### `baseline.py`: reference model

| Function | What it does |
|---|---|
| `flat(split)` | Flattens each `(1, 12, 12)` sample into a vector of 144 numbers. |
| `main()` | Trains a class-balanced logistic regression on those vectors, picks its threshold on validation, evaluates on val and test with the per-author report, and saves `dataset/results/baseline_metrics.json`. |

### `export_numpy.py`: from training to the app

| Function | What it does |
|---|---|
| `export_weights(state, model_config, path)` | Saves the layer weights from the PyTorch model into a `.npz`. |
| `main()` | Exports the weights, checks that numpy and PyTorch match, computes the `balanced` and `low_fpr` thresholds, and writes the metrics to `models/ai_detector_metadata.json`. |

### `tests/test_numpy_parity.py`

Builds random PyTorch networks with 1 and 3 input channels, exports them, and checks that the numpy version in `src/models/cnn_numpy.py` gives the same logits. It is skipped when PyTorch is not installed.

## 7. Results

Model with the default configuration, evaluated on the held-out test split (10,324 functions; the test blocks never appear in training). The threshold (0.50) was chosen on validation.

| | CNN | Logistic baseline |
|---|---|---|
| Balanced accuracy | 0.732 | 0.698 |
| ROC-AUC | 0.823 | 0.774 |
| AI code detected (recall) | 0.663 | 0.610 |
| Human code flagged as AI | 0.199 | 0.213 |

Share correctly classified, per author, in the test split:

| Author | CNN | Baseline |
|---|---|---|
| human (accepted as human) | 0.801 | 0.787 |
| ChatGPT (detected) | 0.634 | 0.575 |
| DeepSeek (detected) | 0.829 | 0.792 |
| Qwen (detected) | 0.512 | 0.448 |

The `low_fpr` mode (threshold 0.68) flags 5.0% of human code as AI on test and detects 44.5% of AI code.

**Test with an AI the model never saw.** To check that the model learns general traits of AI code and does not memorize one model's style, I trained a second copy with `EXCLUDE_TRAIN_SOURCES = ("qwen",)` and tested it on the full test split:

| | Qwen seen in training | Qwen never seen |
|---|---|---|
| Qwen code detected | 0.512 | 0.519 |
| Balanced accuracy (all authors) | 0.732 | 0.731 |
| ROC-AUC | 0.823 | 0.820 |

Detection of Qwen does not drop when it is left out of training, so the model is not relying on traits specific to the AIs it saw. Qwen is hard to detect in both cases. This was a single run with one seed and one held-out model, so treat it as supporting evidence, not as proof.

## 8. Known limits

* The detector is **moderately accurate**: about 20% of human functions are flagged as AI at the default threshold, and a third of AI functions are missed. It should be presented as a signal, not as proof of AI authorship.
* The 12 x 12 matrix discards names, comments and formatting on purpose (they are shortcuts in this dataset), so only structure is left, and Qwen's code looks very human structurally.
* The CNN beats the logistic baseline, but a gradient-boosted tree model scored higher on the same 144 numbers in a quick check (ROC-AUC about 0.85 against 0.82). The convolution is not a natural fit for these matrices.
* The training samples are single methods (a median of 30-53 AST tokens), not whole programs, and the AI code was generated from docstrings. Results may not carry over to whole files or to other LLMs.
* There is no repository column in the raw data, so the split uses blocks of consecutive rows; some leakage between neighbouring blocks is still possible.
* Only Java is supported.
